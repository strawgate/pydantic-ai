from __future__ import annotations

import base64
import contextlib
from asyncio import Lock
from contextlib import AsyncExitStack
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

import pydantic_core
from mcp.types import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool as MCPTool,
)

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import TOOL_SCHEMA_VALIDATOR, messages
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

try:
    from fastmcp.client import Client
    from fastmcp.client.transports import MCPConfigTransport
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import MCPConfig
    from fastmcp.server.server import FastMCP
except ImportError as _import_error:
    raise ImportError(
        'Please install the `fastmcp` package to use the FastMCP server, '
        'you can use the `fastmcp` optional group — `pip install "pydantic-ai-slim[fastmcp]"`'
    ) from _import_error


if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.transports import FastMCPTransport
    from fastmcp.mcp_config import MCPServerTypes


FastMCPToolResult = messages.BinaryContent | dict[str, Any] | str | None

FastMCPToolResults = list[FastMCPToolResult] | FastMCPToolResult


class ToolErrorBehavior(str, Enum):
    """The behavior to take when a tool error occurs."""

    MODEL_RETRY = 'model-retry'
    """Raise a `ModelRetry` containing the tool error message."""

    ERROR = 'raise'
    """Raise the tool error as an exception."""


class FastMCPToolset(AbstractToolset[AgentDepsT]):
    """A toolset that uses a FastMCP client as the underlying toolset."""

    _fastmcp_client: Client[Any] | None = None
    _tool_error_behavior: ToolErrorBehavior

    _tool_retries: int

    _enter_lock: Lock
    _running_count: int
    _exit_stack: AsyncExitStack | None

    def __init__(
        self, fastmcp_client: Client[Any], tool_retries: int = 2, tool_error_behavior: ToolErrorBehavior | None = None
    ):
        self._tool_retries = tool_retries
        self._fastmcp_client = fastmcp_client
        self._enter_lock = Lock()
        self._running_count = 0

        self._tool_error_behavior = tool_error_behavior or ToolErrorBehavior.ERROR

        super().__init__()

    @property
    def id(self) -> str | None:
        return None

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0 and self._fastmcp_client:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.enter_async_context(self._fastmcp_client)
                self._running_count += 1

        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None

        return None

    @property
    def fastmcp_client(self) -> Client[FastMCPTransport]:
        if not self._fastmcp_client:
            msg = 'FastMCP client not initialized'
            raise RuntimeError(msg)

        return self._fastmcp_client

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        mcp_tools: list[MCPTool] = await self.fastmcp_client.list_tools()

        return {
            tool.name: convert_mcp_tool_to_toolset_tool(toolset=self, mcp_tool=tool, retries=self._tool_retries)
            for tool in mcp_tools
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        try:
            call_tool_result: CallToolResult = await self.fastmcp_client.call_tool(name=name, arguments=tool_args)
        except ToolError as e:
            if self._tool_error_behavior == ToolErrorBehavior.MODEL_RETRY:
                raise ModelRetry(message=str(object=e)) from e
            else:
                raise e

        # We don't use call_tool_result.data at the moment because it requires the json schema to be translatable
        # back into pydantic models otherwise it will be missing data.

        return call_tool_result.structured_content or _map_fastmcp_tool_results(parts=call_tool_result.content)

    @classmethod
    def from_fastmcp_server(
        cls, fastmcp_server: FastMCP[Any], tool_error_behavior: ToolErrorBehavior | None = None
    ) -> Self:
        """Build a FastMCPToolset from a FastMCP server.

        Example:
        ```python
        fastmcp_server = FastMCP('my_server')
        @fastmcp_server.tool()
        async def my_tool(a: int, b: int) -> int:
            return a + b

        toolset = FastMCPToolset.from_fastmcp_server(fastmcp_server=fastmcp_server)
        ```
        """
        fastmcp_client: Client[FastMCPTransport] = Client[FastMCPTransport](transport=fastmcp_server)
        return cls(fastmcp_client=fastmcp_client, tool_retries=2, tool_error_behavior=tool_error_behavior)

    @classmethod
    def from_mcp_server(
        cls,
        name: str,
        mcp_server: MCPServerTypes | dict[str, Any],
        tool_error_behavior: ToolErrorBehavior | None = None,
    ) -> Self:
        """Build a FastMCPToolset from an individual MCP server configuration.

        Example:
        ```python
        cls.from_mcp_server(name='my_server', mcp_server={
            'cmd': 'uvx',
            'args': [
                "time-server-mcp",
            ]
        })
        ```
        """
        mcp_config: MCPConfig = MCPConfig.from_dict(config={name: mcp_server})

        return cls.from_mcp_config(mcp_config=mcp_config, tool_error_behavior=tool_error_behavior)

    @classmethod
    def from_mcp_config(
        cls, mcp_config: MCPConfig | dict[str, Any], tool_error_behavior: ToolErrorBehavior | None = None
    ) -> Self:
        """Build a FastMCPToolset from an MCP json-derived / dictionary configuration object.

        Example:
        ```python
        cls.from_mcp_config(mcp_config={
            'mcpServers': {
                'my_server': {
                    'cmd': 'uvx',
                    'args': [
                        "time-server-mcp",
                    ]
                }
            }
        })
        ```
        """
        fastmcp_client: Client[MCPConfigTransport] = Client[MCPConfigTransport](transport=mcp_config)
        return cls(fastmcp_client=fastmcp_client, tool_retries=2, tool_error_behavior=tool_error_behavior)


def convert_mcp_tool_to_toolset_tool(
    toolset: FastMCPToolset[AgentDepsT],
    mcp_tool: MCPTool,
    retries: int,
) -> ToolsetTool[AgentDepsT]:
    """Convert an MCP tool to a toolset tool."""
    return ToolsetTool[AgentDepsT](
        tool_def=ToolDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters_json_schema=mcp_tool.inputSchema,
        ),
        toolset=toolset,
        max_retries=retries,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )


def _map_fastmcp_tool_results(parts: list[ContentBlock]) -> list[FastMCPToolResult]:
    """Map FastMCP tool results to toolset tool results."""
    return [_map_fastmcp_tool_result(part) for part in parts]


def _map_fastmcp_tool_result(part: ContentBlock) -> FastMCPToolResult:
    if isinstance(part, TextContent):
        text = part.text
        if text.startswith(('[', '{')):
            with contextlib.suppress(ValueError):
                result: Any = pydantic_core.from_json(text)
                if isinstance(result, dict | list):
                    return result  # pyright: ignore[reportUnknownVariableType, reportReturnType]
        return text

    if isinstance(part, ImageContent):
        return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)

    if isinstance(part, AudioContent):
        return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)

    if isinstance(part, EmbeddedResource):
        resource = part.resource
        if isinstance(resource, TextResourceContents):
            return resource.text

        # BlobResourceContents
        return messages.BinaryContent(
            data=base64.b64decode(resource.blob),
            media_type=resource.mimeType or 'application/octet-stream',
        )

    msg = f'Unsupported/Unknown content block type: {type(part)}'
    raise ValueError(msg)
