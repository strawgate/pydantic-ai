from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool


@dataclass
class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it.

    See [toolset docs](../toolsets.md#wrapping-a-toolset) for more information.
    """

    dynamic_toolset_func: Callable[[], AbstractToolset[AgentDepsT]]

    @asynccontextmanager
    async def setup(self) -> AsyncGenerator[AbstractToolset[AgentDepsT], Any]:
        async with self.dynamic_toolset_func().setup() as dynamic_toolset:
            yield dynamic_toolset

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        raise NotImplementedError('Dynamic toolsets cannot be used to get tools')

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        raise NotImplementedError('Dynamic toolsets cannot be used to call tools')

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        raise NotImplementedError('Dynamic toolsets cannot be used to apply visitors')
