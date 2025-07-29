from __future__ import annotations

from abc import ABC
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.combined import CombinedToolset

if TYPE_CHECKING:
    pass


BuildToolsetFunc = Callable[[RunContext[AgentDepsT]], Awaitable[AbstractToolset[AgentDepsT]]]


class DynamicToolset(AbstractToolset[AgentDepsT], ABC):
    """A Toolset that is dynamically built during an Agent run based on the first available run context."""

    _build_toolset_fn: BuildToolsetFunc[AgentDepsT]

    _dynamic_toolset: ContextVar[CombinedToolset[AgentDepsT]] = ContextVar(
        '_toolset', default=CombinedToolset[AgentDepsT](toolsets=[])
    )
    _token: Token[CombinedToolset[AgentDepsT]] | None = None
    # _toolset_deps: ContextVar[AgentDepsT | None] = ContextVar('_toolset_deps', default=None)

    def __init__(self, build_toolset_fn: BuildToolsetFunc[AgentDepsT]):
        self._build_toolset_fn = build_toolset_fn

    async def __aenter__(self) -> Self:
        # Store the current toolset in a token, so that it can be reset when the context is exited
        self._token = self._dynamic_toolset.set(CombinedToolset[AgentDepsT](toolsets=[]))
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        # Reset the toolset to the previous toolset, so that it can be used again
        if self._token:
            self._dynamic_toolset.reset(self._token)
        self._token = None
        return None

    @property
    def _toolset(self) -> CombinedToolset[AgentDepsT]:
        if not (toolset := self._dynamic_toolset.get()):
            msg = 'Toolset not initialized. Use the `async with` context manager to initialize the toolset.'
            raise RuntimeError(msg)

        return toolset

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if len(self._toolset.toolsets) == 0 or ctx.run_step == 0:
            toolset = await self._build_toolset_fn(ctx)
            self._toolset.toolsets = [toolset]

        return await self._toolset.get_tools(ctx=ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self._toolset.call_tool(name=name, tool_args=tool_args, ctx=ctx, tool=tool)
