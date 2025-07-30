from __future__ import annotations

from abc import ABC
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

if TYPE_CHECKING:
    pass


BuildToolsetFunc = Callable[[RunContext[AgentDepsT]], Awaitable[AbstractToolset[AgentDepsT]]]


class DynamicToolset(AbstractToolset[AgentDepsT], ABC):
    """A Toolset that is dynamically built during the Agent run."""

    _build_toolset_fn: BuildToolsetFunc[AgentDepsT]

    _toolset_stack: ContextVar[list[AbstractToolset[AgentDepsT] | None]] = ContextVar('_toolset_stack', default=[])

    def __init__(self, build_toolset_fn: BuildToolsetFunc[AgentDepsT]):
        self._build_toolset_fn = build_toolset_fn

    async def __aenter__(self) -> Self:
        # Add a placeholder to the toolset stack, to be replaced during the first call to get_tools
        self._toolset_stack.set([*self._toolset_stack.get(), None])
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        # Pop the top toolset from the stack, to revert to the previous toolset
        self._toolset_stack.set(self._toolset_stack.get()[:-1])
        return None

    @property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Get the current toolset from the stack."""
        return self._toolset_stack.get()[-1]

    @toolset.setter
    def toolset(self, toolset: AbstractToolset[AgentDepsT] | None):
        """Set the current toolset on the stack."""
        self._toolset_stack.set([*self._toolset_stack.get()[:-1], toolset])

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not self.toolset:
            self.toolset = await self._build_toolset_fn(ctx)

        return await self.toolset.get_tools(ctx=ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await tool.toolset.call_tool(name=name, tool_args=tool_args, ctx=ctx, tool=tool)
