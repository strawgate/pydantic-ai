from __future__ import annotations

from typing import Any, Callable

from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServer
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ..models import Model
from . import TemporalSettings
from .function_toolset import temporalize_function_toolset
from .mcp_server import temporalize_mcp_server
from .model import temporalize_model


def temporalize_toolset(toolset: AbstractToolset, settings: TemporalSettings | None) -> list[Callable[..., Any]]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        settings: The temporal settings to use.
    """
    if isinstance(toolset, FunctionToolset):
        return temporalize_function_toolset(toolset, settings)
    elif isinstance(toolset, MCPServer):
        return temporalize_mcp_server(toolset, settings)
    else:
        return []


def temporalize_agent(
    agent: Agent,
    settings: TemporalSettings | None = None,
    temporalize_toolset_func: Callable[
        [AbstractToolset, TemporalSettings | None], list[Callable[..., Any]]
    ] = temporalize_toolset,
) -> list[Callable[..., Any]]:
    """Temporalize an agent.

    Args:
        agent: The agent to temporalize.
        settings: The temporal settings to use.
        temporalize_toolset_func: The function to use to temporalize the toolsets.
    """
    if existing_activities := getattr(agent, '__temporal_activities', None):
        return existing_activities

    settings = settings or TemporalSettings()

    # TODO: Doesn't consider model/toolsets passed at iter time.

    activities: list[Callable[..., Any]] = []
    if isinstance(agent.model, Model):
        activities.extend(temporalize_model(agent.model, settings))

    def temporalize_toolset(toolset: AbstractToolset) -> None:
        activities.extend(temporalize_toolset_func(toolset, settings))

    agent.toolset.apply(temporalize_toolset)

    setattr(agent, '__temporal_activities', activities)
    return activities


# TODO: untemporalize_agent
