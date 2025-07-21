from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from pydantic_ai._run_context import AgentDepsT, RunContext


class _TemporalRunContext(RunContext[AgentDepsT]):
    _data: dict[str, Any]

    def __init__(self, **kwargs: Any):
        self._data = kwargs
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in kwargs},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            data = super().__getattribute__('_data')
            if name in data:
                return data[name]
            raise e  # TODO: Explain how to make a new run context attribute available

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[AgentDepsT]) -> dict[str, Any]:
        return {
            'deps': ctx.deps,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'retry': ctx.retry,
            'run_step': ctx.run_step,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any]) -> RunContext[AgentDepsT]:
        return cls(**ctx)


@dataclass
class TemporalSettings:
    """Settings for Temporal `execute_activity` and Pydantic AI-specific Temporal activity behavior."""

    # Temporal settings
    task_queue: str | None = None
    schedule_to_close_timeout: timedelta | None = None
    schedule_to_start_timeout: timedelta | None = None
    start_to_close_timeout: timedelta | None = None
    heartbeat_timeout: timedelta | None = None
    retry_policy: RetryPolicy | None = None
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL
    activity_id: str | None = None
    versioning_intent: VersioningIntent | None = None
    summary: str | None = None
    priority: Priority = Priority.default

    # Pydantic AI specific
    tool_settings: dict[str, dict[str, TemporalSettings]] | None = None

    def for_tool(self, toolset_id: str, tool_id: str) -> TemporalSettings:
        if self.tool_settings is None:
            return self
        return self.tool_settings.get(toolset_id, {}).get(tool_id, self)

    serialize_run_context: Callable[[RunContext], Any] = _TemporalRunContext.serialize_run_context
    deserialize_run_context: Callable[[dict[str, Any]], RunContext] = _TemporalRunContext.deserialize_run_context

    @property
    def execute_activity_kwargs(self) -> dict[str, Any]:
        return {
            'task_queue': self.task_queue,
            'schedule_to_close_timeout': self.schedule_to_close_timeout,
            'schedule_to_start_timeout': self.schedule_to_start_timeout,
            'start_to_close_timeout': self.start_to_close_timeout,
            'heartbeat_timeout': self.heartbeat_timeout,
            'retry_policy': self.retry_policy,
            'cancellation_type': self.cancellation_type,
            'activity_id': self.activity_id,
            'versioning_intent': self.versioning_intent,
            'summary': self.summary,
            'priority': self.priority,
        }


def initialize_temporal():
    """Explicitly import types without which Temporal will not be able to serialize/deserialize `ModelMessage`s."""
    from pydantic_ai.messages import (  # noqa F401
        ModelResponse,  # pyright: ignore[reportUnusedImport]
        ImageUrl,  # pyright: ignore[reportUnusedImport]
        AudioUrl,  # pyright: ignore[reportUnusedImport]
        DocumentUrl,  # pyright: ignore[reportUnusedImport]
        VideoUrl,  # pyright: ignore[reportUnusedImport]
        BinaryContent,  # pyright: ignore[reportUnusedImport]
        UserContent,  # pyright: ignore[reportUnusedImport]
    )
