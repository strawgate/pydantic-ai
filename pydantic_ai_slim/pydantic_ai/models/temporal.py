from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, AsyncIterable, Awaitable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from pydantic_ai.messages import ModelResponse, AgentStreamEvent, HandleResponseEvent
from .. import RunContext

from ..messages import (
    ModelMessage,
)
from ..settings import ModelSettings
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

__all__ = ('TemporalModel',)


@dataclass
class TemporalSettings:
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


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class ModelRequestParams:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


@dataclass
class TemporalMiddleware[AgentDepsT]:
    settings: TemporalSettings | None = None
    stream_handler: Callable[[RunContext[AgentDepsT], AsyncIterable[AgentStreamEvent | HandleResponseEvent]], Awaitable[None]] | None = None

    def wrap_model(self, model: Model):
        return TemporalModel(model, self.settings)


@dataclass
class TemporalModel(WrapperModel):
    settings: TemporalSettings

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        options: TemporalSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.settings = options or TemporalSettings()

        @activity.defn
        async def request_activity(params: ModelRequestParams) -> ModelResponse:
            return await self.wrapped.request(params.messages, params.model_settings, params.model_request_parameters)

        self.request_activity = request_activity

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.request_activity,
            arg=ModelRequestParams(
                messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
            ),
            **self.settings.__dict__,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        raise NotImplementedError('Use the ')
        yield


def initialize_temporal():
    from pydantic_ai.messages import ModelResponse  # noqa F401  # pyright: ignore[reportUnusedImport]
