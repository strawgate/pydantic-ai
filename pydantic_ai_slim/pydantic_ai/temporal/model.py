from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from ..messages import (
    ModelMessage,
    ModelResponse,
)
from ..models import Model, ModelRequestParameters, StreamedResponse
from ..settings import ModelSettings
from . import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


def temporalize_model(model: Model, settings: TemporalSettings | None = None) -> list[Callable[..., Any]]:
    """Temporalize a model.

    Args:
        model: The model to temporalize.
        settings: The temporal settings to use.
    """
    if activities := getattr(model, '__temporal_activities', None):
        return activities

    settings = settings or TemporalSettings()

    original_request = model.request

    @activity.defn(name='model_request')
    async def request_activity(params: _RequestParams) -> ModelResponse:
        return await original_request(params.messages, params.model_settings, params.model_request_parameters)

    async def request(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=request_activity,
            arg=_RequestParams(
                messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
            ),
            **settings.execute_activity_kwargs,
        )

    @asynccontextmanager
    async def request_stream(
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        raise NotImplementedError('Cannot stream with temporal yet')
        yield

    model.request = request
    model.request_stream = request_stream

    activities = [request_activity]
    setattr(model, '__temporal_activities', activities)
    return activities


# @dataclass
# class TemporalModel(WrapperModel):
#     temporal_settings: TemporalSettings

#     def __init__(
#         self,
#         wrapped: Model | KnownModelName,
#         temporal_settings: TemporalSettings | None = None,
#     ) -> None:
#         super().__init__(wrapped)
#         self.temporal_settings = temporal_settings or TemporalSettings()

#         @activity.defn
#         async def request_activity(params: ModelRequestParams) -> ModelResponse:
#             return await self.wrapped.request(params.messages, params.model_settings, params.model_request_parameters)

#         self.request_activity = request_activity

#     async def request(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> ModelResponse:
#         return await workflow.execute_activity(
#             activity=self.request_activity,
#             arg=ModelRequestParams(
#                 messages=messages, model_settings=model_settings, model_request_parameters=model_request_parameters
#             ),
#             **self.temporal_settings.__dict__,
#         )

#     @asynccontextmanager
#     async def request_stream(
#         self,
#         messages: list[ModelMessage],
#         model_settings: ModelSettings | None,
#         model_request_parameters: ModelRequestParameters,
#     ) -> AsyncIterator[StreamedResponse]:
#         raise NotImplementedError('Cannot stream with temporal yet')
#         yield
