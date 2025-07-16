from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from tenacity import AsyncRetrying

from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings


@dataclass(init=False)
class RetryModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model | KnownModelName,
        controller: AsyncRetrying | None = None,
        stream_controller: AsyncRetrying | Literal[False] | None = None,
    ):
        super().__init__(wrapped)
        self.controller = controller
        self.stream_controller = controller if stream_controller is None else stream_controller

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        async for attempt in self.controller:
            with attempt:
                return await super().request(messages, model_settings, model_request_parameters)
        raise RuntimeError('The retry controller did not make any attempts')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        if not self.stream_controller:
            # No special retrying logic for streaming in this case:
            async with super().request_stream(messages, model_settings, model_request_parameters) as stream:
                yield stream
                return

        entered_stream = False
        async for attempt in self.controller:
            attempt.__enter__()
            try:
                async with super().request_stream(messages, model_settings, model_request_parameters) as stream:
                    entered_stream = True
                    attempt.__exit__(None, None, None)
                    yield stream
                    return
            finally:
                if not entered_stream:
                    attempt.__exit__(None, None, None)
        raise RuntimeError('The retry controller did not make any attempts')
