from typing import Callable

from httpx import AsyncBaseTransport, BaseTransport, Request, Response
from tenacity import AsyncRetrying, Retrying


class TenacityTransport(BaseTransport):
    def __init__(self, wrapped: BaseTransport, controller: Retrying, validate_response: Callable[[Response], None] | None = None):
        self.wrapped = wrapped
        self.controller = controller
        self.validate_response = validate_response

    def handle_request(self, request: Request) -> Response:
        for attempt in self.controller:
            with attempt:
                response = self.wrapped.handle_request(request)
                if self.validate_response:
                    self.validate_response(response)
                return response
        raise RuntimeError('The retry controller did not make any attempts')


class AsyncTenacityTransport(AsyncBaseTransport):
    def __init__(self, wrapped: AsyncBaseTransport, controller: AsyncRetrying, validate_response: Callable[[Response], None] | None = None):
        self.wrapped = wrapped
        self.controller = controller
        self.validate_response = validate_response

    async def handle_async_request(self, request: Request) -> Response:
        async for attempt in self.controller:
            with attempt:
                response = await self.wrapped.handle_async_request(request)
                if self.validate_response:
                    self.validate_response(response)
                return response
        raise RuntimeError('The retry controller did not make any attempts')
