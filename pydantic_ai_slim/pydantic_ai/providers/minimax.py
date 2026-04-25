from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.minimax import minimax_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the MiniMax provider, '
        'you can use the `minimax` optional group — `pip install "pydantic-ai-slim[minimax]"`'
    ) from _import_error


MiniMaxModelName = Literal[
    'MiniMax-M2',
    'MiniMax-Text-01',
    'abab6.5s-chat',
    'abab6.5g-chat',
]


class MiniMaxProvider(Provider[AsyncOpenAI]):
    """Provider for MiniMax API.

    MiniMax exposes an OpenAI-compatible chat completions API. Note the following
    constraints, which are surfaced via API errors at request time:

    - `temperature` must be in the range `(0.0, 1.0]` — `0` is rejected by the API.
    - Structured output via `response_format` is not supported.
    """

    @property
    def name(self) -> str:
        return 'minimax'

    @property
    def base_url(self) -> str:
        return 'https://api.minimax.io/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = minimax_model_profile(model_name)

        # MiniMax's API is OpenAI-compatible, so we use the OpenAIJsonSchemaTransformer
        # for tool schemas. MiniMax does not support `response_format`-based JSON output,
        # so both `supports_json_schema_output` and `supports_json_object_output` are left
        # at the `ModelProfile` default of `False`.
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        ).update(profile)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new MiniMax provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the
                `MINIMAX_API_KEY` environment variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided,
                `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('MINIMAX_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MINIMAX_API_KEY` environment variable or pass it via `MiniMaxProvider(api_key=...)`'
                ' to use the MiniMax provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
