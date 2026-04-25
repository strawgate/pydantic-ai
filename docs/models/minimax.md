# MiniMax

## Install

To use [`MiniMaxProvider`][pydantic_ai.providers.minimax.MiniMaxProvider], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `minimax` optional group:

```bash
pip/uv-add "pydantic-ai-slim[minimax]"
```

## Configuration

To use [MiniMax](https://www.minimax.io/) through their OpenAI-compatible API, sign in to the [MiniMax Platform](https://www.minimax.io/platform) and create an API key.

For a list of available models, see the [MiniMax models documentation](https://www.minimax.io/platform/document/models).

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export MINIMAX_API_KEY='your-api-key'
```

You can then use [`Agent`][pydantic_ai.Agent] by name:

```python
from pydantic_ai import Agent

agent = Agent('minimax:MiniMax-M2')
...
```

Or initialise the model directly with [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.minimax import MiniMaxProvider

model = OpenAIChatModel('MiniMax-M2', provider=MiniMaxProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

You can also customize [`MiniMaxProvider`][pydantic_ai.providers.minimax.MiniMaxProvider] with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.minimax import MiniMaxProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIChatModel(
    'MiniMax-M2',
    provider=MiniMaxProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Provider notes

The MiniMax API is OpenAI-compatible but has a few constraints, surfaced as runtime API errors:

- `temperature` must be in the range `(0.0, 1.0]` — `0` is rejected.
- Structured output via `response_format` (both `json_schema` and `json_object` modes) is not supported. Use [tool output](../output.md#tool-output) for structured results instead.

For thinking-capable models like `MiniMax-M2`, reasoning content is exchanged via the `reasoning_content` field on chat messages and is preserved across turns.
