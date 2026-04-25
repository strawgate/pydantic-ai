from __future__ import annotations as _annotations

from . import ModelProfile


def minimax_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MiniMax model."""
    is_thinking = model_name.lower().startswith('minimax-m')
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_thinking,
    )
