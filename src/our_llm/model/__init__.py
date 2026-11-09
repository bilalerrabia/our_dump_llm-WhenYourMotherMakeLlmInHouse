"""Torch model package: layers -> attention -> GPT.

Torch is imported here, so this package is *not* re-exported from
``our_llm/__init__.py`` -- the CLI and the tokenizer must stay usable without it.

    >>> from our_llm.model import GPTModel
    >>> model = GPTModel(get_preset("debug"))
"""

from our_llm.model.attention import MultiHeadAttention
from our_llm.model.gpt import GPTModel, TransformerBlock, sinusoidal_positional_encoding
from our_llm.model.layers import GELU, FeedForward, LayerNorm

__all__ = [
    "GELU",
    "GPTModel",
    "FeedForward",
    "LayerNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "sinusoidal_positional_encoding",
]
