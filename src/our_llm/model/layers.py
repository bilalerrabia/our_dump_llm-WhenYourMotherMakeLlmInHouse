"""The small layers every transformer block is made of.

Three layers, each written by hand rather than imported from ``torch.nn``:

* :class:`LayerNorm` normalises **across the embedding dimension** of each token,
  then rescales and shifts with two learned vectors. Unlike BatchNorm it does
  not mix information between tokens in a batch.
* :class:`GELU` is the tanh approximation GPT-2 uses. It is smooth and lets a
  little of the negative side through instead of clamping it to zero.
* :class:`FeedForward` expands the embedding 4x, applies GELU, projects back.
  Attention mixes *tokens*; this mixes *features within a token*.

Each one is checked against its ``torch.nn`` counterpart in ``tests/test_model.py``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: Constants of the tanh approximation: GELU(x) = 0.5x(1 + tanh(c(x + a x^3))).
_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


class LayerNorm(nn.Module):
    """Layer normalisation over the last dimension (biased variance, like GPT-2)."""

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # scale = gamma, shift = beta: the two things the network learns.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # population variance
        normalised = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalised + self.shift


class GELU(nn.Module):
    """Gaussian error linear unit, tanh approximation (what GPT-2 ships)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = _GELU_C * (x + _GELU_A * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))


class FeedForward(nn.Module):
    """Position-wise MLP: ``emb_dim -> expand * emb_dim -> GELU -> emb_dim``."""

    def __init__(self, emb_dim: int, expand: int = 4, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.expand = expand
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, expand * emb_dim),
            GELU(),
            nn.Linear(expand * emb_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
