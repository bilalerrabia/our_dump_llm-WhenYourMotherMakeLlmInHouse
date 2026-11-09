"""Causal multi-head self-attention -- one class, two compute paths.

The idea in one line: every token builds a **query**, every token offers a
**key** and a **value**; a token's output is the softmax-weighted average of the
values it is allowed to look at (itself and everything before it).

Shapes along the way, for input ``(B, T, d_in)``::

    q, k, v          (B, T, d_out)          one projection each
    heads            (B, n_heads, T, head_dim)   d_out split and moved next to B
    scores           (B, n_heads, T, T)     q @ k^T / sqrt(head_dim)
    causal mask      upper triangle -> -inf  (token t cannot see t+1 .. T)
    weights          (B, n_heads, T, T)     softmax over the last axis
    context          (B, n_heads, T, head_dim)
    output           (B, T, d_out)          heads concatenated, out_proj applied

Two paths compute the same thing:

* ``use_sdpa=True`` delegates to ``torch.nn.functional.scaled_dot_product_attention``
  (fused kernel, the fast one).
* ``use_sdpa=False`` writes the softmax out by hand (the readable one, and the
  reference the fast path is tested against).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Masked self-attention with ``num_heads`` independent heads."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads:
            raise ValueError(f"d_out ({d_out}) must be divisible by num_heads ({num_heads})")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.use_sdpa = use_sdpa

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        # Not a parameter and not derived state stored in checkpoints: it is a
        # pure function of context_length, so persistent=False keeps the .pt
        # small and makes the mask follow the module to any device.
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        if num_tokens > self.mask.shape[0]:
            raise ValueError(
                f"sequence length {num_tokens} exceeds context_length {self.mask.shape[0]}"
            )

        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        if self.use_sdpa:
            # dropout only while training; is_causal builds the same mask for us
            context = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            context = weights @ values

        # (B, H, T, head_dim) -> (B, T, d_out)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return x.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
