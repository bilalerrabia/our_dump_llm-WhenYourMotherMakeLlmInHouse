"""Causal (masked) self-attention: a token may only look at the past.

The upper-triangular mask is registered as a buffer, truncated to the number of
tokens actually present, filled with -inf and then removed by the softmax.

Run: uv run python src/attentions/causel_attention.py
(NB: the filename typo is legacy; the real module lives at our_llm/model/attention.py.)
"""

import torch
from torch import nn

INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89, 0.12],  # Your
        [0.55, 0.87, 0.66, 0.12],  # journey
        [0.57, 0.85, 0.64, 0.12],  # starts
        [0.22, 0.58, 0.33, 0.12],  # with
        [0.77, 0.25, 0.10, 0.12],  # one
        [0.05, 0.80, 0.55, 0.12],  # step
    ]
)


class CausalAttention(nn.Module):
    """Single-head causal self-attention over a batch ``(b, num_tokens, d_in)``."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # `:num_tokens` keeps short sequences working when context_length is larger.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


def main() -> None:
    batch = torch.stack((INPUTS, INPUTS), dim=0)
    print("batch.shape:", batch.shape)

    d_in = d_out = batch.shape[-1]
    context_length = batch.shape[1]
    torch.manual_seed(123)
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
