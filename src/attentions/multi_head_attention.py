"""Multi-head causal self-attention.

One ``d_out`` projection per Q/K/V, then the last dimension is unrolled into
``num_heads`` heads: (b, T, d_out) -> (b, T, n_heads, head_dim) -> (b, n_heads, T, head_dim).
Heads are computed independently, concatenated and projected with ``out_proj``.

Run: uv run python src/attentions/multi_head_attention.py
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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # projection dim per head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # combine the head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, T, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the projection dim into heads, then move heads next to the batch.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with a causal mask, per head.
        attn_scores = queries @ keys.transpose(2, 3)  # (b, n_heads, T, T)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, n_heads, T, head_dim) -> (b, T, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


def main() -> None:
    batch = torch.stack((INPUTS, INPUTS), dim=0)
    torch.manual_seed(123)

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
