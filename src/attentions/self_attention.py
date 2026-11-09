"""Self-attention with trainable W_query / W_key / W_value.

Q = x Wq, K = x Wk, V = x Wv, weights = softmax(Q K^T / sqrt(d_k)),
context = weights V. Import-safe: see ``main`` for the walkthrough.

Run: uv run python src/attentions/self_attention.py
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


class SelfAttention(nn.Module):
    """Unbatched self-attention (educational; see model/attention.py for the real one)."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        return attn_weights @ values


def main() -> None:
    d_in = INPUTS.shape[1]
    d_out = 4

    # 1. Manual projections with frozen random weights.
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    keys, values, queries = INPUTS @ W_key, INPUTS @ W_value, INPUTS @ W_query
    print("keys/values/queries:", keys.shape, values.shape, queries.shape)

    # 2. Scaled scores, 3. weights, 4. context vectors.
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("attn_weights:\n", attn_weights)
    print("context_vec:\n", attn_weights @ values)

    # The same thing as a module.
    attention = SelfAttention(d_in, d_out)
    print("module context_vec:\n", attention(INPUTS))


if __name__ == "__main__":
    main()
