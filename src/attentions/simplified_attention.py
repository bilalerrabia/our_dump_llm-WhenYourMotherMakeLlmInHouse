"""Simplified attention: dot products -> softmax -> context vectors.

Teaching demo (nothing runs at import). Vocabulary: "Your journey starts with
one step", each token a 3-dim vector.

Run: uv run python src/attentions/simplified_attention.py
"""

import torch

INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)
WORDS = ["Your", "journey", "starts", "with", "one", "step"]


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """Softmax without the max-subtraction trick (overflow-prone, educational)."""
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def main() -> None:
    # 1. One query against every input: attention *scores* are dot products.
    query = INPUTS[1]  # "journey"
    scores = torch.stack([torch.dot(x_i, query) for x_i in INPUTS])
    print("attention scores:", scores)

    # 2. Normalisation (naive, then the real softmax).
    print("scores/sum    :", scores / scores.sum())
    print("softmax_naive :", softmax_naive(scores))
    weights = torch.softmax(scores, dim=0)
    print("softmax       :", weights, "sum:", weights.sum().item())

    # 3. Context vector = weighted sum of the inputs.
    context_vec = torch.zeros(query.shape)
    for i, x_i in enumerate(INPUTS):
        context_vec += weights[i] * x_i
    print("context vector:", context_vec)

    # 4. The same for all queries at once: (6,6) scores -> (6,3) contexts.
    attn_scores = INPUTS @ INPUTS.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("attn_weights :", attn_weights, sep="\n")
    print("row sums     :", attn_weights.sum(dim=-1))
    print("all contexts :", attn_weights @ INPUTS, sep="\n")


if __name__ == "__main__":
    main()
