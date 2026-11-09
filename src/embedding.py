"""Pure-Python embedding layer (a minimal stand-in for torch.nn.Embedding)."""

import math
import random

Matrix = list[list[float]]


def sinusoidal_positional_encoding(
    max_seq_len: int, d_model: int, base: float = 10000.0
) -> Matrix:
    """Fixed positional encoding from 'Attention Is All You Need' (eq. 3.5).

    PE(pos, 2i)   = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))

    Deterministic (no learned weights). Returns a (max_seq_len, d_model)
    matrix, ready to be broadcast-added onto token embeddings.
    """
    # Angle rate per dimension; odd dims reuse the even dim's rate (2i / d_model)
    angle_rates = [base ** (-2 * (i // 2) / d_model) for i in range(d_model)]

    encoding: Matrix = []
    for pos in range(max_seq_len):
        row: list[float] = []
        for i, rate in enumerate(angle_rates):
            angle = pos * rate
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        encoding.append(row)
    return encoding


def get_shape(data) -> tuple[int, ...]:
    """Return the shape of a nested list (mimics tensor.shape)."""
    shape = []
    d = data
    while isinstance(d, list):
        shape.append(len(d))
        if len(d) == 0:
            break
        d = d[0]
    return tuple(shape)


def add_embeddings(token_emb: list[Matrix], pos_emb: Matrix) -> list[Matrix]:
    """Add token + positional embeddings with PyTorch-like broadcasting.

    (batch, seq, dim) + (seq, dim) -> (batch, seq, dim)
    """
    result: list[Matrix] = []
    for batch in token_emb:
        new_batch: list[list[float]] = []
        for i, token_vec in enumerate(batch):
            new_vec = [t + p for t, p in zip(token_vec, pos_emb[i])]
            new_batch.append(new_vec)
        result.append(new_batch)
    return result


class Embedding:
    """Lookup table mapping integer indices to learned float vectors."""

    def __init__(self, vocab_size: int, output_dim: int) -> None:
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.weight: Matrix = [
            [random.gauss(0.0, 1.0) for _ in range(output_dim)]
            for _ in range(vocab_size)
        ]

    def __call__(self, indices: list | list[list[int]]) -> list:
        """Return embeddings for a 1D or 2D list of integer indices."""
        if self._is_1d(indices):
            return [self.weight[i] for i in indices]
        if self._is_2d(indices):
            return [[self.weight[token_id] for token_id in seq] for seq in indices]
        raise ValueError("Indices must be a 1D or 2D list of integers.")

    @staticmethod
    def _is_1d(indices: list) -> bool:
        return bool(indices) and isinstance(indices[0], int)

    @staticmethod
    def _is_2d(indices: list) -> bool:
        return bool(indices) and isinstance(indices[0], list)