"""Document-level train/val split — pure Python, no torch.

Split **by document, never by token percentage**: cutting a token stream at 90 %
would put the second half of a sentence in training and its first half in
validation, and the model would look better than it is. Whole documents go to
one side or the other.

This lives in its own module (rather than in ``data/dataset.py``) so
``our-llm tokenize`` can report the split without importing torch.
"""

from __future__ import annotations

import random


def split_documents(
    n_documents: int, val_ratio: float = 0.10, seed: int = 123
) -> tuple[list[int], list[int]]:
    """Shuffle document indices and cut ``val_ratio`` of them off the end.

    Returns ``(train_indices, val_indices)``: disjoint, sorted, and deterministic
    for a given ``seed``. With fewer than two documents there is nothing to
    validate on, so everything stays in training.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    indices = list(range(n_documents))
    if n_documents < 2 or val_ratio == 0.0:
        return indices, []

    random.Random(seed).shuffle(indices)
    n_val = max(1, round(n_documents * val_ratio))
    n_val = min(n_val, n_documents - 1)  # never hand everything to validation
    return sorted(indices[n_val:]), sorted(indices[:n_val])
