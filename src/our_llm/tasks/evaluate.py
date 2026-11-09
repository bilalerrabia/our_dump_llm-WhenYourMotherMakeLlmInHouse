"""Stage 3 — loss / perplexity of a checkpoint (ROADMAP M4).

Same story as :mod:`our_llm.tasks.train`: the signature is the contract, the
body arrives with `src/our_llm/train/loss.py` (`calc_loss_loader`, `perplexity`).
"""

from __future__ import annotations


def evaluate(
    ckpt: str,
    corpus: str = "data",
    granularity: str = "paragraph",
    split: str = "both",
    batch_size: int | None = None,
    eval_iter: int | None = None,
) -> None:
    """Report training/validation loss and perplexity for a checkpoint.

    Args:
      ckpt: Checkpoint written by `train`.
      corpus: Corpus to evaluate on; must match the checkpoint's vocabulary.
      granularity: `file` or `paragraph` — the split unit (must match training).
      split: `train`, `val` or `both`.
      batch_size: Override the evaluation batch size.
      eval_iter: Number of batches to average over.
    """
    raise NotImplementedError(
        "eval: cross-entropy + perplexity land in M4 (src/our_llm/train/loss.py). "
        f"Asked for {ckpt!r} on the {split} split of {corpus!r}."
    )
