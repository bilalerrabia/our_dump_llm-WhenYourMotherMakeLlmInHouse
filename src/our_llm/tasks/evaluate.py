"""Stage 3 — loss / perplexity of a checkpoint (ROADMAP M4).

`train` already reports the numbers it saw while learning; this stage answers the
different question *"what does this checkpoint score now?"* — after the run, on a
fresh process, against the token file on disk. It rebuilds the model from the
checkpoint's own config (so nothing can drift), refuses a vocabulary mismatch
loudly (D11), and reports cross-entropy in both nats and perplexity.
"""

from __future__ import annotations

from dataclasses import dataclass

from our_llm.config import DataConfig

SPLITS = ("train", "val", "both")


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Loss and perplexity per split; ``None`` means "that split was not scored"."""

    checkpoint: str
    step: int
    split: str
    train_loss: float | None = None
    val_loss: float | None = None
    train_perplexity: float | None = None
    val_perplexity: float | None = None

    def __str__(self) -> str:
        parts = []
        if self.train_loss is not None:
            parts.append(f"train {self.train_loss:.4f} (ppl {self.train_perplexity:,.1f})")
        if self.val_loss is not None:
            parts.append(f"val {self.val_loss:.4f} (ppl {self.val_perplexity:,.1f})")
        return f"eval: step {self.step:,}, " + ", ".join(parts) + f" <- {self.checkpoint}"


def evaluate(
    ckpt: str,
    corpus: str = "data",
    granularity: str = "paragraph",
    split: str = "both",
    batch_size: int | None = None,
    eval_iter: int | None = None,
    processed_dir: str = "data/processed",
    cache_dir: str = "data/cache",
    quiet: bool = False,
) -> EvalResult:
    """Report training/validation loss and perplexity for a checkpoint.

    Args:
      ckpt: Checkpoint written by `train`.
      corpus: Corpus to evaluate on; must match the checkpoint's vocabulary.
      granularity: `file` or `paragraph` — the split unit (must match training).
      split: `train`, `val` or `both`.
      batch_size: Override the evaluation batch size.
      eval_iter: Number of batches to average over (None = every batch).
      processed_dir: Where the token file lives.
      cache_dir: Where the vocabulary cache lives.
      quiet: Suppress the per-split report.

    Returns:
      An :class:`EvalResult`; `pipeline` only checks that no exception was raised.
    """
    # Lazy, like `tasks/train.py`: the task layer stays importable without torch.
    from our_llm.data.dataset import build_loaders
    from our_llm.train.checkpoint import check_vocab, load_checkpoint
    from our_llm.train.loss import calc_loss_loader, perplexity
    from our_llm.utils import get_device

    say = (lambda *a, **k: None) if quiet else print
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    device = get_device()
    checkpoint = load_checkpoint(ckpt, map_location=device)
    cfg = checkpoint.cfg

    data_config = DataConfig(
        corpus_path=corpus,
        granularity=granularity,
        num_merges=int(checkpoint.num_merges or DataConfig().num_merges),
        batch_size=batch_size or DataConfig().batch_size,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
    )
    train_loader, val_loader, prepared = build_loaders(
        data_config, context_length=cfg.context_length
    )
    check_vocab(checkpoint, prepared.tokenizer)

    model = checkpoint.model.to(device).eval()
    say(
        f"Checkpoint: {ckpt} (step {checkpoint.step:,}, vocab {cfg.vocab_size:,}, "
        f"{prepared.n_train_tokens:,} train / {prepared.n_val_tokens:,} val tokens)"
    )

    train_loss: float | None = None
    val_loss: float | None = None
    if split in ("train", "both"):
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        say(f"Train : loss {train_loss:.4f}  perplexity {perplexity(train_loss):,.1f}")
    if split in ("val", "both"):
        if val_loader is None:
            say("Val   : skipped (the split is empty -- use --granularity paragraph)")
        else:
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
            say(f"Val   : loss {val_loss:.4f}  perplexity {perplexity(val_loss):,.1f}")

    return EvalResult(
        checkpoint=ckpt,
        step=checkpoint.step,
        split=split,
        train_loss=train_loss,
        val_loss=val_loss,
        train_perplexity=perplexity(train_loss) if train_loss is not None else None,
        val_perplexity=perplexity(val_loss) if val_loss is not None else None,
    )
