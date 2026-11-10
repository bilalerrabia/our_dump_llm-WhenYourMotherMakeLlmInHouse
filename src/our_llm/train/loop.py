"""The pretraining loop: one AdamW step at a time, evaluated and checkpointed.

The recipe (ROADMAP D10) is intentionally small but not naive:

* **AdamW with two parameter groups.** Weight decay applies to matrices
  (:math:`\\geq` 2-D) only; LayerNorm scales and biases are 1-D, and decaying them
  just pulls them towards zero for no reason. The book skips this distinction.
* **Warmup then cosine decay.** The first steps have huge gradients; ramping the
  LR up over `warmup_steps` avoids blowing the weights up, and the cosine tail
  lets the model settle. Both live in :func:`lr_at_step`, a pure function so a
  test can pin the curve without training.
* **Gradient clipping at 1.0** per step -- cheap insurance against a single bad
  batch.
* **Evaluate every `eval_freq` steps** on `eval_iter` validation batches, write a
  CSV row, and keep the checkpoint with the best validation loss. `best.pt` is
  what `eval`/`sample` consume; `last.pt` is what `--resume` needs.

The loop never imports the tokenizer or decodes text: `sample_fn(model)` is an
optional callback (M5 plugs the real sampler in). That keeps `train/loop.py`
testable without a corpus.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from our_llm.config import TrainConfig
from our_llm.train.checkpoint import save_checkpoint
from our_llm.train.loss import calc_loss_batch, calc_loss_loader, perplexity

CSV_COLUMNS = ("step", "epoch", "train_loss", "val_loss", "lr", "best_val")

#: Row-per-evaluation history, shared by the CSV log and the checkpoint.
Row = dict[str, float | int]


def lr_at_step(
    step: int,
    base_learning_rate: float,
    total_steps: int,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup to ``base_learning_rate``, then cosine decay to ``min_lr_ratio``.

    ``step`` is 0-based: the very first update runs at ``base/warmup_steps`` so an
    AdamW with a large init moment does not jump out of the basin.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def build_optimizer(model: nn.Module, config: TrainConfig | None = None) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (never on norms or biases)."""
    config = config or TrainConfig()
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def new_run_dir(base: str = "runs", now: datetime | None = None) -> str:
    """``runs/<YYYYmmdd-HHMMSS>``, suffixed if two runs start in the same second."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, stamp)
    suffix = 2
    while os.path.exists(path):
        path = os.path.join(base, f"{stamp}-{suffix}")
        suffix += 1
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class TrainHistory:
    """Everything the loop learned about its own trajectory."""

    rows: list[Row] = field(default_factory=list)
    best_val: float = float("inf")
    best_step: int = 0
    best_epoch: int = 0
    checkpoint: str | None = None
    last_checkpoint: str | None = None
    global_step: int = 0
    epochs_done: int = 0
    seconds: float = 0.0

    @property
    def final_train_loss(self) -> float | None:
        for row in reversed(self.rows):
            value = row.get("train_loss")
            if value is not None and math.isfinite(float(value)):
                return float(value)
        return None

    @property
    def final_val_loss(self) -> float | None:
        for row in reversed(self.rows):
            value = row.get("val_loss")
            if value is not None and math.isfinite(float(value)):
                return float(value)
        return None

    def as_dict(self) -> dict:
        """The JSON-friendly half, stored inside every checkpoint."""
        return {
            "rows": self.rows,
            "best_val": self.best_val,
            "best_step": self.best_step,
            "best_epoch": self.best_epoch,
            "global_step": self.global_step,
            "epochs_done": self.epochs_done,
        }

    def __str__(self) -> str:
        tail = f", best val {self.best_val:.4f} @ step {self.best_step}" if math.isfinite(
            self.best_val
        ) else ""
        return (
            f"train: {self.global_step} steps in {self.epochs_done} epoch(s) "
            f"({self.seconds:.1f}s){tail} -> {self.checkpoint or 'no checkpoint'}"
        )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig | None = None,
    *,
    tokenizer: object = None,
    checkpoint_dir: str | None = None,
    log_path: str | None = None,
    sample_fn: Callable[[nn.Module], str] | None = None,
    quiet: bool = False,
    start_epoch: int = 0,
    global_step: int = 0,
    best_val: float | None = None,
    history: dict | None = None,
) -> TrainHistory:
    """Train for ``config.max_epochs`` epochs from ``start_epoch``.

    Args:
      model: The GPT model (moved to ``device`` here).
      train_loader: Shuffled next-token batches.
      val_loader: Held-out batches; ``None`` means "watch the training loss".
      optimizer: Usually from :func:`build_optimizer`; its state is checkpointed.
      device: Where batches go.
      config: :class:`~our_llm.config.TrainConfig` with the recipe.
      tokenizer: Stored in the checkpoint for the vocabulary fingerprint guard.
      checkpoint_dir: Where ``best.pt`` / ``last.pt`` are written (``None`` = nowhere).
      log_path: CSV file, appended per evaluation.
      sample_fn: Called once per epoch with the model; return text to print.
      quiet: Suppress per-evaluation lines.
      start_epoch: Resume point (epochs are counted from here).
      global_step: Resume point for the LR schedule and the step counter.
      best_val: Best validation loss so far (``None`` -> ``inf``).
      history: ``history.as_dict()`` of an earlier run, so the CSV stays complete.
    """
    config = config or TrainConfig()
    say = (lambda *a, **k: None) if quiet else print
    model.to(device)

    resume = dict(history or {})
    state = TrainHistory(
        rows=list(resume.get("rows") or []),
        best_val=float(best_val) if best_val is not None else float("inf"),
        best_step=int(resume.get("best_step", global_step)),
        best_epoch=int(resume.get("best_epoch", start_epoch)),
        global_step=global_step,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, config.max_epochs * steps_per_epoch + global_step)
    recent: deque[float] = deque(maxlen=max(1, config.eval_freq))
    started = time.perf_counter()

    def evaluate_and_log(epoch: int, lr: float) -> float:
        """Average the recent training losses, score val, log, maybe checkpoint."""
        train_loss = sum(recent) / len(recent) if recent else float("nan")
        if val_loader is not None:
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=config.eval_iter
            )
        else:
            val_loss = train_loss

        metric = val_loss if math.isfinite(val_loss) else train_loss
        improved = metric < state.best_val
        if improved:
            state.best_val = metric
            state.best_step = state.global_step
            state.best_epoch = epoch
            if checkpoint_dir:
                state.checkpoint = save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    model,
                    optimizer=optimizer,
                    step=state.global_step,
                    epoch=epoch,
                    tokenizer=tokenizer,  # type: ignore[arg-type]
                    history=state.as_dict(),
                )

        row: Row = {
            "step": state.global_step,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "best_val": round(state.best_val, 6),
        }
        state.rows.append(row)
        if log_path:
            append_csv_row(log_path, row)

        ppl = perplexity(val_loss)
        marker = " *" if improved else ""
        say(
            f"step {state.global_step:>6,} | epoch {epoch + 1}/{config.max_epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | ppl {ppl:>9,.1f} | "
            f"lr {lr:.2e} | best {state.best_val:.4f}{marker}"
        )
        return val_loss

    say(
        f"Training: {steps_per_epoch:,} batches/epoch, {config.max_epochs} epoch(s), "
        f"lr {config.learning_rate}, batch {getattr(train_loader, 'batch_size', '?')}, "
        f"device {device}"
    )

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        for batch_index, (inputs, targets) in enumerate(train_loader):
            lr = lr_at_step(
                state.global_step,
                config.learning_rate,
                total_steps,
                config.warmup_steps,
                config.min_lr_ratio,
            )
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(model, inputs, targets, device)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            state.global_step += 1
            recent.append(loss.item())

            # The last batch of an epoch is scored by the epoch-end evaluation
            # below, so don't log the same step twice.
            is_last_batch = batch_index == steps_per_epoch - 1
            if state.global_step % config.eval_freq == 0 and not is_last_batch:
                evaluate_and_log(epoch, lr)

        # Always close an epoch with a logged evaluation, even when the epoch was
        # shorter than eval_freq -- otherwise a 1-batch corpus logs nothing.
        evaluate_and_log(epoch, optimizer.param_groups[0]["lr"])
        state.epochs_done += 1

        if sample_fn is not None:
            model.eval()
            with torch.no_grad():
                say("--- sample ---")
                say(sample_fn(model))
            model.train()

    state.seconds = time.perf_counter() - started
    if checkpoint_dir:
        state.last_checkpoint = save_checkpoint(
            os.path.join(checkpoint_dir, "last.pt"),
            model,
            optimizer=optimizer,
            step=state.global_step,
            epoch=max(start_epoch, config.max_epochs - 1),
            tokenizer=tokenizer,  # type: ignore[arg-type]
            history=state.as_dict(),
        )
    return state


def append_csv_row(path: str, row: Row) -> None:
    """Append one row, writing the header only when the file is new."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
