"""Training: loss, the loop, checkpoints (ROADMAP M4).

    train/loss.py        cross-entropy + perplexity + a loader-average
    train/loop.py        AdamW groups, LR schedule, eval, CSV log, best checkpoint
    train/checkpoint.py  save/load with the vocabulary fingerprint inside

The stage that ties these together is :mod:`our_llm.tasks.train`; nothing here
knows about the CLI.

The names below are resolved lazily (PEP 562), the same trick as
``our_llm/data/__init__.py``: every module in this package needs torch, and
``import our_llm.tasks`` must stay usable without paying the ~1.5 s import for
stages that only manipulate text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time only, never at runtime
    from our_llm.train.checkpoint import (
        Checkpoint,
        check_vocab,
        load_checkpoint,
        save_checkpoint,
    )
    from our_llm.train.loop import (
        TrainHistory,
        build_optimizer,
        lr_at_step,
        new_run_dir,
        train_model,
    )
    from our_llm.train.loss import calc_loss_batch, calc_loss_loader, perplexity

_LAZY = {
    "Checkpoint": "checkpoint",
    "check_vocab": "checkpoint",
    "load_checkpoint": "checkpoint",
    "save_checkpoint": "checkpoint",
    "TrainHistory": "loop",
    "build_optimizer": "loop",
    "lr_at_step": "loop",
    "new_run_dir": "loop",
    "train_model": "loop",
    "calc_loss_batch": "loss",
    "calc_loss_loader": "loss",
    "perplexity": "loss",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"our_llm.train.{module_name}"), name)
