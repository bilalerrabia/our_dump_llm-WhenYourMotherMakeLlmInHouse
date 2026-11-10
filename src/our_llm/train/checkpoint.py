"""Save and load checkpoints (ROADMAP D11).

A checkpoint is a plain dict:

    {
      "format": 1,
      "cfg": {...GPTConfig fields...},          # rebuilds the architecture
      "model_state": {...},                     # learned tensors only
      "optimizer_state": {...} | None,          # to resume without a LR spike
      "step": int, "epoch": int,
      "tokenizer": {"vocab_fingerprint": ..., "vocab_size": ..., "num_merges": ...},
      "history": {"rows": [...], "best_val": ...}
    }

Two things are deliberately **not** in the state dict: the causal mask and the
sinusoidal table are non-persistent buffers (D17), so a checkpoint holds only
learned weights.

The tokenizer fingerprint is stored *inside* the checkpoint on purpose: loading a
model next to a vocabulary with different merges would otherwise produce
plausible-looking garbage instead of an error. :func:`check_vocab` is that guard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch import nn

from our_llm.config import GPTConfig
from our_llm.data.tokens import vocab_fingerprint
from our_llm.model.gpt import GPTModel
from our_llm.tokenizer.bpe import BPETokenizer

CHECKPOINT_VERSION = 1


@dataclass
class Checkpoint:
    """What `load_checkpoint` hands back: a ready model plus where it came from."""

    path: str
    model: GPTModel
    cfg: GPTConfig
    step: int = 0
    epoch: int = 0
    optimizer_state: dict | None = None
    tokenizer: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    @property
    def vocab_fingerprint(self) -> str | None:
        return self.tokenizer.get("vocab_fingerprint")

    @property
    def num_merges(self) -> int | None:
        return self.tokenizer.get("num_merges")

    @property
    def best_val(self) -> float | None:
        return self.history.get("best_val")


def _tokenizer_meta(tokenizer: BPETokenizer | dict | None) -> dict:
    """Normalise a tokenizer, a token-file meta dict or nothing into the stored dict."""
    if tokenizer is None:
        return {}
    if isinstance(tokenizer, BPETokenizer):
        return {
            "vocab_fingerprint": vocab_fingerprint(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "num_merges": tokenizer.num_merges,
        }
    keys = ("vocab_fingerprint", "vocab_size", "num_merges")
    return {key: tokenizer[key] for key in keys if key in tokenizer}


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: GPTConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    epoch: int = 0,
    tokenizer: BPETokenizer | dict | None = None,
    history: dict | None = None,
) -> str:
    """Write ``path`` and return it (so callers can chain the path along)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    resolved = cfg or getattr(model, "cfg", None)
    if resolved is None:
        raise ValueError("save_checkpoint needs cfg or a model that carries one")

    payload = {
        "format": CHECKPOINT_VERSION,
        "cfg": resolved.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "tokenizer": _tokenizer_meta(tokenizer),
        "history": dict(history or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
    with_optimizer: bool = False,
) -> Checkpoint:
    """Rebuild the model from the checkpoint's config and load the weights.

    ``strict=True`` (torch's default) is the point: a checkpoint whose keys no
    longer match the architecture must fail here, not at step 3,000.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("format")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint format {version} != {CHECKPOINT_VERSION} ({path}); retrain"
        )

    cfg = GPTConfig.from_dict(payload["cfg"])
    model = GPTModel(cfg)
    model.load_state_dict(payload["model_state"])

    return Checkpoint(
        path=path,
        model=model,
        cfg=cfg,
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        optimizer_state=payload.get("optimizer_state") if with_optimizer else None,
        tokenizer=dict(payload.get("tokenizer") or {}),
        history=dict(payload.get("history") or {}),
    )


def check_vocab(checkpoint: Checkpoint, tokenizer: BPETokenizer) -> None:
    """Raise unless the checkpoint's vocabulary is the one in ``tokenizer``."""
    expected = checkpoint.vocab_fingerprint
    if expected is None:
        return  # written without a tokenizer; nothing to compare
    actual = vocab_fingerprint(tokenizer)
    if expected != actual:
        raise ValueError(
            f"checkpoint {checkpoint.path!r} was trained with vocabulary {expected} "
            f"but the data has {actual}; retrain with `our-llm train`"
        )
