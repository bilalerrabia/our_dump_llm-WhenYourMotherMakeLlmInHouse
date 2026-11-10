"""Cross-entropy loss and perplexity.

The model produces ``(B, T, vocab)`` logits but cross-entropy wants
``(samples, classes)``: ``flatten(0, 1)`` collapses batch and time, and each of
the ``B*T`` predictions is scored against the token that actually came next.
Nothing is averaged over time first -- that would weight long documents less.

``perplexity = exp(loss)`` is just the loss in "how many tokens was the model
torn between" units; it is easier to read (vocab 15,228 -> a uniform guess is
perplexity 15,228).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def calc_loss_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mean cross-entropy of one ``(inputs, targets)`` batch."""
    if device is None:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else inputs.device
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def calc_loss_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device | None = None,
    num_batches: int | None = None,
    grad: bool = False,
) -> float:
    """Average loss over ``num_batches`` batches of ``loader`` (``None`` = all).

    Evaluation never needs gradients, so ``grad=False`` (the default) runs under
    ``torch.no_grad()``; pass ``grad=True`` only for the training-loss probe.
    An empty loader returns ``nan`` rather than dividing by zero -- callers check.
    """
    if len(loader) == 0:
        return float("nan")

    total = 0.0
    seen = 0
    with torch.set_grad_enabled(grad):
        for index, (inputs, targets) in enumerate(loader):
            if num_batches is not None and index >= num_batches:
                break
            total += calc_loss_batch(model, inputs, targets, device).item()
            seen += 1
    return total / seen if seen else float("nan")


def perplexity(loss: float) -> float:
    """``exp(loss)``; ``inf`` for a nan/inf loss instead of raising."""
    return float(math.exp(loss)) if math.isfinite(loss) else float("inf")
