"""The single most valuable training test: overfit **one batch**.

A model that cannot memorise 4 short sequences is not regularised, it is broken --
a wrong target shift, a mask that leaks the future, or an optimizer that never
updates. Driving the loss from ``log(vocab)`` (a uniform guess) to below 0.1 on a
fixed batch proves the whole chain: embeddings -> attention -> head -> loss ->
backward -> AdamW.

It runs in a couple of seconds at a deliberately tiny size (vocab 64, ctx 16,
2 layers) -- ``make overfit``.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from our_llm.config import GPTConfig, TrainConfig
from our_llm.model.gpt import GPTModel
from our_llm.train.loop import build_optimizer, train_model

VOCAB = 64
CONTEXT = 16


def one_batch(samples: int = 4) -> TensorDataset:
    """A fixed batch whose next token is always "the next number in the cycle"."""
    pattern = torch.arange(samples * CONTEXT + 1) % VOCAB
    inputs = torch.stack([pattern[i * CONTEXT : i * CONTEXT + CONTEXT] for i in range(samples)])
    targets = torch.stack(
        [pattern[i * CONTEXT + 1 : i * CONTEXT + 1 + CONTEXT] for i in range(samples)]
    )
    return TensorDataset(inputs, targets)


def test_a_tiny_gpt_overfits_one_batch():
    cfg = GPTConfig(
        vocab_size=VOCAB,
        context_length=CONTEXT,
        emb_dim=32,
        n_heads=2,
        n_layers=2,
        drop_rate=0.0,
    )
    torch.manual_seed(0)
    model = GPTModel(cfg)
    config = TrainConfig(
        learning_rate=1e-2,
        warmup_steps=0,
        weight_decay=0.0,
        max_epochs=1000,  # one batch per epoch: 1,000 optimizer steps
        eval_freq=1000,  # one evaluation, at the end of each epoch
        eval_iter=1,
    )
    loader = DataLoader(one_batch(), batch_size=4)

    history = train_model(
        model,
        loader,
        None,
        build_optimizer(model, config),
        torch.device("cpu"),
        config,
        quiet=True,
    )

    assert math.isfinite(history.best_val)
    assert history.best_val < 0.1, f"still at {history.best_val:.4f} -- the model did not fit"
    assert history.global_step == 1000
