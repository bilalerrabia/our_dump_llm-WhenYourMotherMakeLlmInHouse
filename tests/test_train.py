"""Tests for the M4 training stack: loss, LR schedule, optimizer, checkpoints, loop.

Every test here is small on purpose (vocab 64, ctx 16, 2 layers): the things that
break silently in a training loop are the *shapes and the wiring*, not the size,
and a 0.2 s test is one you actually run.
"""

from __future__ import annotations

import csv
import math
import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from our_llm.config import GPTConfig, TrainConfig
from our_llm.model.gpt import GPTModel
from our_llm.tokenizer.bpe import BPETokenizer
from our_llm.train.checkpoint import check_vocab, load_checkpoint, save_checkpoint
from our_llm.train.loop import build_optimizer, lr_at_step, new_run_dir, train_model
from our_llm.train.loss import calc_loss_batch, calc_loss_loader, perplexity

VOCAB = 64
CONTEXT = 16


# --- fixtures ---------------------------------------------------------------


def small_cfg(vocab: int = VOCAB, drop_rate: float = 0.0) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab,
        context_length=CONTEXT,
        emb_dim=32,
        n_heads=2,
        n_layers=2,
        drop_rate=drop_rate,
    )


class UniformLogits(nn.Module):
    """Predicts every token equally: cross-entropy must be exactly ``log(vocab)``."""

    def __init__(self, vocab: int = VOCAB) -> None:
        super().__init__()
        self.vocab = vocab
        self.seen = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen += inputs.shape[0]
        return torch.zeros(inputs.shape[0], inputs.shape[1], self.vocab)


def batch_dataset(n_samples: int = 8, context: int = CONTEXT) -> TensorDataset:
    """Deterministic next-token pairs from a repeating 0..VOCAB-1 pattern."""
    pattern = torch.arange(n_samples * context + context) % VOCAB
    inputs = torch.stack([pattern[i * context : i * context + context] for i in range(n_samples)])
    targets = torch.stack(
        [pattern[i * context + 1 : i * context + context + 1] for i in range(n_samples)]
    )
    return TensorDataset(inputs, targets)


def loader(**kwargs) -> DataLoader:
    return DataLoader(batch_dataset(), batch_size=kwargs.pop("batch_size", 4), **kwargs)


# --- loss -------------------------------------------------------------------


class TestLoss:
    def test_uniform_logits_give_log_vocab(self):
        inputs, targets = next(iter(loader()))
        loss = calc_loss_batch(UniformLogits(), inputs, targets)
        assert torch.isclose(loss, torch.tensor(math.log(VOCAB)), atol=1e-6)

    def test_loader_average_matches_the_batches_it_saw(self):
        model = UniformLogits()
        value = calc_loss_loader(loader(), model)
        assert value == pytest.approx(math.log(VOCAB), abs=1e-6)
        assert model.seen == 8  # all four batches, two samples each

    def test_num_batches_caps_the_work(self):
        model = UniformLogits()
        calc_loss_loader(loader(batch_size=2), model, num_batches=2)  # 4 batches available
        assert model.seen == 4

    def test_empty_loader_is_nan_not_zero(self):
        empty = DataLoader(
            TensorDataset(torch.empty(0, CONTEXT, dtype=torch.long),
                          torch.empty(0, CONTEXT, dtype=torch.long)),
            batch_size=2,
        )
        assert math.isnan(calc_loss_loader(empty, UniformLogits()))

    def test_loss_is_independent_of_batch_shape(self):
        """flatten(0, 1) must score every (batch, time) prediction, not average first."""
        model = UniformLogits()
        flat = calc_loss_loader(loader(batch_size=8), model)
        split = calc_loss_loader(loader(batch_size=2), model)
        assert flat == pytest.approx(split, abs=1e-6)

    def test_perplexity_is_exp_of_loss(self):
        assert perplexity(0.0) == pytest.approx(1.0)
        assert perplexity(math.log(10)) == pytest.approx(10.0)
        assert perplexity(float("nan")) == float("inf")


# --- schedule ---------------------------------------------------------------


class TestLrSchedule:
    def test_linear_warmup_then_cosine_decay(self):
        base, warmup, total, floor = 1.0, 10, 110, 0.1
        assert lr_at_step(0, base, total, warmup, floor) == pytest.approx(0.1)
        assert lr_at_step(4, base, total, warmup, floor) == pytest.approx(0.5)
        assert lr_at_step(9, base, total, warmup, floor) == pytest.approx(1.0)
        assert lr_at_step(10, base, total, warmup, floor) == pytest.approx(1.0)
        assert lr_at_step(total, base, total, warmup, floor) == pytest.approx(floor)
        assert lr_at_step(total * 3, base, total, warmup, floor) == pytest.approx(floor)

    def test_it_never_increases_after_warmup(self):
        values = [lr_at_step(step, 1.0, 200, 20, 0.1) for step in range(20, 210)]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))

    def test_zero_warmup_starts_at_full_rate(self):
        assert lr_at_step(0, 1.0, 10, 0, 0.1) == pytest.approx(1.0)


# --- optimizer --------------------------------------------------------------


class TestOptimizer:
    def test_weight_decay_only_on_matrices(self):
        model = GPTModel(small_cfg())
        config = TrainConfig(weight_decay=0.1)
        optimizer = build_optimizer(model, config)
        decay, no_decay = optimizer.param_groups

        assert decay["weight_decay"] == 0.1
        assert no_decay["weight_decay"] == 0.0
        assert all(parameter.dim() >= 2 for parameter in decay["params"])
        assert all(parameter.dim() < 2 for parameter in no_decay["params"])

    def test_every_parameter_exactly_once_despite_weight_tying(self):
        model = GPTModel(small_cfg())
        optimizer = build_optimizer(model)
        grouped = sum(len(group["params"]) for group in optimizer.param_groups)
        assert grouped == len(list(model.parameters()))
        assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == (
            model.num_parameters()
        )


# --- run directory ----------------------------------------------------------


class TestRunDir:
    def test_creates_a_timestamped_directory(self, tmp_path):
        path = new_run_dir(str(tmp_path / "runs"))
        assert os.path.isdir(path)
        assert os.path.basename(path).count("-") == 1  # YYYYmmdd-HHMMSS

    def test_same_second_does_not_reuse_the_directory(self, tmp_path):
        from datetime import datetime

        now = datetime(2026, 11, 10, 12, 0, 0)
        first = new_run_dir(str(tmp_path), now=now)
        second = new_run_dir(str(tmp_path), now=now)
        assert first != second
        assert second.endswith("-2")


# --- checkpoints ------------------------------------------------------------


class TestCheckpoint:
    def test_round_trip_restores_weights_and_metadata(self, tmp_path):
        cfg = small_cfg()
        model = GPTModel(cfg)
        optimizer = build_optimizer(model)
        path = save_checkpoint(
            str(tmp_path / "best.pt"),
            model,
            optimizer=optimizer,
            step=7,
            epoch=1,
            tokenizer={"vocab_fingerprint": "abc", "vocab_size": VOCAB, "num_merges": 10},
            history={"best_val": 3.5},
        )
        assert os.path.exists(path)

        checkpoint = load_checkpoint(path, with_optimizer=True)
        assert checkpoint.cfg == cfg
        assert (checkpoint.step, checkpoint.epoch) == (7, 1)
        assert checkpoint.vocab_fingerprint == "abc"
        assert checkpoint.num_merges == 10
        assert checkpoint.best_val == 3.5
        assert checkpoint.optimizer_state is not None

        for name, tensor in model.state_dict().items():
            assert torch.equal(checkpoint.model.state_dict()[name], tensor), name

    def test_optimizer_state_is_opt_in(self, tmp_path):
        model = GPTModel(small_cfg())
        path = save_checkpoint(str(tmp_path / "x.pt"), model, optimizer=build_optimizer(model))
        assert load_checkpoint(path).optimizer_state is None

    def test_derived_buffers_are_not_stored(self, tmp_path):
        """The causal mask is non-persistent: a checkpoint holds learned tensors only."""
        model = GPTModel(small_cfg())
        path = save_checkpoint(str(tmp_path / "x.pt"), model)
        payload = torch.load(path, weights_only=False)
        assert all("mask" not in key for key in payload["model_state"])

    def test_wrong_format_is_refused(self, tmp_path):
        path = str(tmp_path / "old.pt")
        torch.save({"format": 0, "cfg": {}, "model_state": {}}, path)
        with pytest.raises(ValueError, match="format"):
            load_checkpoint(path)

    def test_missing_checkpoint_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.pt"):
            load_checkpoint(str(tmp_path / "nope.pt"))

    def test_vocab_mismatch_fails_loudly(self, tmp_path):
        model = GPTModel(small_cfg())
        first = BPETokenizer({258: (97, 98)})
        second = BPETokenizer({258: (99, 100)})
        path = save_checkpoint(str(tmp_path / "x.pt"), model, tokenizer=first)

        checkpoint = load_checkpoint(path)
        check_vocab(checkpoint, first)  # same vocab: fine
        with pytest.raises(ValueError, match="vocabulary"):
            check_vocab(checkpoint, second)


# --- the loop ---------------------------------------------------------------


class TestTrainModel:
    def test_loss_falls_and_every_artifact_is_written(self, tmp_path):
        cfg = small_cfg()
        model = GPTModel(cfg)
        config = TrainConfig(
            learning_rate=1e-2,
            warmup_steps=0,
            max_epochs=20,
            eval_freq=10,
            eval_iter=1,
        )
        history = train_model(
            model,
            loader(),
            None,
            build_optimizer(model, config),
            torch.device("cpu"),
            config,
            checkpoint_dir=str(tmp_path),
            log_path=str(tmp_path / "metrics.csv"),
            quiet=True,
        )

        assert history.rows
        assert history.global_step == 20 * 2  # 8 samples / batch 4 = 2 steps per epoch
        assert history.final_train_loss < math.log(VOCAB) - 1.0
        assert history.checkpoint == str(tmp_path / "best.pt")
        assert os.path.exists(history.checkpoint)
        assert os.path.exists(tmp_path / "last.pt")
        assert history.epochs_done == 20
        assert "best val" in str(history)

        with open(tmp_path / "metrics.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(history.rows)
        assert rows[0]["step"] == "2"  # the first epoch-end evaluation (eval_freq 10)

    def test_validation_loader_drives_the_best_checkpoint(self, tmp_path):
        cfg = small_cfg()
        model = GPTModel(cfg)
        config = TrainConfig(learning_rate=1e-2, warmup_steps=0, max_epochs=2, eval_iter=1)
        history = train_model(
            model,
            loader(),
            loader(shuffle=False),
            build_optimizer(model, config),
            torch.device("cpu"),
            config,
            checkpoint_dir=str(tmp_path),
            quiet=True,
        )
        assert math.isfinite(history.best_val)
        assert history.best_step > 0
        assert all("val_loss" in row for row in history.rows)

    def test_sample_hook_is_called_once_per_epoch(self, tmp_path):
        model = GPTModel(small_cfg())
        config = TrainConfig(learning_rate=1e-2, warmup_steps=0, max_epochs=2, eval_freq=100)
        calls = []
        train_model(
            model,
            loader(),
            None,
            build_optimizer(model, config),
            torch.device("cpu"),
            config,
            sample_fn=lambda m: calls.append(m) or "text",
            quiet=True,
        )
        assert len(calls) == 2
