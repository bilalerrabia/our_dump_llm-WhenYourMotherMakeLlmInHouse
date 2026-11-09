"""Stage 2 — pretrain the GPT (ROADMAP M4).

The training loop itself is not written yet. This module owns the *interface*:
the signature below is the contract the CLI, `pipeline` and (later)
`src/our_llm/train/loop.py` all speak. Filling it in means implementing
`train_model` + the checkpoint save/load and calling them here -- nothing about
the entry point has to change.

Why raise instead of print: `pipeline` runs stages in order and must be able to
tell "skipped, not started yet" (NotImplementedError) from "worked" and
"crashed". An exception carries that distinction; a print does not.
"""

from __future__ import annotations

from our_llm.config import DEFAULT_NUM_MERGES, TrainConfig, get_preset

#: Where the loop will land. Kept as text so the error message stays honest.
_MODULE = "src/our_llm/train/loop.py"


def train(
    preset: str = "tiny",
    corpus: str = "data",
    num_merges: int = DEFAULT_NUM_MERGES,
    granularity: str = "paragraph",
    token_file: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    pos_emb: str | None = None,
    use_sdpa: bool = False,
    run_dir: str = "runs",
    resume: str | None = None,
    seed: int = 123,
) -> None:
    """Pretrain the GPT on the tokenized corpus and save a checkpoint.

    Args:
      preset: `debug`, `tiny` or `gpt2-small` (see `config.PRESETS`).
      corpus: Corpus file or directory; tokenized on the fly if needed.
      num_merges: BPE merges the vocabulary must have.
      granularity: `file` or `paragraph` — the train/val split unit.
      token_file: Train on this encoded token file (what `tokenize` wrote) instead
        of tokenizing the corpus again.
      epochs: Override `TrainConfig.max_epochs`.
      batch_size: Override the batch size.
      learning_rate: Override `TrainConfig.learning_rate`.
      pos_emb: `learned` or `sinusoidal`; default comes from the preset.
      use_sdpa: Use the fused attention kernel instead of the manual softmax.
      run_dir: Where checkpoints, CSV logs and samples are written.
      resume: Path to a checkpoint to continue from.
      seed: Seed for the model init and the loaders.
    """
    cfg = get_preset(preset)
    defaults = TrainConfig()
    raise NotImplementedError(
        "train: the loop lands in M4 (" + _MODULE + "). "
        f"Preset {preset!r} is ready: {cfg.num_parameters():,} parameters "
        f"(ctx {cfg.context_length}, emb {cfg.emb_dim}, {cfg.n_layers} layers, "
        f"vocab {cfg.vocab_size:,}); recipe = AdamW lr {learning_rate or defaults.learning_rate} "
        f"wd {defaults.weight_decay}, {epochs or defaults.max_epochs} epochs, "
        f"run_dir {run_dir!r}. Stages M0-M3 (tokenizer, data, model) are done."
    )
