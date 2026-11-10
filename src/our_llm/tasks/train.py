"""Stage 2 — pretrain the GPT (ROADMAP M4).

This is the wiring, not the algorithm: the recipe lives in
:mod:`our_llm.train.loop`, the loss in :mod:`our_llm.train.loss` and the
checkpoint format in :mod:`our_llm.train.checkpoint`. Here we only

1. turn the corpus into a token file + loaders (reusing `tokenize`'s output when
   the pipeline hands us a `token_file`),
2. size the model from the *tokenizer's* vocabulary (never a hardcoded 50,257),
3. build the optimizer and run the loop,
4. return the best checkpoint path -- that is what `eval` and `sample` consume.

`NotImplementedError` used to signal "this stage has not landed yet"; the
pipeline keeps that distinction (a skip is not a failure) even now that M4 does.
"""

from __future__ import annotations

import os
from dataclasses import replace

from our_llm.config import DEFAULT_NUM_MERGES, DataConfig, TrainConfig, get_preset


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
    val_ratio: float = 0.10,
    processed_dir: str = "data/processed",
    cache_dir: str = "data/cache",
    eval_freq: int | None = None,
    quiet: bool = False,
) -> str:
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
      val_ratio: Fraction of documents held out for validation.
      processed_dir: Where the token file is written/reused.
      cache_dir: Where vocabulary caches live.
      eval_freq: Override how many steps between validation passes.
      quiet: Suppress the per-evaluation lines (the summary is still printed).

    Returns:
      The path to the best checkpoint (`best.pt`, or `last.pt` if validation never
      improved -- e.g. a corpus with a single document).
    """
    # Torch is imported *here*, not at module level: `import our_llm.tasks` must
    # stay cheap for the torch-free stages (tokenize, train-tokenizer) -- the same
    # reason `our_llm/data/__init__.py` resolves its torch names lazily.
    from our_llm.data.dataset import build_loaders, prepare
    from our_llm.model.gpt import GPTModel
    from our_llm.train.checkpoint import check_vocab, load_checkpoint
    from our_llm.train.loop import build_optimizer, new_run_dir, train_model
    from our_llm.utils import count_parameters, get_device, human_count, seed_everything

    say = (lambda *a, **k: None) if quiet else print
    seed_everything(seed)
    device = get_device()

    # -- data ----------------------------------------------------------------
    data_config = DataConfig(
        corpus_path=corpus,
        granularity=granularity,
        num_merges=num_merges,
        val_ratio=val_ratio,
        batch_size=batch_size or DataConfig().batch_size,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        seed=seed,
    )
    prepared = prepare(data_config, token_file=token_file)
    if prepared.n_train_tokens == 0:
        raise ValueError(f"no training tokens for corpus {corpus!r}")

    say(
        f"Data   : {prepared.n_train_tokens:,} train / {prepared.n_val_tokens:,} val tokens, "
        f"vocab {prepared.vocab_size:,} ({prepared.tokenizer.num_merges:,} merges)"
    )

    # -- model ---------------------------------------------------------------
    base = get_preset(preset)
    if pos_emb:
        base = replace(base, pos_emb=pos_emb)  # validates in __post_init__
    cfg = base.with_vocab(prepared.vocab_size)

    trainer = TrainConfig(
        preset=preset,
        num_merges=prepared.tokenizer.num_merges,
        seed=seed,
        learning_rate=learning_rate or TrainConfig().learning_rate,
        max_epochs=epochs or TrainConfig().max_epochs,
        eval_freq=eval_freq or TrainConfig().eval_freq,
    )

    train_loader, val_loader, prepared = build_loaders(
        data_config, context_length=cfg.context_length, prepared=prepared
    )

    # -- resume / fresh ------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val: float | None = None
    history: dict | None = None

    if resume:
        checkpoint = load_checkpoint(resume, map_location=device, with_optimizer=True)
        check_vocab(checkpoint, prepared.tokenizer)
        if checkpoint.cfg.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"checkpoint vocab {checkpoint.cfg.vocab_size} != data vocab {cfg.vocab_size}"
            )
        if pos_emb and checkpoint.cfg.pos_emb != pos_emb:
            raise ValueError(
                f"cannot switch pos_emb to {pos_emb!r} while resuming a "
                f"{checkpoint.cfg.pos_emb!r} checkpoint"
            )
        cfg = checkpoint.cfg
        model = checkpoint.model
        start_epoch = checkpoint.epoch + 1
        global_step = checkpoint.step
        best_val = checkpoint.history.get("best_val")
        history = checkpoint.history
        out_dir = os.path.dirname(os.path.abspath(resume)) or new_run_dir(run_dir)
        say(f"Resume : step {global_step}, epoch {start_epoch} of {trainer.max_epochs} ({resume})")
    else:
        model = GPTModel(cfg, use_sdpa=use_sdpa)
        out_dir = new_run_dir(run_dir)

    optimizer = build_optimizer(model, trainer)
    if resume and checkpoint.optimizer_state:
        optimizer.load_state_dict(checkpoint.optimizer_state)

    say(
        f"Model  : preset {preset!r}, ctx {cfg.context_length}, emb {cfg.emb_dim}, "
        f"{cfg.n_layers} layers, {cfg.n_heads} heads, pos_emb {cfg.pos_emb!r}"
    )
    say(
        f"         {count_parameters(model):,} parameters "
        f"({human_count(count_parameters(model))}), device {device}, {trainer.max_epochs} epoch(s)"
    )
    say(f"Run    : {out_dir}")

    result = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        trainer,
        tokenizer=prepared.tokenizer,
        checkpoint_dir=out_dir,
        log_path=os.path.join(out_dir, "metrics.csv"),
        quiet=quiet,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val=best_val,
        history=history,
    )

    say(str(result))
    say(f"Metrics: {os.path.join(out_dir, 'metrics.csv')}")
    return result.checkpoint or result.last_checkpoint or ""
