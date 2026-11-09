# Project Memory — our-llm

## What this is
A learn-by-building GPT project ("When your mother makes an LLM in house"): a
from-scratch BPE tokenizer, corpus reader, GPT model and training loop following
*Build a LLM from Scratch* (Raschka). Corpus: `data/dd.txt` (441 KB, Project
Gutenberg Frankenstein). `README.md` is still empty; notes live in `todo.txt`,
`ROADMAP.md` (the plan + decision log) and notebooks under `src/*/` and `parts/`.

**The pipeline is being rebuilt package-by-package**. `our_llm/` is the
production path (torch, typed, tested); the old top-level `src/*.py` modules are
the pure-Python *reference* implementations, kept and tested but no longer used
by the pipeline (ROADMAP D2).

## Layout — new package (`src/our_llm/`)
- `__init__.py` — public API + `main()` for the `our-llm` console script
  (delegates to `our_llm.app.main`).
- `config.py` — frozen dataclasses `GPTConfig` / `DataConfig` / `TrainConfig`,
  presets `debug` / `tiny` / `gpt2-small`, and `GPTConfig.num_parameters()`
  (no torch needed). Pinned: tiny **13,039,872**, debug 1,078,400,
  gpt2-small 124,412,160 (tied).
- `app.py` — the **fire** CLI. `class OurLLM` maps each name in `tasks.COMMANDS`
  to a `staticmethod`; `main(argv)` validates flags then calls `fire.Fire`.
  Fire silently ignores an unknown `--flag`, so `validate_flags` (using
  `inspect.signature`, accepting `--num_merges`/`--num-merges`/`--nummerges` and
  `--nofoo` for bools) turns a typo into an error with a "did you mean" hint.
  Catches `NotImplementedError` -> "not implemented yet" + exit 1, and
  `ValueError`/`FileNotFoundError` -> "error: ..." + exit 1. Fire's own flags go
  after `--` (`main.py tokenize -- --help`); it prints help to **stderr**.
- `tasks/` — **one module per pipeline stage**, all torch-free:
  - `tokenize.py` — done. `pick_tokenizer()` (explicit -> retrain -> corpus cache ->
    legacy -> train; the legacy file is only reused when `num_merges` matches) and
    `tokenize(...)` -> `TokenizeResult` (vocab/n_tokens/split/token_file/reused).
  - `train.py` — M4 stub: signature only, raises `NotImplementedError`.
  - `evaluate.py` — M4 stub (loss/perplexity).
  - `sample.py` — M5 stub (greedy/temperature/top-k/top-p/repetition penalty).
  - `pipeline.py` — `pipeline()` runs tokenize -> train -> eval -> sample, passing
    the token file then the checkpoint along; returns `PipelineReport` with one
    `StageOutcome` per stage: `ok` / `skipped` (stage raised `NotImplementedError`)
    / `failed` (stop + `SystemExit(1)`). `parse_stages("all" | "tokenize,train")`
    always re-orders into pipeline order. Extra flags: `--stages`, `--checkpoint`,
    `--prompt`, `--run-dir`, `--cache-dir`, `--processed-dir`, `--epochs`,
    `--quiet`, `--trace`.
  - `__init__.py` — re-exports every stage + `COMMANDS` dict (single source of truth
    for the CLI) and `STAGES`/`PipelineReport`/`StageOutcome`/`TokenizeResult`.
- `utils.py` — `get_device`, `seed_everything`, `count_parameters`, `human_count`.
- `tokenizer/special.py` — **frozen id layout**: 0-255 bytes, 256 `<|endoftext|>`,
  257 `<|pad|>`, 258+ merges. `vocab_size = 258 + len(merges)`.
- `tokenizer/bpe.py` — `BPETokenizer`: rank-based fast encode (memoised per word),
  `train` over unique words weighted by frequency, `save`/`load` (self-describing
  `_meta` header, auto-migrates legacy caches), `load_or_train`,
  `tokenizer_cache_path`, `corpus_hash`.
- `data/corpus.py` — `read_corpus(path, extensions, granularity)` -> `Corpus(docs, sources)`;
  `find_data_files` filters by extension. `--granularity paragraph` (the default) splits
  files on blank lines, so a single novel still yields ~857 split-able documents.
  `Corpus` exposes three hashes: `fingerprint` (= `files_hash`, raw file text -- keys the
  **vocab** cache), and `structure_fingerprint` (document boundaries -- keys the **token
  file**), because the same characters laid out as 1 doc vs 857 docs are different tokens.
- `data/split.py` — `split_documents(n_documents, val_ratio, seed)` (seeded shuffle,
  90/10, never empties train). Extracted from `data/dataset.py` **so the tokenize
  stage does not import torch**; `dataset.py` re-imports it for compatibility.
- `data/__init__.py` — `corpus`/`split`/`tokens` eagerly, the torch-backed names
  (`prepare`, `build_loaders`, `GPTDataset`, ...) lazily via PEP 562 `__getattr__`:
  `from our_llm.data import prepare` still works, `import our_llm.tasks` does not
  load torch (~1.5 s saved).
- `data/tokens.py` — M2a. `encode_corpus` (per document, trailing `<|endoftext|>`) ->
  `uint16` `.bin` (no header, `np.memmap`-able) + `.meta.json` sidecar: version,
  corpus/structure/vocab fingerprints, dtype, n_tokens, doc_spans (one line), doc_source
  index, eot_id. `open_tokens` (memmap|read), `load_meta`, `check_compatible`
  (raises on stale version / different vocabulary), `token_file_path`, `tokens_dtype`.
- `data/dataset.py` — M2b. `GPTDataset` (sliding window, `stride=None` -> context length,
  targets = inputs shifted by 1, `torch.long`), `gather`, `prepare()` (corpus -> cached
  token file -> train/val arrays), `make_loader` / `build_loaders` (seeded
  `torch.Generator`; raises with a reason on an empty stream, torch itself refuses).
  `PreparedData` carries ids, docs, meta, tokenizer.
- `model/layers.py` — M3. Hand-written `LayerNorm` (`scale`/`shift`, `unbiased=False`) and
  tanh `GELU`, `FeedForward(emb_dim, expand=4, drop_rate=0.0)`; parity-tested against
  `nn.LayerNorm` / `nn.GELU(approximate="tanh")` at 1e-6.
- `model/attention.py` — M3. `MultiHeadAttention(d_in, d_out, context_length, dropout,
  num_heads, qkv_bias=False, use_sdpa=False)`. Manual softmax path **and** `F.scaled_dot_product_attention`
  path agree to 1e-6 (both tested); causal mask is a **non-persistent bool buffer** so it never
  lands in a checkpoint; raises on `T > context_length`. 1-head + identity `out_proj` == legacy
  `CausalAttention`.
- `model/gpt.py` — M3. `GPTModel(cfg, use_sdpa=False)` (pre-norm blocks, `nn.ModuleList`,
  GPT-2 init std 0.02 with residual projections `/sqrt(2L)`, init *before* tying), `TransformerBlock`,
  `sinusoidal_positional_encoding` (fixed non-persistent buffer, **0 params**; `cfg.num_parameters()`
  accounts for it). Also `model.num_parameters()` and `.cfg` stored for M4 checkpoints.
- `examples/pure_python/pipeline_demo.py` — the old `main.py` walkthrough (slow on purpose).

## Layout — legacy reference modules (still tested)
- `src/tokenizer.py` — `Tokenizer` with `map_replace: {new_id: (l, r)}`, merges from 256.
  **Reference implementation**; `our_llm.tokenizer` is a superset and must stay
  merge-for-merge compatible (there is a test for exactly that).
- `src/data_loader.py` — pure-Python `DatasetPreparation` + `create_dataloader`.
- `src/embedding.py` — pure-Python `Embedding`, `sinusoidal_positional_encoding`,
  `add_embeddings`, `get_shape`.
- `src/attentions/` — `SelfAttention`, `CausalAttention` (note filename typo
  `causel_attention.py`), `MultiHeadAttention`. Import-safe: demos live under
  `if __name__ == "__main__"`; previously `multi_head_attention.py` raised
  `NameError: batch` on import.
- `src/LLM_arch/llm_class.py` — `DummyGPTModel`, `LayerNorm`, `GELU`, `FeedForward`,
  `TransformerBlock` + GPT-2 124M cfg. **Superseded by `our_llm/model/` (M3)**; kept as the
  reference the new code was ported from.
- `main.py` — 8-line entry point: `from our_llm.app import main` + `main()`.
  No `sys.path` hack, no argparse; `python main.py` prints the Fire command list.

## Commands
- `uv run our-llm tokenize` (= `uv run python main.py tokenize`) — corpus -> BPE
  cache -> `data/processed/tokens-*.bin` + meta, prints token count and the
  train/val split (0.1 s; 96,061 tokens, 857 docs, 771/86 split).
  Flags: `--granularity file|paragraph`, `--val-ratio`, `--processed-dir`, `--force`,
  `--nowrite` (was `--no-tokens`), `--retrain`, `--vocab-cache`, `--cache-dir`,
  `--num-merges`, `--quiet`.
- `uv run our-llm pipeline` — every stage in order (today: tokenize ok,
  train/eval/sample `skipped` with the reason). `--stages tokenize,train`,
  `--preset`, `--checkpoint`, `--prompt`, `--run-dir`, `--quiet`, `--trace`.
- `uv run our-llm tokenize --retrain` — full retrain, ~68 s for 14,970 merges
- `make` (= sync + tokenize), `make pipeline [STAGES=tokenize]`, `make test`,
  `make overfit` (M4), `make lint|fmt` (needs ruff)
- `uv run pytest -q` — **260 passing** (test_config / test_tokenizer_v2 / test_corpus /
  test_tokens / test_dataset / test_model / test_tasks / test_app + the legacy
  suites); full suite ~1.7 s
- Real data end-to-end: `build_loaders(DataConfig(), context_length=256)` -> 335 train
  windows / 41 batches, 39 val windows / 5 batches, 0.015 s startup
- Deps: **fire**, numpy, regex, torch, tqdm (+ pytest). Python >= 3.14, managed by uv.
- `tiktoken`, `matplotlib`, `pandas`, `ruff`, `mypy` are **not installed**.

## Git workflow
History is one commit per milestone part, **each commit's tree must be green**
(verified by checking out the commit in a temp `git worktree` and running pytest
against it). Order commits so that every import resolves at that commit:
`tooling -> tokenizer -> core(config+corpus) -> data/tokens -> data/dataset -> cli -> docs`.
Message style: `type(scope): imperative summary` (+ bullet body explaining *why*),
types used so far: `chore`, `feat`, `docs`, `fix`. Commits through M3:

```
7a8ed7d chore(tooling)   scaffolding + import-safe attention demos
7f0be76 feat(tokenizer)  fast BPE, frozen ids, versioned cache
74d4e08 feat(core)       config presets, utils, corpus reader
1044187 feat(data)       memmapped uint16 token file (M2a)
7ff095d feat(data)       sliding-window dataset + doc split (M2b)
f3f030e feat(cli)        tokenize writes the token file, reports the split
6d2093d docs             M0-M2 recorded
7bf2864 docs(agent)      git workflow: one green commit per part
382bfae feat(model)      GPTModel, causal attention, layers (M3)
<this>  docs             M3 recorded (audit, checklist, D17)
```
`data/cache/` and `data/processed/` are gitignored: never commit token binaries.

## `parts/` — reference notebooks (book source material, NOT project code)
`parts/LLM Loss function.ipynb` (513 cells) and `parts/LLM Pretraining.ipynb` (558 cells).
Their first ~4,200 extracted lines are **byte-identical**: both are the *Build a LLM from Scratch*
walkthrough ch2 -> ch5 (tokenization -> dataloader -> embeddings -> attention -> GPT model ->
generation -> cross-entropy/perplexity -> train/val loss). `Pretraining` then continues with
the actual `train_model_simple` loop and decoding strategies (multinomial + temperature);
top-k is only *mentioned*, never implemented. Nothing later (ch6 fine-tuning, ch7 instruction
tuning, checkpoint save/load, loading GPT-2 weights) is present.

They are the *book's* code, not ours: `tiktoken` "gpt2" encoding (vocab 50,257), corpus
`the-verdict.txt` (20,479 chars, downloaded from GitHub), learned `nn.Embedding` positional
weights, `torch.utils.data.DataLoader`. Key APIs to port:
`GPTModel` (full, vs our `DummyGPTModel`), `generate_text_simple`, `text_to_token_ids`/`token_ids_to_text`,
`calc_loss_batch` (`cross_entropy(logits.flatten(0,1), targets.flatten())`), `calc_loss_loader`,
`evaluate_model`, `generate_and_print_sample`, `softmax_with_temperature`, `print_sampled_tokens`,
`GPTDatasetV1`/`create_dataloader_v1`.
Book hyperparams: ctx 1024 -> **256** for CPU, batch 2, stride=ctx, 90/10 split,
AdamW lr=4e-4 wd=0.1, 10 epochs, eval_freq=5/eval_iter=5; results train 9.781->0.391,
val 9.856->6.372 (overfits a 20k-char story).
- Gotcha: the notebooks cannot be run as-is here (missing deps + `the-verdict.txt`).

## Conventions / gotchas
- **Import paths**: the editable install's `site-packages/our_llm.pth` puts **`src/` itself**
  on `sys.path`, so both `import our_llm` and `from src.tokenizer import ...` work.
  `pyproject` pytest `pythonpath = [".", "src"]` makes that explicit.
- **Perf**: `BPETokenizer.encode` is 0.08 s for 438 KB (memoised, rank-based). The old
  `src/tokenizer.py` replays every merge and takes ~180 s — never put it on a hot path.
- **Vocab ids are frozen** (see `tokenizer/special.py`). A checkpoint's `out_head` shape
  depends on it; never reorder specials. Always derive `vocab_size` from the tokenizer
  (`cfg.with_vocab(tok.vocab_size)`), never hardcode 50,257.
- **Tokenizer cache** = `data/cache/vocab-<files_hash>-m<num_merges>.json`, keyed on the raw
  file text so changing `--granularity` does not invalidate it. `load_or_train(..., key=...)`
  lets callers pass that key explicitly. The root-level `tokenizer_vocab*.json` are the *old*
  format kept as a legacy-migration fixture.
- **Token file** = `data/processed/tokens-<structure_hash>-m<num_merges>.bin` (+ `.meta.json`),
  gitignored, regenerated in ~0.1 s; `check_compatible` fails loudly if the vocabulary or the
  file version changed. Bump `TOKEN_FILE_VERSION` in `data/tokens.py` on layout changes.
- **Split rule**: split by *document*, never by token percentage - a cut 90 % through the
  byte stream lands mid-sentence and leaks. Tests assert the ranges are disjoint.
- `data/gg` is a near-duplicate of `data/dd.txt` without an extension; the corpus reader
  ignores it (it was double-loading the book before).
- `parts/` is **gitignored** — notebook knowledge must be copied into `notes/` to survive.
- `python 3.14` + `torch` — if an op misbehaves, 3.14 is the first suspect.
- `ROADMAP.md` holds the full pipeline plan + 18 architecture decisions (D1-D18) with rationale,
  and the M0-M7 milestones (M0-M3 done; M4 = loss + training loop is next). Change a decision by
  editing that table, not in code. **D18** = fire CLI + `tasks/` (one module per stage) + the
  `pipeline` command.
- **Adding a stage method**: write the function in `tasks/<name>.py` (signature + `Args:`
  docstring is the CLI), import it in `tasks/__init__.py`, add it to `COMMANDS` **and** to
  `OurLLM` in `app.py` (a test asserts the two match), and to `STAGES` if `pipeline` should run
  it. Never put logic in `app.py` or `main.py`.
- `.env` holds an API key — never commit it.
- Style: type hints, module docstrings, small pure functions. `todo.txt` tracks next steps
  (GELU derivative, lint-strict, README from notebook notes).
- `src/data_loader.py` (pure Python) is the *teaching* twin of `our_llm/data/dataset.py`;
  keep them separately tested, do not merge.
