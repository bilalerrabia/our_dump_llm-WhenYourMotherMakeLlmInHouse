# ROADMAP — our-llm

End-to-end plan: **corpus → BPE → tokens → dataset → GPT → training → sampling**.
Everything is written from scratch on purpose (that is the point of the project);
`parts/*.ipynb` is the *Build a LLM from Scratch* reference, `src/` is our reimplementation.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## 0. Where we are today (audit)

| Stage | Status | Where | Broken / missing |
|---|---|---|---|
| Read corpus | `[x]` | `our_llm/data/corpus.py` | Fixed: extension filter excludes the `data/gg` duplicate; documents kept as split boundaries |
| BPE train | `[x]` | `our_llm/tokenizer/bpe.py` | Fixed: 14,970 merges retrained on the real corpus in **68 s**, cached by fingerprint |
| BPE encode | `[x]` | `our_llm/tokenizer/bpe.py` | Fixed: **0.08 s** for the whole corpus (was ~180 s); special tokens added |
| Batching | `[x]` | `our_llm/data/dataset.py` | Fixed: torch tensors, memmapped token file, sentence-safe doc split, seeded loaders |
| Embeddings | `[x]` | `our_llm/model/gpt.py` | learned by default (`pos_emb`); sinusoidal is a fixed non-persistent buffer, 0 params |
| Attention | `[x]` | `our_llm/model/attention.py` | causal MHA, manual + sdpa paths parity-tested; legacy demos stay in `src/attentions/` |
| GPT model | `[x]` | `our_llm/model/gpt.py` | `GPTModel` (pre-norm blocks, tying, GPT-2 init); live count == `cfg.num_parameters()` = **13,039,872** |
| Loss / perplexity | `[ ]` | — | only in `parts/LLM Loss function.ipynb` (M4) |
| Train / val split | `[x]` | `our_llm/data/dataset.py` | Fixed: 90/10 by *document* (paragraph), never mid-sentence |
| Training loop | `[ ]` | — | only in `parts/LLM Pretraining.ipynb` (M4) |
| Checkpointing | `[ ]` | — | — (M4) |
| Sampling | `[ ]` | — | only in notebooks (greedy + temperature) (M5) |
| CLI / package | `[x]` | `our_llm/app.py` + `our_llm/tasks/` | fire commands `tokenize|train|eval|sample|pipeline`; `pipeline` runs all four in order |
| Tests | `[~]` | `tests/` | **260 passing**; tokenizer/corpus/config/tokens/dataset/model/tasks/app/legacy + old pure-Python suites |
| Docs / lint | `[ ]` | `README.md` | empty; no ruff/mypy/CI (M6) |

---

## 1. Architecture decisions (ADR log)

Each row = one decision, already argued. Change it by editing this table, not by accident.

### D1 — Package layout: one real package
**Decision.** Move everything into `src/our_llm/` (`config.py`, `tokenizer/`, `data/`,
`model/`, `train/`, `generate.py`, `cli.py`, `utils.py`). Delete the `sys.path` hack in
`main.py`, replace `main.py` with `cli.py`. Entry point: `uv run our-llm <cmd>`.
**Why.** Today the editable install's `.pth` puts **`src/` itself** on the path, so
`import tokenizer` / `from attentions... import` only work by accident and clash with
third-party names (`data`, `tokenizer`, `attentions` are all terrible top-level names).
**Consequence.** Imports become `from our_llm.model.gpt import GPTModel`; tests drop
`from src.` too.

### D2 — Torch is the production path, pure Python is the teaching path
**Decision.** The pipeline that trains is **PyTorch only**. The pure-Python versions
(`data_loader.py`, `embedding.py`, BPE internals) move to `examples/pure_python/` (still
tested, still importable) and are never on the hot path.
**Why.** Two parallel implementations double the maintenance and forced the
sinusoidal-vs-learned inconsistency. Keeping them as *examples* preserves the didactic value.
**Consequence.** `main.py`'s demo chain is retired; the story it told now lives in `examples/`.

### D3 — Tokenizer: keep our own BPE (not `tiktoken`), make it fast + versioned
**Decision.** Keep `src/our_llm/tokenizer/bpe.py`. Three upgrades:
1. **Fast encode** via merge-rank priority queue (GPT-2 style) instead of replaying all
   `N` merges → O(word len) per word instead of O(N × words). Target: full corpus < 5 s.
2. **Special tokens**: `<|endoftext|>` (doc separator) and `<|pad|>` (dataloader padding);
   `vocab_size = 256 + 2 + len(merges)`.
3. **Cache key** = `sha256(corpus) + num_merges`, stored as `data/cache/vocab-<hash>.json`
   (+ `.merges`). Stale cache becomes impossible.
**Why.** `tiktoken` would make the project a wrapper; the current encode cost and the
stale cache are the two real defects. Keep training comparison-able with the book's GPT-2 numbers.
**Alternative rejected.** Ship `tiktoken` "gpt2" like the book → loses the from-scratch premise.

### D4 — Positional embeddings: **learned** by default, sinusoidal behind a flag
**Decision.** `GPTConfig.pos_emb: Literal["learned", "sinusoidal"] = "learned"`.
**Why.** Learned matches `parts/` and `DummyGPTModel` and GPT-2 parity (needed if we ever
load GPT-2 weights). `todo.txt` asks for the *Attention Is All You Need* version — keep it as a
real experiment (`our-llm train --pos-emb sinusoidal`) and compare val loss. `src/embedding.py`
already has the sinusoidal code to reuse.
**Consequence.** `cfg["context_length"]` becomes a hard limit for learned PE.

### D5 — Data: encode **once** to a binary token file, then memmap
**Decision.** `our-llm tokenize` writes `data/processed/tokens-<corpus_hash>.bin` as `uint16`
(vocab 15k < 65536) + a small `.meta.json` (vocab hash, dtype, n_tokens, doc offsets).
Training reads it with `numpy.memmap` → 0 s startup.
**Why.** Encoding costs minutes; the corpus does not change between runs. Also enables exact
reproducibility. Split into train/val **at document boundaries** (90/10 by default) to avoid
leaking text between splits.
**Fallback.** `dtype=uint32` if we ever cross 65,535 tokens in the vocab.

### D6 — Attention: one class, two code paths
**Decision.** `MultiHeadAttention` with `qkv_bias` and optional `F.scaled_dot_product_attention`
(fused, fast) — `--sdpa/--manual` for the didactic comparison. Causal mask is a registered
buffer, truncated to `T`. Delete the module-level demo code from all attention files
(keep demos in `examples/` or notebooks).
**Why.** Fixes the `NameError: batch` import crash and keeps one implementation to trust.
**Sanity test.** `num_heads=1` must equal the single-head `CausalAttention` output.

### D7 — Norm / FFN: keep hand-written LayerNorm + tanh-GELU
**Decision.** Hand-written `LayerNorm` (scale/shift, `unbiased=False`) and tanh-approx GELU,
as in the book, with tests asserting they match `nn.LayerNorm` / `nn.GELU(approximate="tanh")`
within tolerance. `todo.txt` asks for the exact-GELU derivative derivation → do it in the notes,
not the model.
**Why.** Educational value + a cheap equivalence test that proves correctness.

### D8 — Model init + size presets
**Decision.** std=0.02 normal init for weights/biases, residual projections scaled by
`1/sqrt(2*n_layers)`. Presets in `config.py`:

| preset | ctx | emb | heads | layers | vocab | ≈params |
|---|---|---|---|---|---|---|
| `debug` | 64 | 64 | 4 | 2 | 15,228 | 0.2 M |
| `tiny` | 256 | 384 | 6 | 4 | 15,228 | **≈13 M** |
| `gpt2-small` | 1024 | 768 | 12 | 12 | 50,257 | 163 M |

`todo.txt` line 1 says **13.37 M**. `tiny` as specced above (tied, vocab 15,228, ctx 256,
D=384, L=4, heads 6) measures **13,039,872** — verified by `GPTConfig.num_parameters()`
and pinned in `tests/test_config.py`. Hitting exactly 13.37 M needs one dim retuned
(e.g. ctx 1,116); low priority, but keep the pinned number in sync if the preset changes.
Untied `out_head` would instead add `V*D` = 5,847,552 → 18,887,424.
For reference: `gpt2-small` measures 124,412,160 and `debug` 1,078,400.
**Why.** A parameter count is the cheapest end-to-end correctness signal we have.
`vocab_size` must come from the tokenizer, never be hardcoded.

### D9 — Weight tying: off by default, on for `tiny`
**Decision.** `tie_embeddings: bool`. Tying `out_head.weight = tok_emb.weight` saves `V*D`
params (≈5.9 M at `tiny`) — that is what makes 13 M reachable.

### D10 — Training recipe
**Decision.**
- Loss: `F.cross_entropy(logits.flatten(0,1), targets.flatten())` (+ `perplexity = exp(loss)`).
- Optimizer: `AdamW`, and **weight decay only on 2-D params** (exclude norms/biases) — the book
  skips this; it matters.
- Grad clip 1.0; LR warmup (100 steps) + cosine decay to 10%.
- Eval every `eval_freq` steps on `eval_iter` batches, **save best-by-val** checkpoint,
  log a CSV (`runs/<ts>/metrics.csv`), sample one text per epoch.
- Seeding: `torch.manual_seed`, `numpy`, `random` — one `seed_everything()`.
**Baseline.** Book's `train_model_simple`: AdamW lr=4e-4 wd=0.1, 10 epochs, batch 2, ctx 256.

### D11 — Checkpointing
**Decision.** `save_checkpoint(path, model, optimizer, step, cfg, tokenizer_meta)` as a dict;
`load_checkpoint` restores and asserts `cfg` matches the model. Store the tokenizer vocab
hash **inside** the checkpoint so a mismatched vocab fails loudly instead of producing garbage.

### D12 — Sampling
**Decision.** `generate()` with `temperature`, `top_k`, `top_p`, `repetition_penalty`, greedy
(`temperature=0`). One function, not four. KV cache **later** (stretch), guarded by a test that
cached and uncached outputs are identical.

### D13 — CLI + config
**Decision.** `argparse` (stdlib, no new deps) subcommands:
```
uv run our-llm tokenize [--corpus data/] [--num-merges 15000]
uv run our-llm train    --preset tiny [--pos-emb learned|sinusoidal] [--epochs 10]
uv run our-llm sample   --ckpt runs/<id>/best.pt --prompt "The monster" --temperature 0.8 --top-k 40
uv run our-llm eval     --ckpt ...          # train/val loss + perplexity
```
All knobs live in frozen dataclasses in `config.py`; CLI is a thin parser over them.

### D14 — Testing strategy
**Decision.** The single most valuable test is **overfit one batch**: a 2-layer debug model must
drive loss < 0.1 on a single batch within a few hundred steps. Plus:
attention parity (1 head vs single head), `nn.LayerNorm`/`nn.GELU` equivalence, loss matches a
hand-computed example, decode(encode(x)) == x (incl. UTF-8 + special tokens), token cache
round-trip, checkpoint round-trip, sampler determinism at `temperature=0`.

### D15 — Quality gates
**Decision.** `ruff check` + `ruff format` + `mypy --strict` on `src/our_llm`, `pytest -q`,
GitHub Actions running all three. `README.md` written from the notebook notes
(`todo.txt` line 15) — it is currently empty, which is the biggest repo-level gap.

### D16 — Repo hygiene
**Decision.** Drop `data/gg` (duplicate corpus), honour `DATA_EXTENSIONS` or delete the constant,
gitignore `data/processed/`, `runs/`, `data/cache/`. `parts/` is already gitignored — copy the
few cells we actually port into `notes/` so the reasoning survives outside a 500-cell notebook.

### D17 — M3 implementation details
**Decision.**
- **Compute choice is a constructor argument, architecture is config.** `GPTModel(cfg, use_sdpa=False)`
  — sdpa vs manual produces identical logits, so it must not appear in a checkpoint's config.
- **Derived state stays out of checkpoints.** The causal mask and the sinusoidal table are
  `persistent=False` buffers: recomputed on creation, so a `.pt` holds only learned tensors and
  no `context_length`-shaped junk.
- **Init before tying.** `apply(_init_weights)` first, then `out_head.weight = tok_emb.weight`,
  then the residual scaling — otherwise the Linear init would overwrite the shared embedding.
- **Residual dropout lives in the block** (`drop_shortcut`) around `att` and `ff`, and `FeedForward`
  keeps its internal dropout at 0 by default, so dropout is applied once per sub-layer, not twice.
- **Sinusoidal positions cost zero parameters**, so `GPTConfig.num_parameters()` returns
  `0` for `pos` when `pos_emb == "sinusoidal"`; the live model and the formula still agree.
**Why.** These four are the places where a checkpoint silently changes meaning (or a param count
silently drifts) if done differently. Tests pin all of them in `tests/test_model.py`.

### D18 — CLI: `fire` + one module per stage + a `pipeline` command
**Decision.**
- **`fire` instead of `argparse`.** Each stage is a plain function; `OurLLM` in `app.py` exposes it
  as a Fire command. Signatures (types + defaults + docstring `Args:`) *are* the CLI, so there is
  no second place to update when a parameter is added. `--num-merges` == `--num_merges`.
- **One file per stage** in `src/our_llm/tasks/` (`tokenize.py`, `train.py`, `evaluate.py`,
  `sample.py`) plus `pipeline.py` that chains them. `tasks/__init__.py` re-exports all of them and
  a `COMMANDS` dict; `app.py` is the only Fire-aware file, `main.py` is 8 lines.
- **`pipeline()` = tokenize → train → eval → sample**, passing each stage's output to the next
  (the token file, then the checkpoint). Three statuses: `ok`, `skipped` (stage raises
  `NotImplementedError` — its milestone has not landed), `failed` (anything else, stops the run).
  `--stages all|tokenize,train` selects a subset, always executed in pipeline order.
- **Flags are validated before Fire runs.** Fire silently drops an unknown `--flag` and runs with
  defaults, which for `train` would start a long run at the wrong settings; `app.validate_flags`
  turns that into an error with a "did you mean" hint.
- **`cli.py` (argparse) is deleted**, not kept in parallel — two CLIs drift.
**Why.** The pipeline stages must be callable from a notebook, a test and the shell, and adding a
flag should be a one-line change. The `skipped` status is what makes the entry point usable during
M4/M5: `pipeline` works today and will keep working as each stage gets its body.
**Also.** `data/split.py` was extracted from `data/dataset.py` (and `data/__init__.py` resolves the
torch-backed names lazily) so `our_llm.tasks` and `our-llm tokenize` no longer import torch:
`tokenize` startup dropped the ~1.5 s torch import.

---

## 2. Target file layout

```
src/our_llm/
├── __init__.py          # public API: GPTModel, GPTConfig, train, generate
├── config.py            # dataclasses + presets (debug/tiny/gpt2-small)
├── app.py               # fire component: tokenize | train | eval | sample | pipeline
├── tasks/               # one module per stage, no torch
│   ├── tokenize.py      # corpus -> BPE -> .bin + split summary  (done)
│   ├── train.py         # pretrain                                         (M4)
│   ├── evaluate.py      # loss / perplexity                                (M4)
│   ├── sample.py        # generate()                                       (M5)
│   └── pipeline.py      # runs the stages above in order
├── utils.py             # device, seed_everything, count_parameters, logging
├── tokenizer/
│   ├── bpe.py           # train / fast encode / decode  (from src/tokenizer.py)
│   └── special.py       # <|endoftext|>, <|pad|>
├── data/
│   ├── corpus.py        # find files (ext-filtered), read, dedupe, doc boundaries
│   ├── split.py         # doc-level train/val split (pure Python, torch-free)
│   ├── tokens.py        # encode-once .bin writer + memmap reader
│   └── dataset.py       # torch Dataset (sliding window) + train/val dataloaders
├── model/
│   ├── layers.py        # LayerNorm, GELU, FeedForward
│   ├── attention.py     # CausalAttention, MultiHeadAttention (sdpa + manual)
│   └── gpt.py           # GPTModel, init, tie_embeddings, count_parameters
├── train/
│   ├── loss.py          # calc_loss_batch, calc_loss_loader, perplexity
│   ├── loop.py          # train_model, evaluate_model, scheduler, CSV log
│   └── checkpoint.py    # save/load
├── generate.py          # generate(): greedy/temperature/top-k/top-p
examples/pure_python/    # old embedding.py / data_loader.py / attention demos
notes/                   # distilled notebook notes (source for README)
tests/                   # mirrors src tree
```

---

## 3. Phased plan

### M0 — Clean slate (½ day) — **done**
- [x] Stop the double-load: `our_llm/data/corpus.py` filters by extension (kept `data/gg`
      on disk rather than deleting user data; it is now simply ignored)
- [x] Create `our_llm/` skeleton + `config.py` with the 3 presets
- [x] Remove the `sys.path` hack: `main.py` is now a 5-line alias for `our-llm`
- [x] `Makefile`: `make tokenize|train|sample|test|overfit|lint|fmt`
- [x] Move the old pipeline demo to `examples/pure_python/pipeline_demo.py`
- [x] Delete module-level demo code from `src/attentions/*` (now import-safe; `NameError` gone)
- [x] `.gitignore`: `data/cache/`, `data/processed/`, `runs/`, tool caches

### M1 — Tokenizer v2 (1–2 days) — **done**
- [x] `our_llm/tokenizer/bpe.py`: rank-based encode + per-word memoisation
      (**0.08 s** for 438 KB, down from ~180 s — ~2000x)
- [x] `<|endoftext|>` (256) / `<|pad|>` (257), merges from 258; `vocab_size = 258 + merges`
- [x] Versioned cache `data/cache/vocab-<corpus_sha16>-m<merges>.json` + legacy migration on load
- [x] Full retrain on the real corpus: 68 s → `vocab-905e38dc26f492e1-m14970.json`
- [x] Tests: 52 new (round-trip incl. UTF-8/specials, fast-vs-slow equivalence,
      merge-for-merge parity with `src/tokenizer.py`, cache keying, legacy migration)

### M2 — Data pipeline (done)
- [x] `corpus.py`: reader with extension filter + document boundaries (`--granularity file|paragraph`)
- [x] `tokens.py`: `our-llm tokenize` → `uint16` `.bin` + `.meta.json` (corpus/structure/vocab hashes, n_tokens, doc spans)
- [x] `dataset.py`: torch sliding-window dataset, tensors, 90/10 doc-level split, seeded loaders
- [x] Tests: window shift, no train/val leakage, memmap round-trip (179 passing total)

### M3 — Model — **done**
- [x] `model/layers.py`: hand-written `LayerNorm` (biased var, scale/shift) and tanh `GELU`,
      `FeedForward` (4x expand) — each checked against `nn.LayerNorm` / `nn.GELU(approximate="tanh")`
- [x] `model/attention.py`: `MultiHeadAttention` with causal mask (non-persistent bool buffer),
      dropout, `qkv_bias`, and `use_sdpa` — manual and sdpa paths agree to 1e-6
- [x] 1-head parity: equals legacy `CausalAttention` once `out_proj` is the identity
- [x] `model/gpt.py`: `GPTModel` = tok/pos emb + `n_layers` pre-norm blocks + final norm + `out_head`,
      weight tying, and GPT-2 init (std 0.02, residual projections / sqrt(2L))
- [x] `pos_emb="sinusoidal"` works and costs **0** parameters (`cfg.num_parameters()` updated)
- [x] `count_parameters(model)` == `cfg.num_parameters()`: debug 1,078,400 / tiny **13,039,872**
- [x] Forward-shape test `(B,T)` -> `(B,T,vocab)`; context overflow, bad rank, causality all tested
- [x] Tests: 32 new in `tests/test_model.py` (211 passing total)

### M4 — Loss + loop (2 days)
- [x] `tasks/train.py` + `tasks/evaluate.py` exist with the final signatures (the CLI and
      `pipeline` already call them; they raise `NotImplementedError` until the loop lands)
- [ ] `loss.py`: cross-entropy + perplexity + `calc_loss_loader`
- [ ] `loop.py`: `train_model`, `evaluate_model`, AdamW (decay only 2-D), warmup+cosine, grad clip, CSV log
- [ ] `checkpoint.py`: save/load incl. vocab hash assert — `train` returns the checkpoint path,
      which is what `pipeline` hands to `eval`/`sample`
- [ ] **Overfit-one-batch test** green at `debug` preset
- [ ] `our-llm train` end-to-end on `data/dd.txt` (`make pipeline STAGES=all`)

### M5 — Pretrain + sample (1–2 days)
- [ ] `generate.py`: greedy / temperature / top-k / top-p / repetition penalty
- [ ] `our-llm sample` + printed samples per epoch during training
- [ ] Baseline run on Frankenstein: record loss/perplexity curve in `notes/`
- [ ] Experiment: `--pos-emb learned` vs `sinusoidal` (closes `todo.txt` line 6)

### M6 — Docs + CI (1 day)
- [ ] `README.md` from `notes/` (architecture diagram, decisions, usage)
- [ ] ruff + mypy strict + pytest in GitHub Actions
- [ ] `notes/` distilled from `parts/` (loss, perplexity, training loop, temperature)

### M7 — Stretch
- [ ] Load released GPT-2 weights into `gpt2-small` (needs `tiktoken` compat or a vocab remap)
- [ ] Ch6 fine-tuning (classification) / ch7 instruction tuning
- [ ] KV-cache decode (with a cached==uncached equivalence test)
- [ ] AMP / `torch.compile`, 4-bit inference
- [ ] Train on a bigger corpus than 441 KB

---

## 4. Risks / known traps

1. **Vocab mismatch** — `vocab_size` hardcoded `50257` in `llm_class.py:90` will silently work but
   produce a wrong `out_head` shape once our tokenizer (15,228) is wired in.
2. **Stale tokenizer cache** — the cache at repo root was trained on the *old* corpus; any run today
   trains against mismatched merges. M1 fixes this; until then, delete both JSON files to retrain.
3. **Overfitting is expected** — book's val loss plateaus at 6.37 while train hits 0.39 on a
   20k-char story. Don't "fix" it with more epochs; the fix is more data (M7).
4. **`parts/` is gitignored** — the notebook knowledge is not in the repo. Anything we rely on must
   be copied into `notes/`.
5. **Python 3.14** — `torch` on 3.14 is bleeding edge; if an op misbehaves, that is the first suspect.
