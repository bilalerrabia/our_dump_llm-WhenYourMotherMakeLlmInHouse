# Project Memory — our-llm

## What this is
A learn-by-building GPT project ("When your mother makes an LLM in house"):
from-scratch BPE tokenizer, data loader, embeddings in **pure Python**, then
PyTorch attention/GPT modules following *Build a LLM from Scratch* (Raschka).
Corpus: `data/dd.txt` (441 KB, Project Gutenberg Frankenstein) **plus** `data/gg`
(440 KB, near-duplicate without extension — `find_data_files` has no extension filter,
so the same book is loaded twice). `README.md` is empty; notes live in `todo.txt`
and notebooks under `src/*/` and `parts/` (Loss function, Pretraining).

## Layout
- `main.py` — entry point: read `data/*.txt` -> tokenize -> sliding-window batches -> embeddings.
  Config constants at top (CONTEXT_WINDOW=4, STRIDE=4, BATCH_SIZE=8, OUTPUT_DIM=256, MERGE_RATIO=10).
  num_merges = len(text) // 10; vocab_size = 256 + len(tokenizer.map_replace).
- `src/tokenizer.py` — `Tokenizer`: BPE, GPT-2 regex pre-tokenizer, `map_replace: {new_id: (l, r)}`,
  `train/encode/decoder/save_vocab/load_vocab` (vocab.json + derived `*_merges.json`).
- `src/data_loader.py` — pure-Python `DatasetPreparation` (input = window, target = shifted by 1)
  + `create_dataloader` generator (shuffle/drop_last).
- `src/embedding.py` — pure-Python `Embedding` lookup, `sinusoidal_positional_encoding`,
  `add_embeddings` (broadcast (B,S,D)+(S,D)), `get_shape`.
- `src/attentions/` — PyTorch, executed at import (module-level demo code):
  `simplified_attention.py`, `self_attention.py` (+`SelfAttention` class),
  `causel_attention.py` (`CausalAttention`, note the filename typo),
  `multi_head_attention.py` (`MultiHeadAttention`, references undefined `batch` at module level -> crashes if imported).
- `src/LLM_arch/llm_class.py` — GPT skeleton: `DummyGPTModel`, `LayerNorm`, `GELU`,
  `FeedForward`, `TransformerBlock`; cfg dict (GPT-2 124M sizes: vocab 50257, ctx 1024, emb 768).
- `src/our_llm/__init__.py` — stub `main()` (declared as console script `our-llm` in pyproject).
- Tests: `tests/` cover tokenizer, data_loader, embedding — **53 passed** via `uv run pytest -q`
  (or `make test`). `tests/conftest.py` makes `src` importable.

## Commands
- Run: `make` (uv sync + `uv run python3 main.py`) or `uv run python main.py`
- Test: `make test`
- Python >= 3.14, managed with uv; deps: numpy, regex, torch, tqdm (+ pytest dev).
- Notebooks contain the full book walkthrough; `src/` modules are the distilled versions.

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
weights, `torch.utils.data.DataLoader`. Key APIs to port into `src/`:
`GPTModel` (full, vs our `DummyGPTModel`), `generate_text_simple`, `text_to_token_ids`/`token_ids_to_text`,
`calc_loss_batch` (`cross_entropy(logits.flatten(0,1), targets.flatten())`), `calc_loss_loader`,
`evaluate_model`, `generate_and_print_sample`, `softmax_with_temperature`, `print_sampled_tokens`,
`GPTDatasetV1`/`create_dataloader_v1` (= our `DatasetPreparation`/`create_dataloader` with tensors).
Book hyperparams: ctx 1024 -> **256** for CPU, batch 2, stride=ctx, 90/10 split,
AdamW lr=4e-4 wd=0.1, 10 epochs, eval_freq=5/eval_iter=5; results train 9.781->0.391,
val 9.856->6.372 (overfits a 20k-char story).
- Gotcha: `tiktoken`, `matplotlib`, `pandas` are **not** installed and `the-verdict.txt` is not in
  the repo, so these notebooks cannot be run as-is here.

## Conventions / gotchas
- Tests import as `from src.tokenizer import ...`; `main.py` inserts `src/` into sys.path.
  **The editable install already puts `src/` itself on `sys.path`** (`site-packages/our_llm.pth`
  contains only the `src` dir), so `import tokenizer`, `from attentions... import` work by
  accident and `our_llm/` is not a real package root. ROADMAP D1 fixes this.
- Perf fact: `Tokenizer.encode` replays every merge → **0.83 s per 2,000 chars** (~3 min for the
  full corpus, every run). Encoding is never cached to disk (ROADMAP D5/M2).
- `parts/` is **gitignored** — notebook knowledge must be copied into `notes/` to survive in git.
- `data/gg` is a near-duplicate of `data/dd.txt` (extensionless, so it is loaded too).
- `python 3.14` + `torch` — if an op misbehaves, 3.14 is the first suspect.
- `ROADMAP.md` holds the full pipeline plan + 16 architecture decisions (D1-D16) with rationale.
  Change a decision by editing that table, not silently in code.
- `tokenizer_vocab.json` / `tokenizer_vocab_merges.json` are the trained cache at repo root
  (14,970 merges -> vocab 15,226, trained 2025-10-23 on the older corpus); delete them to
  force retraining. Cache is never invalidated when the data changes.
- `DATA_EXTENSIONS` in main.py is declared but **not used** by `find_data_files`.
- `.env` holds an API key — never commit it.
- Style: type hints, module docstrings, small pure functions. `todo.txt` tracks next steps
  (GELU derivative, lint-strict, README from notebook notes).
