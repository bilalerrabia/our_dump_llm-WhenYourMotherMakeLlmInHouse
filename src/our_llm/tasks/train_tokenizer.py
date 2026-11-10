"""Stage 0 — learn the BPE vocabulary ("train the tokenizer", ROADMAP M1).

`tokenize` only *uses* a vocabulary: it loads a cache when one exists and trains
one when it does not. This stage is the explicit counterpart — it guarantees the
merges exist, trains them when they do not, and reports where they came from, so
the pipeline can say "the tokenizer was trained" instead of "a cache was found".

    uv run python main.py train-tokenizer                 # train if missing
    uv run python main.py train-tokenizer --retrain       # learn them again (~68 s)
    uv run python main.py train-tokenizer --num-merges 1000

Deterministic: training the same corpus with the same `num_merges` twice gives
the same merges, so the cache is the only reason to train a second time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from our_llm.config import DEFAULT_NUM_MERGES
from our_llm.data.corpus import Corpus, read_corpus
from our_llm.tokenizer.bpe import BPETokenizer, corpus_hash, tokenizer_cache_path


@dataclass(frozen=True, slots=True)
class TokenizerTrainResult:
    """What stage 0 produced -- the vocabulary the rest of the run will use."""

    vocab_size: int
    num_merges: int
    corpus_chars: int
    n_documents: int
    fingerprint: str
    cache_path: str
    seconds: float
    trained: bool

    def __str__(self) -> str:
        how = f"trained in {self.seconds:.1f}s" if self.trained else "already trained"
        return (
            f"train-tokenizer: vocab {self.vocab_size:,} ({self.num_merges:,} merges from "
            f"{self.corpus_chars:,} chars) -> {self.cache_path} ({how})"
        )


def train_tokenizer(
    corpus: str = "data",
    num_merges: int = DEFAULT_NUM_MERGES,
    granularity: str = "paragraph",
    cache_dir: str = "data/cache",
    retrain: bool = False,
    quiet: bool = False,
) -> TokenizerTrainResult:
    """Learn the BPE merges for this corpus and cache them.

    Args:
      corpus: A corpus file or directory (default ``data``).
      num_merges: How many merges to learn.
      granularity: `file` or `paragraph` — irrelevant to the merges themselves
        (they depend on the raw characters), but it decides how the corpus is read.
      cache_dir: Where the vocabulary cache is written.
      retrain: Learn the merges again even when a matching cache exists.
      quiet: Suppress the progress report and the training progress bar.
    """
    say = (lambda *a, **k: None) if quiet else print

    corpus_data = read_corpus(corpus, granularity=granularity)
    if not corpus_data:
        raise FileNotFoundError(f"No corpus files found in {corpus!r}")

    key = corpus_data.files_hash or corpus_hash(corpus_data.text)
    path = tokenizer_cache_path(cache_dir, key, num_merges)

    say(f"Corpus : {len(corpus_data.docs)} {granularity} document(s)")
    say(f"         {len(corpus_data.text):,} characters, sha256[:16]={corpus_data.fingerprint}")

    start = time.perf_counter()
    trained = False
    tokenizer: BPETokenizer

    if os.path.exists(path) and not retrain:
        tokenizer = BPETokenizer.load(path)
        if not tokenizer.satisfies(num_merges):  # the cache name says otherwise; trust nothing
            trained = True
            tokenizer = _learn(corpus_data, num_merges, path, quiet)
            seconds = time.perf_counter() - start
        else:
            seconds = 0.0
            say(f"Merges : {num_merges:,} already trained -> {path}")
    else:
        trained = True
        tokenizer = _learn(corpus_data, num_merges, path, quiet)
        seconds = time.perf_counter() - start
        say(f"Merges : {num_merges:,} learned in {seconds:.1f}s -> {path}")

    say(f"Vocab  : {tokenizer.vocab_size:,} tokens ({tokenizer.num_merges:,} merges)")

    return TokenizerTrainResult(
        vocab_size=tokenizer.vocab_size,
        num_merges=tokenizer.num_merges,
        corpus_chars=len(corpus_data.text),
        n_documents=len(corpus_data.docs),
        fingerprint=corpus_data.fingerprint,
        cache_path=path,
        seconds=seconds,
        trained=trained,
    )


def _learn(
    corpus: Corpus,
    num_merges: int,
    path: str,
    quiet: bool,
) -> BPETokenizer:
    """Train the merges and write the cache (the slow branch)."""
    tokenizer = BPETokenizer().train(corpus.text, num_merges, show_progress=not quiet)
    tokenizer.save(path)
    return tokenizer
