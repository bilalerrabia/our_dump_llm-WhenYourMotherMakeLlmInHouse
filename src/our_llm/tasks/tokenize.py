"""Stage 1 — corpus -> BPE vocabulary -> token file.

Everything the rest of the pipeline needs to start: a tokenizer that matches the
corpus, a `uint16` token file plus its `.meta.json` sidecar, and the
document-level train/val split summarised.

This module is the *implementation*; the CLI in :mod:`our_llm.app` only exposes
it. Methods live in their own files so each stage can be imported (and tested)
without dragging the others in.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from our_llm.config import DEFAULT_NUM_MERGES
from our_llm.data.corpus import Corpus, read_corpus
from our_llm.data.split import split_documents
from our_llm.data.tokens import (
    check_compatible,
    encode_corpus,
    load_meta,
    meta_path,
    token_file_path,
    write_tokens,
)
from our_llm.tokenizer.bpe import (
    BPETokenizer,
    corpus_hash,
    load_or_train,
    tokenizer_cache_path,
)

#: Cache left behind by the first tokenizer implementation (merges start at 256).
LEGACY_CACHE = "tokenizer_vocab_merges.json"


@dataclass(frozen=True, slots=True)
class TokenizeResult:
    """What stage 1 produced -- passed along the pipeline instead of re-reading."""

    vocab_size: int
    num_merges: int
    n_tokens: int
    n_documents: int
    train_docs: int
    val_docs: int
    train_tokens: int
    val_tokens: int
    token_file: str | None
    fingerprint: str
    reused: bool

    def __str__(self) -> str:
        where = self.token_file or "not written"
        return (
            f"tokenize: {self.n_tokens:,} tokens, vocab {self.vocab_size:,} "
            f"({self.num_merges:,} merges), train {self.train_docs} docs/"
            f"{self.train_tokens:,} tok, val {self.val_docs} docs/{self.val_tokens:,} tok -> {where}"
        )


def pick_tokenizer(
    corpus: Corpus,
    num_merges: int = DEFAULT_NUM_MERGES,
    cache_dir: str = "data/cache",
    vocab_cache: str | None = None,
    retrain: bool = False,
) -> tuple[BPETokenizer, str]:
    """Choose a tokenizer: explicit file -> retrain -> corpus cache -> legacy -> train.

    The vocabulary depends only on the raw characters, so the cache key is the
    files hash (not the document layout, which changes with ``--granularity``).
    """
    key = corpus.files_hash or corpus_hash(corpus.text)
    path = tokenizer_cache_path(cache_dir, key, num_merges)

    if retrain:
        tokenizer = BPETokenizer().train(corpus.text, num_merges)
        tokenizer.save(path)
        return tokenizer, f"trained on this corpus -> {path}"

    if vocab_cache:
        return BPETokenizer.load(vocab_cache), f"loaded from {vocab_cache}"

    if os.path.exists(path):
        return BPETokenizer.load(path), f"cache hit {path}"

    if os.path.exists(LEGACY_CACHE):
        legacy = BPETokenizer.load(LEGACY_CACHE)
        # Only a drop-in replacement when the merge count matches what was asked
        # for -- otherwise every downstream checkpoint would have the wrong vocab.
        if legacy.satisfies(num_merges):
            return legacy, (
                f"loaded from {LEGACY_CACHE} (legacy file, merges shifted to the 258+ "
                f"layout; use --retrain for a corpus-matched vocabulary)"
            )

    tokenizer = load_or_train(corpus.text, num_merges, cache_dir, key=key)
    return tokenizer, f"trained -> {path}"


def tokenize(
    corpus: str = "data",
    num_merges: int = DEFAULT_NUM_MERGES,
    granularity: str = "paragraph",
    val_ratio: float = 0.10,
    cache_dir: str = "data/cache",
    processed_dir: str = "data/processed",
    vocab_cache: str | None = None,
    retrain: bool = False,
    force: bool = False,
    write: bool = True,
    quiet: bool = False,
) -> TokenizeResult:
    """Train/load the BPE tokenizer and encode the corpus to a token file.

    Args:
      corpus: A corpus file or directory (default ``data``).
      num_merges: Merges to learn when a retrain happens.
      granularity: ``file`` or ``paragraph`` — what counts as one document, and
        therefore what the train/val split is made of.
      val_ratio: Fraction of documents held out for validation.
      cache_dir: Where vocabulary caches live.
      processed_dir: Where the token file and its metadata are written.
      vocab_cache: Load these merges instead of training/finding a cache.
      retrain: Ignore caches and learn the merges again.
      force: Re-encode even when a current token file already exists.
      write: Set to false (`--nowrite`) to report only, leaving the disk untouched.
      quiet: Suppress the progress report.
    """
    say = (lambda *a, **k: None) if quiet else print

    corpus_data = read_corpus(corpus, granularity=granularity)
    if not corpus_data:
        raise FileNotFoundError(f"No corpus files found in {corpus!r}")

    sources = sorted(set(corpus_data.sources))
    say(f"Corpus : {len(corpus_data.docs)} {granularity} document(s) from {len(sources)} file(s)")
    for source in sources:
        say(f"         - {source}")
    say(f"         {len(corpus_data.text):,} characters, sha256[:16]={corpus_data.fingerprint}")

    tokenizer, provenance = pick_tokenizer(corpus_data, num_merges, cache_dir, vocab_cache, retrain)
    say(f"Vocab  : {tokenizer.vocab_size:,} tokens ({tokenizer.num_merges:,} merges)")
    say(f"         {provenance}")

    # Encode document by document (that is what yields splittable spans) and
    # reuse the file when it is still valid.
    path = token_file_path(processed_dir, corpus_data.structure_fingerprint, tokenizer.num_merges)
    cached = os.path.exists(path) and os.path.exists(meta_path(path))

    start = time.perf_counter()
    reused = False
    token_file: str | None = None

    if cached and not force:
        meta = load_meta(path)
        check_compatible(meta, tokenizer)  # raises if the vocabulary changed underneath
        n_tokens = int(meta["n_tokens"])
        reused = True
        token_file = path
        say(f"Tokens : {n_tokens:,} (already current: {path}, nothing rewritten)")
    else:
        encoded = encode_corpus(tokenizer, corpus_data)
        n_tokens = encoded.n_tokens
        if write:
            meta = write_tokens(path, encoded, corpus=corpus_data, tokenizer=tokenizer)
            token_file = path
        else:
            meta = {
                "dtype": str(encoded.ids.dtype),
                "n_documents": len(corpus_data.docs),
                "doc_spans": encoded.doc_spans,
            }
        elapsed = time.perf_counter() - start
        ratio = len(corpus_data.text) / max(n_tokens, 1)
        note = "" if write else " (not written: --nowrite)"
        say(f"Tokens : {n_tokens:,} ({ratio:.2f} chars/token, encoded in {elapsed:.2f}s){note}")
    say(f"         dtype {meta['dtype']}, {meta.get('n_documents', 0)} document(s) of tokens")

    spans = [(int(s), int(e)) for s, e in meta.get("doc_spans") or []]
    train_docs, val_docs = split_documents(len(spans) or len(corpus_data.docs), val_ratio)
    train_tokens = sum(e - s for s, e in (spans[i] for i in train_docs if i < len(spans)))
    val_tokens = sum(e - s for s, e in (spans[i] for i in val_docs if i < len(spans)))
    say(
        f"Split  : train {len(train_docs)} docs / {train_tokens:,} tokens, "
        f"val {len(val_docs)} docs / {val_tokens:,} tokens (val_ratio {val_ratio})"
    )
    if spans and not val_docs:
        say(
            "         note: only one document -> nothing to validate on "
            "(try --granularity paragraph)"
        )

    return TokenizeResult(
        vocab_size=tokenizer.vocab_size,
        num_merges=tokenizer.num_merges,
        n_tokens=n_tokens,
        n_documents=int(meta.get("n_documents", len(corpus_data.docs))),
        train_docs=len(train_docs),
        val_docs=len(val_docs),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        token_file=token_file,
        fingerprint=corpus_data.fingerprint,
        reused=reused,
    )
