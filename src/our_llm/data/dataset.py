"""Torch datasets and dataloaders: token ids -> (inputs, targets) batches.

Two ideas carry the whole file:

1. **Sliding window.** For a start ``i``, ``inputs = ids[i:i+T]`` and
   ``targets = ids[i+1:i+T+1]`` — the shift is what makes it next-token
   prediction. ``stride`` is how far the window moves per sample; ``stride == T``
   is non-overlapping, ``stride == 1`` is maximal overlap (memory is cheap here
   because samples are views into one memmapped array, not copies).
2. **Split by document, never by token.** Slicing a token stream at 90% would put
   the second half of a sentence in training and its first half in validation —
   the model would look better than it is. Whole documents go to one side or the
   other, and documents are separated by `<|endoftext|>` so no window ever spans
   two of them meaningfully.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from our_llm.config import DataConfig
from our_llm.data.corpus import Corpus, read_corpus
from our_llm.data.split import split_documents
from our_llm.data.tokens import (
    check_compatible,
    encode_corpus,
    load_meta,
    meta_path,
    open_tokens,
    token_file_path,
    write_tokens,
)
from our_llm.tokenizer.bpe import BPETokenizer, load_or_train

Span = tuple[int, int]


class GPTDataset(Dataset):
    """Sliding-window next-token dataset over a flat array of token ids."""

    def __init__(
        self,
        ids: Sequence[int] | np.ndarray,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        self.ids = ids
        self.context_length = context_length
        self.stride = context_length if stride is None else stride
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        # A sample needs context_length inputs *and* one extra token for the
        # last target, so a stream of exactly T tokens yields zero samples.
        self._n = (
            max(0, (len(ids) - context_length - 1) // self.stride + 1)
            if len(ids) > context_length
            else 0
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self._n:
            raise IndexError(f"index {index} out of range for {self._n} samples")
        start = index * self.stride
        window = self.ids[start : start + self.context_length + 1]
        # `np.array` copies explicitly: it widens uint16 -> int64 for the
        # embedding lookup and avoids holding a read-only memmap view in a tensor.
        chunk = torch.from_numpy(np.array(window, dtype=np.int64))
        return chunk[:-1], chunk[1:]

    def __repr__(self) -> str:
        return (
            f"GPTDataset(n_tokens={len(self.ids)}, context_length={self.context_length}, "
            f"stride={self.stride}, n_samples={self._n})"
        )


# --- splitting --------------------------------------------------------------

# `split_documents` lives in `data/split.py` (pure Python) so the tokenizer stage
# can report the split without importing torch; it is imported above and
# re-exported from here because this is where callers look for it.


def gather(ids: np.ndarray, spans: Sequence[Span], document_indices: Sequence[int]) -> np.ndarray:
    """Concatenate the selected documents into one contiguous array."""
    if not document_indices:
        return np.empty(0, dtype=ids.dtype)
    return np.concatenate([ids[spans[i][0] : spans[i][1]] for i in document_indices])


# --- prepared data ----------------------------------------------------------


@dataclass
class PreparedData:
    """Token arrays for both splits plus everything needed to reason about them."""

    train_ids: np.ndarray
    val_ids: np.ndarray
    train_docs: list[int]
    val_docs: list[int]
    meta: dict
    path: str
    tokenizer: BPETokenizer
    corpus: Corpus

    @property
    def n_train_tokens(self) -> int:
        return int(len(self.train_ids))

    @property
    def n_val_tokens(self) -> int:
        return int(len(self.val_ids))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def prepare(
    config: DataConfig | None = None,
    tokenizer: BPETokenizer | None = None,
    force: bool = False,
) -> PreparedData:
    """Corpus -> (cached) token file -> train/val arrays.

    Encoding happens only when the token file is missing or ``force`` is set, so
    repeated runs start in milliseconds.
    """
    config = config or DataConfig()
    corpus = read_corpus(config.corpus_path, config.extensions, granularity=config.granularity)
    if not corpus:
        raise FileNotFoundError(f"No corpus files found in {config.corpus_path!r}")

    tokenizer = tokenizer or load_or_train(
        corpus.text,
        config.num_merges,
        config.cache_dir,
        show_progress=False,
        key=corpus.files_hash or corpus.fingerprint,
    )
    path = token_file_path(
        config.processed_dir, corpus.structure_fingerprint, tokenizer.num_merges
    )

    if force or not _token_file_exists(path):
        encoded = encode_corpus(tokenizer, corpus)
        meta = write_tokens(path, encoded, corpus=corpus, tokenizer=tokenizer)
        ids: np.ndarray = encoded.ids
    else:
        meta = load_meta(path)
        check_compatible(meta, tokenizer)
        ids = open_tokens(path)

    spans = [(int(start), int(end)) for start, end in meta["doc_spans"]]
    train_docs, val_docs = split_documents(len(spans), config.val_ratio, config.seed)

    return PreparedData(
        train_ids=gather(ids, spans, train_docs),
        val_ids=gather(ids, spans, val_docs),
        train_docs=train_docs,
        val_docs=val_docs,
        meta=meta,
        path=path,
        tokenizer=tokenizer,
        corpus=corpus,
    )


def _token_file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.exists(meta_path(path))


# --- loaders ----------------------------------------------------------------


def make_loader(
    ids: np.ndarray,
    context_length: int,
    batch_size: int,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 123,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GPTDataset(ids, context_length=context_length, stride=stride)
    if len(dataset) == 0:
        # torch refuses to build a DataLoader over zero samples; say why instead.
        raise ValueError(
            f"no samples: {len(ids)} token(s) cannot fill a window of {context_length} "
            f"inputs + 1 target"
        )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )


def build_loaders(
    config: DataConfig | None = None,
    context_length: int = 256,
    tokenizer: BPETokenizer | None = None,
    force: bool = False,
) -> tuple[DataLoader, DataLoader | None, PreparedData]:
    """Train loader, validation loader (``None`` if the split is empty) and the data."""
    config = config or DataConfig()
    prepared = prepare(config, tokenizer=tokenizer, force=force)
    stride = config.stride or context_length

    train_loader = make_loader(
        prepared.train_ids,
        context_length=context_length,
        batch_size=config.batch_size,
        stride=stride,
        shuffle=True,
        drop_last=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    val_loader = None
    if prepared.n_val_tokens > context_length:
        val_loader = make_loader(
            prepared.val_ids,
            context_length=context_length,
            batch_size=config.batch_size,
            stride=stride,
            shuffle=False,
            drop_last=False,
            seed=config.seed,
            num_workers=config.num_workers,
        )
    return train_loader, val_loader, prepared
