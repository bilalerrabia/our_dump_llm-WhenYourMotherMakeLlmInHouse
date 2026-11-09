"""On-disk token files: encode the corpus **once**, then memmap it.

Training startup should not depend on how big the corpus is, and a run must be
reproducible from what is on disk. So `our-llm tokenize` produces:

    data/processed/tokens-<corpus_hash>-m<merges>.bin      uint16/uint32, no header
    data/processed/tokens-<corpus_hash>-m<merges>.meta.json

The binary is a raw little-endian array (no header) so `np.memmap` can open it
directly; everything else lives in the JSON sidecar: dtype, token count, the
document spans used for a leakage-free train/val split, and the fingerprints of
both the corpus and the vocabulary (so a stale file is detectable, not silent).

Documents are encoded **individually** and joined with `<|endoftext|>`. That
costs one extra token per document and buys three things: exact document spans,
a train/val split that can respect document boundaries, and the ability to
decode any span back to the original text.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from our_llm.data.corpus import Corpus
from our_llm.tokenizer.bpe import BPETokenizer
from our_llm.tokenizer.special import EOT_TOKEN

TOKEN_FILE_VERSION = 1

#: 65,536 distinct ids fit in uint16; the vocab is ~15k, so this is just a guard.
UINT16_LIMIT = 2**16

TOKEN_SUFFIX = ".bin"
META_SUFFIX = ".meta.json"


def sha16(text: str) -> str:
    """Short stable fingerprint of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tokens_dtype(vocab_size: int) -> np.dtype:
    """uint16 while the vocabulary fits, uint32 beyond that (ROADMAP D5 fallback)."""
    return np.dtype(np.uint16 if vocab_size <= UINT16_LIMIT else np.uint32)


def vocab_fingerprint(tokenizer: BPETokenizer) -> str:
    """Stable hash of the merge rules: two tokenizers are interchangeable iff equal."""
    payload = json.dumps(
        [[new_id, list(pair)] for new_id, pair in sorted(tokenizer.merges.items())],
        separators=(",", ":"),
    )
    return sha16(payload)


def token_file_path(directory: str, structure_hash: str, num_merges: int) -> str:
    """`<dir>/tokens-<structure>-m<merges>.bin`.

    Keyed by the *document structure* hash (not just the text) because the same
    characters split into different documents produce a different token array,
    and by the merge count because a different vocabulary gives a different id
    space.
    """
    return os.path.join(directory, f"tokens-{structure_hash}-m{num_merges}{TOKEN_SUFFIX}")


def meta_path(bin_path: str) -> str:
    return bin_path[: -len(TOKEN_SUFFIX)] + META_SUFFIX


@dataclass
class EncodedCorpus:
    """Token ids for a whole corpus plus where each document lives."""

    ids: np.ndarray
    doc_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])

    def document(self, index: int) -> np.ndarray:
        start, end = self.doc_spans[index]
        return self.ids[start:end]

    def decode_document(
        self, tokenizer: BPETokenizer, index: int, skip_special: bool = True
    ) -> str:
        return tokenizer.decode(self.document(index).tolist(), skip_special=skip_special)


def encode_corpus(
    tokenizer: BPETokenizer, corpus: Corpus, add_eot: bool = True
) -> EncodedCorpus:
    """Encode every document separately and record its span.

    The trailing `<|endoftext|>` per document is what the model learns as "a new
    document starts here"; it also makes the spans unambiguous.
    """
    pieces: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    dtype = tokens_dtype(tokenizer.vocab_size)
    specials = tokenizer.special_ids if add_eot else ()

    for doc in corpus.docs:
        ids = tokenizer.encode(doc, allowed_special=specials)
        if add_eot:
            ids.append(tokenizer.special_ids[EOT_TOKEN])
        chunk = np.asarray(ids, dtype=dtype)
        pieces.append(chunk)
        spans.append((cursor, cursor + chunk.shape[0]))
        cursor += chunk.shape[0]

    ids = np.concatenate(pieces) if pieces else np.empty(0, dtype=dtype)
    return EncodedCorpus(ids=ids, doc_spans=spans)


def write_tokens(
    path: str, encoded: EncodedCorpus, *, corpus: Corpus, tokenizer: BPETokenizer
) -> dict:
    """Write the raw array plus its sidecar; returns the metadata written."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    encoded.ids.tofile(path)

    # One entry per document would repeat a filename 857 times; store the unique
    # file list plus a per-document index into it.
    sources = sorted(set(corpus.sources))
    source_index = {source: i for i, source in enumerate(sources)}

    meta = {
        "version": TOKEN_FILE_VERSION,
        "corpus_hash": corpus.fingerprint,
        "structure_hash": corpus.structure_fingerprint,
        "vocab_fingerprint": vocab_fingerprint(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "num_merges": tokenizer.num_merges,
        "dtype": str(encoded.ids.dtype),
        "n_tokens": encoded.n_tokens,
        "n_documents": len(encoded.doc_spans),
        "doc_spans": [[start, end] for start, end in encoded.doc_spans],
        "sources": sources,
        "doc_source": [source_index[source] for source in corpus.sources],
        "eot_id": tokenizer.special_ids[EOT_TOKEN],
    }
    with open(meta_path(path), "w", encoding="utf-8") as handle:
        _dump_meta(meta, handle)
    return meta


def _dump_meta(meta: dict, handle) -> None:
    """One key per line, but the long arrays kept on single lines.

    Plain `json.dump(indent=1)` turns 857 spans into ~3,400 lines of diff noise;
    this stays readable, valid JSON and reviewable.
    """
    compact = {"doc_spans", "doc_source"}
    head = {key: value for key, value in meta.items() if key not in compact}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: {json.dumps(value)},")
    spans = ", ".join(f"[{int(start)},{int(end)}]" for start, end in meta["doc_spans"])
    lines.append(f' "doc_spans": [{spans}],')
    lines.append(f' "doc_source": [{", ".join(str(int(i)) for i in meta["doc_source"])}]')
    lines.append("}")
    handle.write("\n".join(lines) + "\n")


def load_meta(path: str) -> dict:
    """Read the JSON sidecar belonging to a token file."""
    with open(meta_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def open_tokens(path: str, mmap: bool = True) -> np.ndarray:
    """Memmap (default) or read a token file; dtype and length come from the sidecar.

    Mapping means a 96k-token file and a 96M-token file cost the same to open.
    """
    meta = load_meta(path)
    dtype = np.dtype(meta["dtype"])
    shape = (int(meta["n_tokens"]),)
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return np.fromfile(path, dtype=dtype).reshape(shape)


def doc_spans(meta: dict) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in meta["doc_spans"]]


def check_compatible(meta: dict, tokenizer: BPETokenizer) -> None:
    """Fail loudly if the file was produced by a different vocabulary."""
    if meta.get("version") != TOKEN_FILE_VERSION:
        raise ValueError(
            f"token file version {meta.get('version')} != {TOKEN_FILE_VERSION}; "
            "re-run `our-llm tokenize --force`"
        )
    expected = vocab_fingerprint(tokenizer)
    if meta["vocab_fingerprint"] != expected:
        raise ValueError(
            "token file was produced with a different vocabulary "
            f"({meta['vocab_fingerprint']} != {expected}); "
            "re-run `our-llm tokenize --force`"
        )
    if int(meta["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError(
            f"token file vocab_size {meta['vocab_size']} != tokenizer {tokenizer.vocab_size}"
        )
