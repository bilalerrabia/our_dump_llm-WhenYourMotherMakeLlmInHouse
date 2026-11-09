"""Corpus reading, the on-disk token cache, and torch datasets.

Two tiers on purpose:

* **No torch** — `corpus`, `split`, `tokens`. These are all `our-llm tokenize`
  needs, and importing torch costs ~1.5 s.
* **Torch** — `dataset`, resolved lazily below via :pep:`562` `__getattr__`, so
  `import our_llm.data.corpus` (and the `tokenize` stage) never pays for it.
  `from our_llm.data import prepare` still works and still imports torch.
"""

from our_llm.data.corpus import DATA_EXTENSIONS, Corpus, find_data_files, read_corpus
from our_llm.data.split import split_documents
from our_llm.data.tokens import (
    EncodedCorpus,
    check_compatible,
    encode_corpus,
    load_meta,
    open_tokens,
    token_file_path,
    tokens_dtype,
    vocab_fingerprint,
    write_tokens,
)

#: Names that live in `our_llm.data.dataset` (which imports torch).
_LAZY = frozenset({"GPTDataset", "PreparedData", "build_loaders", "gather", "make_loader", "prepare"})

__all__ = [
    "DATA_EXTENSIONS",
    "Corpus",
    "EncodedCorpus",
    "GPTDataset",
    "PreparedData",
    "build_loaders",
    "check_compatible",
    "encode_corpus",
    "find_data_files",
    "gather",
    "load_meta",
    "make_loader",
    "open_tokens",
    "prepare",
    "read_corpus",
    "split_documents",
    "token_file_path",
    "tokens_dtype",
    "vocab_fingerprint",
    "write_tokens",
]


def __getattr__(name: str):
    if name in _LAZY:
        from our_llm.data import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
