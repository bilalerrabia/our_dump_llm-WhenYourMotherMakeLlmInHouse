"""Reading the training corpus."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

#: Only these extensions are read. Without a filter, stray files (e.g. the
#: extensionless `data/gg` duplicate that used to double the corpus) get
#: swallowed into training.
DATA_EXTENSIONS: tuple[str, ...] = (".txt", ".md")

#: What counts as one document (= one unit of the train/val split).
GRANULARITIES: tuple[str, ...] = ("file", "paragraph")

#: A blank line (with optional spaces/tabs) separates paragraphs.
PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n")


@dataclass
class Corpus:
    """Training text split into documents (documents are natural split boundaries)."""

    docs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    separator: str = "\n\n"
    #: Fingerprint of the raw file contents, independent of ``granularity``.
    #: The BPE vocabulary depends on the characters only, so its cache must not
    #: be invalidated just because we started splitting into paragraphs.
    files_hash: str = ""

    @property
    def text(self) -> str:
        return self.separator.join(self.docs)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Short hash of the *text* (what the vocabulary depends on)."""
        return self.sha256[:16]

    @property
    def structure_fingerprint(self) -> str:
        """Short hash of the *document boundaries* (what the token file depends on).

        Content alone is not enough: the same text split into 1 file or 1,400
        paragraphs yields the same characters but a different token array (one
        `<|endoftext|>` per document) and a different split.
        """
        digest = hashlib.sha256()
        for doc in self.docs:
            digest.update(hashlib.sha256(doc.encode("utf-8")).digest())
            digest.update(b"\x00")
        return digest.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.text)

    def __bool__(self) -> bool:
        return bool(self.docs)


def find_data_files(path: str, extensions: tuple[str, ...] = DATA_EXTENSIONS) -> list[str]:
    """Return the readable corpus files under ``path`` (sorted, extension-filtered)."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such corpus path: {path}")

    files = [
        os.path.join(root, name)
        for root, _, names in os.walk(path)
        for name in names
        if os.path.splitext(name)[1].lower() in extensions
    ]
    return sorted(files)


def sha16(text: str) -> str:
    """Short stable fingerprint of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_corpus(
    path: str,
    extensions: tuple[str, ...] = DATA_EXTENSIONS,
    separator: str = "\n\n",
    granularity: str = "file",
) -> Corpus:
    """Read every corpus file under ``path``.

    ``granularity`` decides what counts as a document (the unit of the train/val
    split):

    * ``"file"`` — one document per file.
    * ``"paragraph"`` — one document per blank-line-separated block. A single
      novel then still has ~1,400 documents to split, instead of one — the
      difference between having a validation set and not having one. Paragraphs
      also never cut a sentence in half, which is exactly what a token-level
      percentage split would do.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")

    corpus = Corpus(separator=separator)
    raw_texts: list[str] = []
    for file in find_data_files(path, extensions):
        try:
            with open(file, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:  # unreadable file: skip, don't fail the run
            print(f"Warning: cannot read {file}: {exc}")
            continue
        if not text.strip():
            continue
        raw_texts.append(text)
        docs = [text] if granularity == "file" else split_paragraphs(text)
        for doc in docs:
            corpus.docs.append(doc)
            corpus.sources.append(file)

    # What the vocabulary is trained on: the characters, not the doc boundaries.
    corpus.files_hash = sha16(separator.join(raw_texts))
    return corpus


def split_paragraphs(text: str, min_chars: int = 1) -> list[str]:
    """Split on blank lines, stripping blocks and dropping empty/too-short ones."""
    blocks = (block.strip() for block in PARAGRAPH_SEPARATOR.split(text))
    return [block for block in blocks if len(block) >= min_chars]
