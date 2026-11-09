"""Byte Pair Encoding tokenizer.

Two differences from the first implementation in ``src/tokenizer.py``:

1. **Fast encode.** Instead of replaying every merge over the whole corpus
   (O(num_merges x words)), each pre-token is merged by repeatedly collapsing
   the pair with the lowest merge rank, and the result is memoised per word.
   That is O(len(word)**2) once per *unique* word.
2. **Self-describing, versioned cache.** The saved JSON carries a ``_meta``
   header, and legacy files (merges starting at 256, before special tokens
   existed) are shifted to the current layout on load.

Merging the lowest-rank pair iteratively is equivalent to applying the merge
rules in rank order; ``tests/test_tokenizer_v2.py`` asserts the two agree.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence

import regex as re
from tqdm import tqdm

from our_llm.tokenizer.special import (
    BASE_VOCAB_SIZE,
    ID_TO_SPECIAL,
    MERGE_START,
    SPECIAL_IDS,
    SPECIAL_TOKENS,
)

#: GPT-2 pre-tokenizer: contractions, words, numbers, punctuation, whitespace.
PRE_TOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

CACHE_VERSION = 1


def corpus_hash(text: str) -> str:
    """Short, stable fingerprint of a corpus (used to key the tokenizer cache)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tokenizer_cache_path(cache_dir: str, text_hash: str, num_merges: int) -> str:
    """Cache filename derived from the corpus fingerprint *and* the merge count."""
    return os.path.join(cache_dir, f"vocab-{text_hash}-m{num_merges}.json")


class BPETokenizer:
    """Byte-level BPE tokenizer with a fast encoder."""

    def __init__(self, merges: dict[int, tuple[int, int]] | None = None) -> None:
        self.pat = PRE_TOKENIZE_PATTERN
        self.merges: dict[int, tuple[int, int]] = dict(merges or {})
        self.special_ids: dict[str, int] = dict(SPECIAL_IDS)
        self._ranks: dict[tuple[int, int], int] = {
            pair: new_id for new_id, pair in self.merges.items()
        }
        self._bytes_cache: dict[int, bytes] = {}
        self._word_cache: dict[tuple[int, ...], tuple[int, ...]] = {}

    # --- introspection -------------------------------------------------------

    @property
    def num_merges(self) -> int:
        return len(self.merges)

    @property
    def vocab_size(self) -> int:
        return MERGE_START + len(self.merges)

    def __repr__(self) -> str:
        return f"BPETokenizer(num_merges={self.num_merges}, vocab_size={self.vocab_size})"

    def to_readable_vocab(self) -> dict[str, int]:
        """``{token_string: id}`` for every id, for humans and for debugging."""
        vocab: dict[str, int] = {}
        for i in range(BASE_VOCAB_SIZE):
            try:
                vocab[bytes([i]).decode("utf-8")] = i
            except UnicodeDecodeError:
                vocab[f"<0x{i:02x}>"] = i
        vocab.update(SPECIAL_IDS)
        for new_id in sorted(self.merges):
            raw = self._expand(new_id)
            try:
                vocab[raw.decode("utf-8")] = new_id
            except UnicodeDecodeError:
                vocab[f"<0x{raw.hex()}>"] = new_id
        return vocab

    # --- pre-tokenization ----------------------------------------------------

    def encoder(self, text: str) -> list[list[int]]:
        """Split ``text`` into pre-tokens, each a list of raw byte ids."""
        return [list(match.encode("utf-8")) for match in self.pat.findall(text)]

    # --- encoding ------------------------------------------------------------

    def encode(self, text: str, allowed_special: Iterable[str] = ()) -> list[int]:
        """Encode ``text`` to token ids.

        Special tokens are only recognised when named in ``allowed_special``
        (mirrors ``tiktoken``'s safe behaviour: user text can never smuggle in
        a control token by accident).
        """
        ids: list[int] = []
        for chunk, is_special in self._split_on_specials(text, allowed_special):
            if is_special:
                ids.append(self.special_ids[chunk])
                continue
            for match in self.pat.findall(chunk):
                ids.extend(self._encode_word(tuple(match.encode("utf-8"))))
        return ids

    def _split_on_specials(
        self, text: str, allowed_special: Iterable[str]
    ) -> list[tuple[str, bool]]:
        allowed = sorted(
            (t for t in allowed_special if t in self.special_ids),
            key=len,
            reverse=True,
        )
        if not text or not allowed:
            return [(text, False)]

        pattern = re.compile("(" + "|".join(re.escape(t) for t in allowed) + ")")
        allowed_set = set(allowed)
        return [(part, part in allowed_set) for part in pattern.split(text) if part]

    def _encode_word(self, word: tuple[int, ...]) -> tuple[int, ...]:
        """Merge lowest-rank pairs until no pair has a rank left."""
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        parts = list(word)
        while len(parts) > 1:
            best_id: int | None = None
            best_index = -1
            for i in range(len(parts) - 1):
                new_id = self._ranks.get((parts[i], parts[i + 1]))
                if new_id is not None and (best_id is None or new_id < best_id):
                    best_id, best_index = new_id, i
            if best_index < 0:
                break
            parts[best_index : best_index + 2] = [best_id]  # type: ignore[list-item]

        result = tuple(parts)
        self._word_cache[word] = result
        return result

    # --- decoding ------------------------------------------------------------

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Decode token ids back to text."""
        out: list[str] = []
        buffer: list[int] = []

        def flush() -> None:
            if buffer:
                out.append(bytes(buffer).decode("utf-8", errors="replace"))
                buffer.clear()

        for token_id in ids:
            special = ID_TO_SPECIAL.get(token_id)
            if special is not None:
                flush()
                if not skip_special:
                    out.append(special)
            else:
                buffer.extend(self._expand(token_id))
        flush()
        return "".join(out)

    def _expand(self, token_id: int) -> bytes:
        """Bytes behind ``token_id`` (iterative would be faster; cached anyway)."""
        cached = self._bytes_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id < BASE_VOCAB_SIZE:
            raw = bytes([token_id])
        else:
            pair = self.merges.get(token_id)
            if pair is None:
                raise ValueError(f"Missing merge for token {token_id}")
            raw = self._expand(pair[0]) + self._expand(pair[1])
        self._bytes_cache[token_id] = raw
        return raw

    # --- training ------------------------------------------------------------

    def train(
        self, text: str, num_merges: int, show_progress: bool = True
    ) -> BPETokenizer:
        """Learn ``num_merges`` merges on ``text`` (returns ``self``).

        Pair statistics are computed over *unique* words weighted by frequency,
        which is ~10x faster than counting every occurrence.
        """
        self.merges.clear()
        self._ranks.clear()
        self._bytes_cache.clear()
        self._word_cache.clear()

        word_counts = Counter(tuple(word) for word in self.encoder(text))
        words = [list(word) for word in word_counts]
        counts = list(word_counts.values())

        next_id = MERGE_START
        steps: Iterable[int] = range(num_merges)
        if show_progress:
            steps = tqdm(steps, desc=f"training BPE ({len(words)} unique words)")

        for _ in steps:
            pair_counts: dict[tuple[int, int], int] = {}
            for word, count in zip(words, counts):
                for pair in zip(word, word[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
            if not pair_counts:
                break

            # Most frequent pair; ties broken by the larger pair id (and so on,
            # deterministically) -- same rule as the first implementation.
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            words = self.replace(words, best_pair, next_id)
            self.merges[next_id] = best_pair
            self._ranks[best_pair] = next_id
            next_id += 1

        self._word_cache.clear()
        return self

    @staticmethod
    def replace(
        words: list[list[int]], max_pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """Replace every non-overlapping ``max_pair`` (left to right) with ``new_id``."""
        replaced: list[list[int]] = []
        for word in words:
            out: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            replaced.append(out)
        return replaced

    # --- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Write merges plus a ``_meta`` header (single self-describing file)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "_meta": {
                "cache_version": CACHE_VERSION,
                "base_vocab_size": BASE_VOCAB_SIZE,
                "merge_start": MERGE_START,
                "special_tokens": list(SPECIAL_TOKENS),
                "num_merges": len(self.merges),
                "vocab_size": self.vocab_size,
            },
            "merges": {str(k): list(v) for k, v in sorted(self.merges.items())},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load a cache, migrating legacy files that start at id 256."""
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        raw = data.get("merges", data) if isinstance(data, dict) else {}
        return cls(cls._normalise_merges(raw))

    @staticmethod
    def _normalise_merges(raw: dict) -> dict[int, tuple[int, int]]:
        merges = {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
        if not merges:
            return {}
        offset = MERGE_START - min(merges)
        if offset <= 0:
            return merges
        # Legacy cache written before special tokens existed: shift merges and
        # any pair references to them, leaving raw bytes (0-255) untouched.
        return {
            new_id + offset: (
                left + offset if left >= BASE_VOCAB_SIZE else left,
                right + offset if right >= BASE_VOCAB_SIZE else right,
            )
            for new_id, (left, right) in merges.items()
        }


def load_or_train(
    text: str,
    num_merges: int,
    cache_dir: str = "data/cache",
    show_progress: bool = True,
    key: str | None = None,
) -> BPETokenizer:
    """Load the cache for this ``(corpus, num_merges)`` or train it.

    ``key`` overrides the cache key (default: a hash of ``text``) so callers can
    make the cache depend on the raw corpus rather than on how the text was
    assembled into documents.
    """
    path = tokenizer_cache_path(cache_dir, key or corpus_hash(text), num_merges)
    if os.path.exists(path):
        tokenizer = BPETokenizer.load(path)
        if tokenizer.num_merges == num_merges:
            return tokenizer
    tokenizer = BPETokenizer().train(text, num_merges, show_progress=show_progress)
    tokenizer.save(path)
    return tokenizer
