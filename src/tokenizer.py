"""Byte Pair Encoding (BPE) tokenizer.

Trains merge rules on raw text, encodes text to token ids, and decodes
ids back to text. The learned merges live in ``self.map_replace``:
``{new_token_id: (left_id, right_id)}``.
"""

import json
import regex as re
from tqdm import tqdm

BASE_VOCAB_SIZE = 256  # ids 0-255 are the raw bytes


class Tokenizer:
    """A Byte Pair Encoding (BPE) tokenizer implementation."""

    def __init__(self):
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.map_replace: dict[int, tuple[int, int]] = {}  # new_id -> (left, right)

    # --- persistence ---------------------------------------------------------

    def save_vocab(self, filepath: str) -> None:
        """Save a readable vocab.json and an internal merges.json."""
        # 1. Build the human-readable vocabulary { "string": id }
        vocab = self._build_readable_vocab()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        print(f"Readable vocabulary cached to {filepath}")

        # 2. Save the internal merges file { "id": [pair1, pair2] }
        merges_filepath = self._merges_path(filepath)
        serializable_merges = {str(k): list(v) for k, v in self.map_replace.items()}
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f)

    def load_vocab(self, filepath: str) -> None:
        """Load merges from the internal merges.json."""
        merges_filepath = self._merges_path(filepath)
        try:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.map_replace = {int(k): tuple(v) for k, v in data.items()}
            print(f"Vocabulary loaded from cache.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {merges_filepath}. Please train the tokenizer first."
            )

    def _merges_path(self, filepath: str) -> str:
        """Derive the merges cache path from the vocab cache path."""
        return filepath.replace(".json", "_merges.json")

    def _token_bytes(self, token_id: int) -> bytes:
        """Recursively get the full byte sequence behind a merged token."""
        if token_id < BASE_VOCAB_SIZE:
            return bytes([token_id])
        left, right = self.map_replace[token_id]
        return self._token_bytes(left) + self._token_bytes(right)

    def _build_readable_vocab(self) -> dict[str, int]:
        """Build { readable string: token id } for base bytes and merges."""
        vocab: dict[str, int] = {}

        # Base bytes (0-255) for completeness
        for i in range(BASE_VOCAB_SIZE):
            try:
                char = bytes([i]).decode("utf-8")
            except UnicodeDecodeError:
                char = f"<0x{i:02x}>"  # fallback for non-printable bytes
            vocab[char] = i

        # Merged tokens
        for new_id in sorted(self.map_replace.keys()):
            b = self._token_bytes(new_id)
            try:
                char = b.decode("utf-8")
            except UnicodeDecodeError:
                char = f"<0x{b.hex()}>"  # fallback for incomplete multi-byte sequences
            vocab[char] = new_id

        return vocab

    # --- encoding / decoding --------------------------------------------------

    def encoder(self, text: str) -> list[list[int]]:
        """Pre-tokenize text into words, each a list of raw byte ids."""
        return [list(match.encode("utf-8")) for match in self.pat.findall(text)]

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids by applying the learned merges in order."""
        words = self.encoder(text)

        for new_id, pair in tqdm(sorted(self.map_replace.items()), desc="encoding..."):
            words = self.replace(words, pair, new_id)

        # Flatten the list of words back into a single list of token ids
        return [token for word in words for token in word]

    def decoder(self, tokens_ids: list[int]) -> str:
        """Decode token ids back to text by expanding the merges."""
        expanded_bytes: list[int] = []

        for token in tokens_ids:
            stack = [token]
            while stack:
                current = stack.pop()
                if current > BASE_VOCAB_SIZE - 1:
                    pair = self.map_replace.get(current)
                    if not pair:
                        raise ValueError(f"Missing merge for token {current}")
                    stack.append(pair[1])
                    stack.append(pair[0])
                else:
                    expanded_bytes.append(current)

        return bytes(expanded_bytes).decode("utf-8", errors="replace")

    # --- training -------------------------------------------------------------

    def pairs_counter(self, words: list[list[int]]) -> dict[tuple[int, int], int]:
        """Count frequencies of adjacent token pairs within each word."""
        counts: dict[tuple[int, int], int] = {}
        for word in words:
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def replace(
        self, words: list[list[int]], max_pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """Replace occurrences of max_pair with new_id within each word."""
        new_words: list[list[int]] = []
        for word in words:
            new_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, text: str, num_merges: int) -> list[int]:
        """Learn num_merges BPE merges on text and return the encoded ids."""
        words = self.encoder(text)
        next_id = BASE_VOCAB_SIZE
        target = next_id + num_merges

        for _ in tqdm(range(next_id, target), desc="Tokenizer Training"):
            tokens_pairs = self.pairs_counter(words)
            if not tokens_pairs:
                break

            max_pair = max((v, k) for k, v in tokens_pairs.items())
            words = self.replace(words, max_pair[1], next_id)
            self.map_replace[next_id] = max_pair[1]
            next_id += 1

        return [token for word in words for token in word]
