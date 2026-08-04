import regex as re
import json
from tqdm import tqdm

class Tokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer implementation.
    """
    def __init__(self):
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.map_replace = {}  # Dictionary to store merge mappings

    def save_vocab(self, filepath: str):
        """Save a readable vocab.json and an internal merges.json."""
        
        # Helper to recursively get the full byte sequence for a merged token
        def get_bytes(token_id: int) -> bytes:
            if token_id < 256:
                return bytes([token_id])
            pair = self.map_replace[token_id]
            return get_bytes(pair[0]) + get_bytes(pair[1])

        # 1. Build the human-readable vocabulary { "string": id }
        vocab = {}
        # Add base bytes (0-255) to the vocab for completeness
        for i in range(256):
            try:
                char = bytes([i]).decode("utf-8")
            except UnicodeDecodeError:
                char = f"<0x{i:02x}>" # Fallback for non-printable bytes
            vocab[char] = i

        # Add merged tokens
        for new_id in sorted(self.map_replace.keys()):
            b = get_bytes(new_id)
            try:
                char = b.decode("utf-8")
            except UnicodeDecodeError:
                char = f"<0x{b.hex()}>" # Fallback for incomplete multi-byte sequences
            vocab[char] = new_id

        # Save the readable vocab file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        print(f"Readable vocabulary cached to {filepath}")

        # 2. Save the internal merges file { "id": [pair1, pair2] }
        merges_filepath = filepath.replace('.json', '_merges.json')
        serializable_merges = {str(k): list(v) for k, v in self.map_replace.items()}
        with open(merges_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_merges, f)

    def load_vocab(self, filepath: str):
        """Load merges from the internal merges.json."""
        merges_filepath = filepath.replace('.json', '_merges.json')
        try:
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.map_replace = {int(k): tuple(v) for k, v in data.items()}
            print(f"Vocabulary loaded from cache.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {merges_filepath}. Please train the tokenizer first.")

    def encoder(self, text: str):
        """Pre-tokenize text into a list of words, each represented as a list of bytes."""
        words = []
        for match in self.pat.findall(text):
            words.append(list(match.encode("utf-8")))
        return words

    def encode(self, text: str):
        """Encode new text into token IDs by applying learned merges."""
        words = self.encoder(text)
        
        # Apply merges in the order they were learned
        for new_id, pair in tqdm(sorted(self.map_replace.items()), desc="encoding..."):
            words = self.replace(words, pair, new_id)
            
        # Flatten the list of words back into a single list of token IDs
        return [token for word in words for token in word]

    def decoder(self, tokens_ids: list[int]):
        """Decode tokens back to text by expanding merges."""
        expanded_bytes = []

        for token in tokens_ids:
            stack = [token]
            
            while stack:
                current = stack.pop()
                if current > 255:
                    pair = self.map_replace.get(current)
                    if not pair:
                        raise ValueError(f"Missing merge for token {current}")
                    stack.append(pair[1])
                    stack.append(pair[0])
                else:
                    expanded_bytes.append(current)

        return bytes(expanded_bytes).decode("utf-8", errors="replace")

    def pairs_counter(self, words: list[list[int]]):
        """Count frequencies of adjacent token pairs within each word."""
        counts = {}
        for word in words:
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def replace(self, words: list[list[int]], max_pair: tuple, replace: int):
        """Replace occurrences of max_pair with the new token within each word."""
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                    new_word.append(replace)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, text: str, num_merges: int):
        """Train the tokenizer by performing merges on the text."""
        words = self.encoder(text)
        replace_car = 256
        target = replace_car + num_merges

        for i in tqdm(range(replace_car, target), desc="Tokenizer Training"):
            tokens_pairs = self.pairs_counter(words)
            if not tokens_pairs:
                break
            
            max_pair = max((v, k) for k, v in tokens_pairs.items())
            words = self.replace(words, max_pair[1], replace_car)
            self.map_replace[replace_car] = max_pair[1]
            replace_car += 1

        return [token for word in words for token in word]