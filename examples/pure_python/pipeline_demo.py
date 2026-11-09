"""The original pure-Python pipeline: read -> BPE -> batches -> embeddings.

This is the *teaching* version that was `main.py` before the `our_llm` package
existed. It is deliberately dependency-free (no torch, no tensors) so every step
is visible; the production path is `uv run our-llm tokenize`.

Kept as an example because it is the clearest statement of the shape of the
problem: tokens in, (inputs, targets) out, embeddings + positions added.

Run it with:
    uv run python examples/pure_python/pipeline_demo.py

WARNING: this uses the OLD tokenizer in ``src/tokenizer.py``, which replays every
merge over the whole corpus (minutes on 440 KB). `our_llm.tokenizer` is ~1600x
faster; that file is kept only as the reference implementation.
"""

import os
import random

from src.data_loader import DatasetPreparation, create_dataloader
from src.embedding import (
    Embedding,
    add_embeddings,
    get_shape,
    sinusoidal_positional_encoding,
)
from src.tokenizer import Tokenizer

DATA_DIR = "data"
VOCAB_CACHE_FILE = "tokenizer_vocab.json"
MERGE_RATIO = 10  # num_merges = len(data) // MERGE_RATIO (10% of the text)
CONTEXT_WINDOW = 4
STRIDE = 4
BATCH_SIZE = 8
OUTPUT_DIM = 256
RANDOM_SEED = 123

DATA_EXTENSIONS = (".txt", ".md")


def find_data_files(path: str) -> list[str]:
    """Return the text files to read, sorted so the corpus is reproducible."""
    if not os.path.isdir(path):
        return [path]
    return sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(path)
        for name in names
        if os.path.splitext(name)[1].lower() in DATA_EXTENSIONS
    )


def read_data(path: str) -> str:
    """Read and concatenate the training text from a file or a directory."""
    files = find_data_files(path)
    if not files:
        print(f"Error: no training files found in {path}")
        return ""

    chunks: list[str] = []
    for file in files:
        try:
            with open(file, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            print(f"Error reading file {file}: {exc}")
            continue
        print(f"Loaded {file}: {len(text)} characters")
        chunks.append(text)

    return "\n\n".join(chunks)


def build_tokenizer(text: str, num_merges: int, vocab_cache_file: str) -> Tokenizer:
    """Load the tokenizer from cache, or train it and cache the result."""
    tokenizer = Tokenizer()
    if os.path.exists(vocab_cache_file):
        tokenizer.load_vocab(vocab_cache_file)
    else:
        print(f"No cache found. Training tokenizer with {num_merges} merges...")
        tokenizer.train(text, num_merges=num_merges)
        tokenizer.save_vocab(vocab_cache_file)
    return tokenizer


def make_dataloader(token_ids: list[int]):
    """Create input/target batches over sliding windows of token ids."""
    dataset = DatasetPreparation(token_ids, context_window=CONTEXT_WINDOW, stride=STRIDE)
    return create_dataloader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )


def run_embeddings(
    inputs: list[list[int]], vocab_size: int, context_window: int, output_dim: int
) -> list:
    """Add token + positional embeddings for one batch (pure Python)."""
    random.seed(RANDOM_SEED)  # seed for reproducible random weights

    token_embedding_layer = Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print("Token embeddings shape:", get_shape(token_embeddings))

    # Fixed sinusoidal positional encoding (Attention Is All You Need, eq. 3.5)
    pos_embeddings = sinusoidal_positional_encoding(context_window, output_dim)
    print("Positional embeddings shape:", get_shape(pos_embeddings))

    return add_embeddings(token_embeddings, pos_embeddings)


def main() -> None:
    # 1. Read the data (all text files inside DATA_DIR)
    data_str = read_data(DATA_DIR)
    if not data_str:
        return
    num_merges = len(data_str) // MERGE_RATIO
    print(f"Original text length: {len(data_str)} characters")

    # 2. Initialize Tokenizer
    tokenizer = build_tokenizer(data_str, num_merges, VOCAB_CACHE_FILE)

    # Vocabulary size: 256 base bytes + number of learned merges
    vocab_size = 256 + len(tokenizer.map_replace)
    print(f"Vocabulary size: {vocab_size}")

    # 3. Encode the text
    data_tokens = tokenizer.encode(data_str)
    print(f"Total tokens after encoding: {len(data_tokens)}")

    # 4. Create Dataset and DataLoader
    dataloader = make_dataloader(data_tokens)

    # 5. Fetch the first batch
    print("\n--- Fetching First Batch ---")
    inputs, targets = next(iter(dataloader))
    print("Inputs (Token IDs):")
    for seq in inputs:
        print(seq)

    # 6. Embeddings (Pure Python)
    print("\n--- Running Embeddings ---")
    final_embeddings = run_embeddings(inputs, vocab_size, CONTEXT_WINDOW, OUTPUT_DIM)
    print("Final input embeddings shape:", get_shape(final_embeddings))


if __name__ == "__main__":
    main()
