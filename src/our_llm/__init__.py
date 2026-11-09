"""our-llm — a GPT built from scratch, one stage at a time.

Public API (import light things eagerly; torch-backed pieces stay lazy so the
CLI and the tokenizer work without paying for the import):

    >>> from our_llm import GPTConfig, BPETokenizer, PRESETS
"""

from our_llm.config import PRESETS, DataConfig, GPTConfig, TrainConfig, get_preset
from our_llm.data.corpus import Corpus, read_corpus
from our_llm.tokenizer.bpe import BPETokenizer, load_or_train
from our_llm.tokenizer.special import MERGE_START, SPECIAL_TOKENS, vocab_size

__version__ = "0.2.0"

__all__ = [
    "BPETokenizer",
    "Corpus",
    "DataConfig",
    "GPTConfig",
    "MERGE_START",
    "PRESETS",
    "SPECIAL_TOKENS",
    "TrainConfig",
    "get_preset",
    "load_or_train",
    "read_corpus",
    "vocab_size",
]


def main() -> None:
    """Console-script entry point (`uv run our-llm ...`)."""
    from our_llm.app import main as cli_main

    cli_main()
