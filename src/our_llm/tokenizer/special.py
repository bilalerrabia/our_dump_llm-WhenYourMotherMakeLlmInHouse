"""Special tokens and the fixed vocabulary layout.

The id layout is **frozen**. Never reorder it: ids must stay stable across
retrainings, otherwise every checkpoint's ``out_head`` becomes meaningless.

    0   .. 255      raw bytes                 (BASE_VOCAB_SIZE)
    256             <|endoftext|>             document separator
    257             <|pad|>                   dataloader padding
    258 .. N        learned BPE merges        (MERGE_START)

This is the same layout GPT-2 uses (256 = <|endoftext|>), just with one extra
special token, so ``vocab_size = 258 + len(merges)``.
"""

BASE_VOCAB_SIZE = 256

EOT_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

#: Order defines the ids. Append only.
SPECIAL_TOKENS: tuple[str, ...] = (EOT_TOKEN, PAD_TOKEN)

SPECIAL_IDS: dict[str, int] = {
    token: BASE_VOCAB_SIZE + i for i, token in enumerate(SPECIAL_TOKENS)
}
ID_TO_SPECIAL: dict[int, str] = {i: token for token, i in SPECIAL_IDS.items()}

#: First id available for a learned merge.
MERGE_START = BASE_VOCAB_SIZE + len(SPECIAL_TOKENS)


def vocab_size(num_merges: int) -> int:
    """Total vocabulary size for a tokenizer that learned ``num_merges`` merges."""
    return MERGE_START + num_merges
