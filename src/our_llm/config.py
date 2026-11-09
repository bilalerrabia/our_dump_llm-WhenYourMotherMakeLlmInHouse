"""Model / data configuration and the size presets.

Everything tunable lives here as a frozen dataclass so the CLI is a thin parser
over it and a checkpoint can record exactly what it was trained with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from our_llm.tokenizer.special import MERGE_START

#: Merges learned by the tokenizer cache currently in the repo.
DEFAULT_NUM_MERGES = 14_970

POS_EMB_CHOICES = ("learned", "sinusoidal")


@dataclass(frozen=True, slots=True)
class GPTConfig:
    """A GPT-2 style decoder-only transformer."""

    vocab_size: int = MERGE_START + DEFAULT_NUM_MERGES  # 15,228 -> from the tokenizer
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 4
    drop_rate: float = 0.1
    qkv_bias: bool = False
    pos_emb: str = "learned"  # "learned" | "sinusoidal"
    tie_embeddings: bool = True  # share tok_emb and out_head weights

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.pos_emb not in POS_EMB_CHOICES:
            raise ValueError(f"pos_emb must be one of {POS_EMB_CHOICES}")

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

    def num_parameters(self) -> int:
        """Parameter count straight from the architecture (no torch needed).

        Token + positional embeddings, per block: QKV + out_proj + FFN + two
        LayerNorms, then the final LayerNorm and (unless tied) the output head.
        Sinusoidal positions are a fixed buffer, so they cost nothing.
        """
        dim = self.emb_dim
        tok = self.vocab_size * dim
        pos = 0 if self.pos_emb == "sinusoidal" else self.context_length * dim

        qkv_bias = dim if self.qkv_bias else 0
        attention = 3 * (dim * dim + qkv_bias) + (dim * dim + dim)
        feed_forward = (dim * 4 * dim + 4 * dim) + (4 * dim * dim + dim)
        norms = 4 * dim  # 2 per block, each scale + shift
        block = attention + feed_forward + norms

        head = 0 if self.tie_embeddings else self.vocab_size * dim
        return tok + pos + self.n_layers * block + 2 * dim + head

    def with_vocab(self, vocab_size: int) -> GPTConfig:
        """Copy with the vocabulary size taken from a trained tokenizer."""
        return replace(self, vocab_size=vocab_size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> GPTConfig:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in known})


#: `debug` overfits one batch in seconds; `tiny` is the "13.37 M" target;
#: `gpt2-small` is the real GPT-2 124M shape (needs tiktoken vocab + a GPU).
PRESETS: dict[str, GPTConfig] = {
    "debug": GPTConfig(
        vocab_size=MERGE_START + DEFAULT_NUM_MERGES,
        context_length=64,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        drop_rate=0.0,
        tie_embeddings=True,
    ),
    "tiny": GPTConfig(
        vocab_size=MERGE_START + DEFAULT_NUM_MERGES,
        context_length=256,
        emb_dim=384,
        n_heads=6,
        n_layers=4,
        drop_rate=0.1,
        tie_embeddings=True,
    ),
    "gpt2-small": GPTConfig(
        vocab_size=50_257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        tie_embeddings=True,
    ),
}


def get_preset(name: str) -> GPTConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown preset {name!r}. Known: {sorted(PRESETS)}") from None


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Corpus + batching settings."""

    corpus_path: str = "data"
    extensions: tuple[str, ...] = (".txt", ".md")
    granularity: str = "paragraph"  # "file" | "paragraph": the split unit
    cache_dir: str = "data/cache"
    processed_dir: str = "data/processed"
    num_merges: int = DEFAULT_NUM_MERGES
    val_ratio: float = 0.10
    batch_size: int = 8
    stride: int | None = None  # None -> stride = context_length
    num_workers: int = 0
    seed: int = 123


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation settings (used from M4 on)."""

    preset: str = "tiny"
    num_merges: int = DEFAULT_NUM_MERGES
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    max_epochs: int = 10
    eval_freq: int = 5
    eval_iter: int = 5
    seed: int = 123
    run_dir: str = "runs"
