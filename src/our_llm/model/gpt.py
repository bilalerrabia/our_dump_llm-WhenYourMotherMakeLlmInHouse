"""The GPT model: token + positional embeddings -> N blocks -> norm -> logits.

A block is *pre-norm*: normalise, then mix (attention), then add the residual;
normalise, then transform (feed-forward), then add the residual. Adding the
input back is what lets gradients reach the early layers and lets the model
start as "mostly an identity function".

Two knobs from :class:`~our_llm.config.GPTConfig` change the parameter layout:

* ``pos_emb="learned"`` stores one vector per position (GPT-2). ``"sinusoidal"``
  stores nothing and computes the fixed *Attention Is All You Need* waves.
* ``tie_embeddings=True`` reuses the token-embedding matrix as the output head
  (``out_head.weight is tok_emb.weight``) -- that is 5.8M of the 13M params at
  the ``tiny`` preset, and it is why 13M is reachable here.

``count_parameters(model)`` must equal ``cfg.num_parameters()`` for every
combination; ``tests/test_model.py`` pins that.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from our_llm.config import GPTConfig
from our_llm.model.attention import MultiHeadAttention
from our_llm.model.layers import FeedForward, LayerNorm


def sinusoidal_positional_encoding(
    context_length: int, emb_dim: int, base: float = 10_000.0
) -> torch.Tensor:
    """Fixed ``(context_length, emb_dim)`` encoding (``Attention Is All You Need``, eq. 3.5).

    Even dimensions get ``sin(pos / base^(2i/d))``, odd ones ``cos`` of the same
    angle. Deterministic, so it costs zero parameters and zero checkpoint bytes.
    """
    position = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, emb_dim, 2, dtype=torch.float32) * (-math.log(base) / emb_dim)
    )
    pe = torch.zeros(context_length, emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward, each wrapped in a residual connection."""

    def __init__(self, cfg: GPTConfig, use_sdpa: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            use_sdpa=use_sdpa,
        )
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """Decoder-only transformer that turns token ids into next-token logits."""

    def __init__(self, cfg: GPTConfig, use_sdpa: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_sdpa = use_sdpa

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        if cfg.pos_emb == "learned":
            self.pos_emb: nn.Embedding | torch.Tensor = nn.Embedding(
                cfg.context_length, cfg.emb_dim
            )
        else:  # "sinusoidal": a buffer, not a parameter
            self.register_buffer(
                "pos_emb",
                sinusoidal_positional_encoding(cfg.context_length, cfg.emb_dim),
                persistent=False,
            )
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.ModuleList(
            TransformerBlock(cfg, use_sdpa=use_sdpa) for _ in range(cfg.n_layers)
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()
        if cfg.tie_embeddings:
            self.out_head.weight = self.tok_emb.weight

    # -- init -------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """GPT-2 trick: shrink the projections that feed a residual by 1/sqrt(2 N)."""
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.trf_blocks:
            nn.init.normal_(block.att.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ff.layers[2].weight, mean=0.0, std=std)

    # -- forward ----------------------------------------------------------
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` token ids -> ``(B, T, vocab_size)`` logits."""
        if in_idx.ndim != 2:
            raise ValueError(f"expected (batch, seq) token ids, got shape {tuple(in_idx.shape)}")
        seq_len = in_idx.shape[1]
        if seq_len > self.cfg.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length {self.cfg.context_length}"
            )

        x = self.tok_emb(in_idx)
        if isinstance(self.pos_emb, nn.Embedding):
            positions = torch.arange(seq_len, device=in_idx.device)
            x = x + self.pos_emb(positions)
        else:
            x = x + self.pos_emb[:seq_len]
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x)

        return self.out_head(self.final_norm(x))

    def num_parameters(self) -> int:
        """Live parameter count (compare against ``cfg.num_parameters()``)."""
        return sum(p.numel() for p in self.parameters())
