"""Stage 4 — sample text from a checkpoint (ROADMAP M5).

The signature already carries the whole decoding contract (D12: one `generate()`
with temperature / top-k / top-p / repetition penalty, greedy at
`temperature=0`), so the CLI and `pipeline` will not change when
`src/our_llm/generate.py` lands.
"""

from __future__ import annotations


def sample(
    ckpt: str,
    prompt: str = "The monster",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    num_samples: int = 1,
    seed: int | None = None,
) -> str:
    """Generate text from a checkpoint and return it.

    Args:
      ckpt: Checkpoint written by `train`.
      prompt: Text to condition on.
      max_new_tokens: How many tokens to append.
      temperature: 0 means greedy decoding.
      top_k: Keep only the k most likely tokens (None = no truncation).
      top_p: Nucleus sampling threshold (None = off).
      repetition_penalty: >1 shrinks already-used tokens' scores.
      num_samples: How many independent continuations.
      seed: Seed the sampler for reproducible output.
    """
    raise NotImplementedError(
        "sample: decoding lands in M5 (src/our_llm/generate.py). "
        f"Asked for {max_new_tokens} tokens from {ckpt!r}, prompt {prompt!r}, "
        f"temperature {temperature}, top_k {top_k}."
    )
