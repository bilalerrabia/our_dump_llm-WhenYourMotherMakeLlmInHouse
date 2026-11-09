"""The whole pipeline in one call: train-tokenizer -> tokenize -> train -> eval -> sample.

Each stage is a separate module (:mod:`~our_llm.tasks.train_tokenizer`,
`tokenize`, `train`, `evaluate`, `sample`); this file only *orders* them and
passes the outputs along, so:

    uv run python main.py pipeline                        # train everything, in order
    uv run python main.py pipeline --stages tokenize      # just the data stage
    uv run python main.py pipeline --stages tokenize,train --preset debug
    uv run python main.py pipeline --retrain-vocab        # relearn the BPE merges first

Two training stages, in this order: `train-tokenizer` learns the BPE merges,
`train` pretrains the GPT on the tokens those merges produce. `tokenize` sits
between them and is a no-op when the cache is warm, so a full run is
`pipeline --stages all`; `--retrain-vocab` forces the merges to be learned again.

Three outcomes per stage: ``ok`` (returned), ``skipped`` (raises
``NotImplementedError`` — the stage's milestone has not landed yet) and
``failed`` (anything else). A skip is expected during development and does not
stop the run; a failure does, because every later stage would consume broken
input.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field

from our_llm.config import DEFAULT_NUM_MERGES
from our_llm.tasks.evaluate import evaluate
from our_llm.tasks.sample import sample
from our_llm.tasks.tokenize import TokenizeResult, tokenize
from our_llm.tasks.train import train
from our_llm.tasks.train_tokenizer import train_tokenizer

#: Execution order. Later stages consume what earlier stages return.
STAGES: tuple[str, ...] = ("train-tokenizer", "tokenize", "train", "eval", "sample")

#: Things `pipeline` knows how to hand from one stage to the next. Every name is
#: lower-cased before lookup, so `--stages Train_Tokenizer` works too.
_STAGE_ALIASES = {
    "train_tokenizer": "train-tokenizer",
    "tokenizer": "train-tokenizer",
    "vocab": "train-tokenizer",
    "bpe": "train-tokenizer",
    "train_llm": "train",
    "llm": "train",
    "evaluate": "eval",
    "sampling": "sample",
}


@dataclass(slots=True)
class StageOutcome:
    """One row of the run report."""

    name: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""
    value: object = None
    error: str | None = field(default=None)

    def __str__(self) -> str:
        return f"{self.name} {self.status}" + (f": {self.detail}" if self.detail else "")


@dataclass(slots=True)
class PipelineReport:
    """What `pipeline` returns: the per-stage outcomes, plus quick counts."""

    outcomes: list[StageOutcome] = field(default_factory=list)

    def _count(self, status: str) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == status)

    @property
    def ok(self) -> int:
        return self._count("ok")

    @property
    def skipped(self) -> int:
        return self._count("skipped")

    @property
    def failed(self) -> int:
        return self._count("failed")

    def __iter__(self):
        return iter(self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, index):
        return self.outcomes[index]

    def __str__(self) -> str:
        return f"pipeline: {self.ok} ok, {self.skipped} skipped, {self.failed} failed"


def parse_stages(stages: str | Sequence[str] | None = "all") -> list[str]:
    """Turn ``"all"``, ``"tokenize,train"`` or a list into a validated stage list.

    Order always follows :data:`STAGES`, so ``--stages sample,tokenize`` still
    runs tokenize first — a stage cannot consume data that was never produced.
    """
    if stages is None or (isinstance(stages, str) and stages.strip().lower() == "all"):
        return list(STAGES)

    if isinstance(stages, str):
        requested = [part.strip().lower() for part in stages.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in stages]

    canonical = {name: name for name in STAGES}
    canonical.update({alias: real for alias, real in _STAGE_ALIASES.items()})
    unknown = [name for name in requested if name not in canonical]
    if unknown:
        raise ValueError(
            f"unknown stage(s) {unknown}; pick from {list(STAGES)} or 'all'"
        )
    wanted = {canonical[name] for name in requested}
    return [name for name in STAGES if name in wanted]


def pipeline(
    corpus: str = "data",
    num_merges: int = DEFAULT_NUM_MERGES,
    granularity: str = "paragraph",
    val_ratio: float = 0.10,
    preset: str = "tiny",
    stages: str | Sequence[str] = "all",
    retrain_vocab: bool = False,
    run_dir: str = "runs",
    cache_dir: str = "data/cache",
    processed_dir: str = "data/processed",
    prompt: str = "The monster",
    checkpoint: str | None = None,
    epochs: int | None = None,
    quiet: bool = False,
    trace: bool = False,
) -> PipelineReport:
    """Run the pipeline end to end and print a per-stage report.

    Args:
      corpus: Corpus file or directory.
      num_merges: BPE merges the vocabulary must have.
      granularity: `file` or `paragraph` — the train/val split unit.
      val_ratio: Fraction of documents held out for validation.
      preset: `debug`, `tiny` or `gpt2-small`.
      stages: `all`, a comma list (`train-tokenizer,train`) or a list.
      retrain_vocab: Learn the BPE merges again instead of reusing the cache.
      run_dir: Where `train` writes checkpoints and logs.
      cache_dir: Where vocabulary caches live.
      processed_dir: Where the token file is written/reused.
      prompt: Prompt for the sampling stage.
      checkpoint: Evaluate/sample this checkpoint instead of the one `train` returns.
      epochs: Override the training epoch count.
      quiet: Only print the summary, not each stage's output.
      trace: Print the full traceback of a failing stage.
    """
    plan = parse_stages(stages)
    say = (lambda *a, **k: None) if quiet else print
    outcomes: list[StageOutcome] = []

    say(f"Pipeline: {' -> '.join(plan)}")
    say(f"          corpus={corpus!r} granularity={granularity} preset={preset!r}")

    tokenized: TokenizeResult | None = None
    ckpt = checkpoint

    for name in plan:
        say("")
        say(f"== {name} ==")
        try:
            if name == "train-tokenizer":
                trained_vocab = train_tokenizer(
                    corpus=corpus,
                    num_merges=num_merges,
                    granularity=granularity,
                    cache_dir=cache_dir,
                    retrain=retrain_vocab,
                    quiet=quiet,
                )
                outcomes.append(StageOutcome(name, "ok", str(trained_vocab), trained_vocab))
            elif name == "tokenize":
                tokenized = tokenize(
                    corpus=corpus,
                    num_merges=num_merges,
                    granularity=granularity,
                    val_ratio=val_ratio,
                    cache_dir=cache_dir,
                    processed_dir=processed_dir,
                    quiet=quiet,
                )
                outcomes.append(StageOutcome(name, "ok", str(tokenized), tokenized))
            elif name == "train":
                trained = train(
                    preset=preset,
                    corpus=corpus,
                    num_merges=num_merges,
                    granularity=granularity,
                    token_file=tokenized.token_file if tokenized else None,
                    epochs=epochs,
                    run_dir=run_dir,
                )
                ckpt = str(trained) if trained else ckpt
                outcomes.append(StageOutcome(name, "ok", f"checkpoint {ckpt}", trained))
            elif name == "eval":
                if ckpt is None:
                    outcomes.append(
                        StageOutcome(name, "skipped", "no checkpoint: train produced none")
                    )
                else:
                    evaluate(ckpt=ckpt, corpus=corpus, granularity=granularity)
                    outcomes.append(StageOutcome(name, "ok", f"evaluated {ckpt}"))
            else:  # sample
                if ckpt is None:
                    outcomes.append(
                        StageOutcome(name, "skipped", "no checkpoint: train produced none")
                    )
                else:
                    text = sample(ckpt=ckpt, prompt=prompt)
                    outcomes.append(StageOutcome(name, "ok", f"{len(text)} chars", text))
        except NotImplementedError as exc:
            outcomes.append(StageOutcome(name, "skipped", str(exc)))
        except Exception as exc:  # noqa: BLE001 - reported, then the run stops
            outcomes.append(
                StageOutcome(name, "failed", str(exc), error=traceback.format_exc())
            )
            break

    report = PipelineReport(outcomes)
    _print_report(report)
    if trace:
        for outcome in outcomes:
            if outcome.status == "failed" and outcome.error:
                print(outcome.error)

    if report.failed:
        raise SystemExit(1)
    return report


def _print_report(report: PipelineReport) -> None:
    """One aligned line per stage plus a count."""
    width = max((len(outcome.name) for outcome in report.outcomes), default=0)
    print("")
    print("Summary")
    for outcome in report.outcomes:
        detail = outcome.detail.strip().splitlines()[0] if outcome.detail.strip() else ""
        print(f"  {outcome.name.ljust(width)}  {outcome.status:<8} {detail}")
    print(f"  -> {report}")
