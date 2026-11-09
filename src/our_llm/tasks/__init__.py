"""One function per pipeline stage, plus the `pipeline` that chains them.

Import surface (``from our_llm.tasks import train_tokenizer, tokenize, train,
evaluate, sample, pipeline``)::

    tasks/train_tokenizer.py  corpus -> BPE vocabulary (learns the merges)  (done, M1)
    tasks/tokenize.py         vocabulary -> token file + split              (done, M2)
    tasks/train.py            token file -> GPT checkpoint                  (M4)
    tasks/evaluate.py         loss / perplexity of a checkpoint             (M4)
    tasks/sample.py           text generation from a checkpoint             (M5)
    tasks/pipeline.py         runs the stages above in order

Each stage lives in its own file so it can be read, imported and tested on its
own; nothing here imports torch, so `import our_llm.tasks` stays cheap.
"""

from our_llm.tasks.evaluate import evaluate
from our_llm.tasks.pipeline import STAGES, PipelineReport, StageOutcome, parse_stages, pipeline
from our_llm.tasks.sample import sample
from our_llm.tasks.tokenize import TokenizeResult, pick_tokenizer, tokenize
from our_llm.tasks.train import train
from our_llm.tasks.train_tokenizer import TokenizerTrainResult, train_tokenizer

#: Python names Fire turns into commands (`train_tokenizer` -> `train-tokenizer`),
#: in execution order. `app.py` builds its component from this dict.
#: `train_llm` is an alias for `train` so the two training stages read as a pair.
COMMANDS: dict[str, object] = {
    "train_tokenizer": train_tokenizer,
    "tokenize": tokenize,
    "train": train,
    "train_llm": train,
    "eval": evaluate,
    "sample": sample,
    "pipeline": pipeline,
}

__all__ = [
    "COMMANDS",
    "STAGES",
    "PipelineReport",
    "StageOutcome",
    "TokenizeResult",
    "TokenizerTrainResult",
    "evaluate",
    "parse_stages",
    "pick_tokenizer",
    "pipeline",
    "sample",
    "tokenize",
    "train",
    "train_tokenizer",
]
