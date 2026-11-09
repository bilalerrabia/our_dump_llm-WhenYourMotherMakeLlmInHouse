"""Entry point — `python main.py <command>` driven by python-fire.

    uv run python main.py --help
    uv run python main.py tokenize --corpus data --num-merges 14970
    uv run python main.py train --preset tiny              # M4
    uv run python main.py sample --ckpt runs/best.pt       # M5
    uv run python main.py pipeline                         # everything, in order

Nothing pipeline-shaped lives here: each stage is a module under
``src/our_llm/tasks/`` (`tokenize.py`, `train.py`, `evaluate.py`, `sample.py`)
and ``tasks/pipeline.py`` chains them. This file only hands the imported
functions to Fire (see ``our_llm/app.py``), so the same commands are available
from the console script: `uv run our-llm pipeline`.

The original pure-Python walkthrough lives in
``examples/pure_python/pipeline_demo.py``.
"""

from our_llm.app import main

if __name__ == "__main__":
    main()
