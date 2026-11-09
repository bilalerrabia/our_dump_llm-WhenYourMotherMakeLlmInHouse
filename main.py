"""Entry point — an alias for the CLI.

The real pipeline lives in the `our_llm` package:

    uv run our-llm tokenize      # corpus -> BPE -> tokens
    uv run our-llm train         # pretrain the GPT (ROADMAP M4)
    uv run our-llm sample        # generate text  (ROADMAP M5)
    uv run pytest -q             # tests

`python main.py` is kept for muscle memory; it forwards to `our-llm`.
The original pure-Python walkthrough now lives in
``examples/pure_python/pipeline_demo.py``.
"""

from our_llm.cli import main

if __name__ == "__main__":
    import sys

    # No arguments -> the first pipeline stage, so `python main.py` still shows something.
    raise SystemExit(main(sys.argv[1:] or ["tokenize"]))
