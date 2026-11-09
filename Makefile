.PHONY: all sync tokenize pipeline train sample test lint fmt

all: sync
	@uv run our-llm tokenize

sync:
	@uv sync --quiet

# Corpus -> BPE vocabulary -> encoded tokens (works today)
tokenize:
	@uv run our-llm tokenize

# Every stage in order: tokenize -> train -> eval -> sample
# (STAGES=tokenize to run just one, e.g. `make pipeline STAGES=tokenize`)
pipeline:
	@uv run our-llm pipeline --stages $${STAGES:-all} --preset $${PRESET:-tiny}

# Pretrain the GPT (ROADMAP M4) / generate text (M5)
train:
	@uv run our-llm train --preset $${PRESET:-tiny}

sample:
	@uv run our-llm sample --ckpt $${CKPT:?set CKPT=runs/<id>/best.pt} --prompt "$${PROMPT:-The monster}"

# Overfits one batch in seconds: the fastest way to know the model is wired right
overfit:
	@uv run pytest -q tests/test_overfit.py

test:
	@uv run pytest -q

fmt:
	@uv run ruff format .
	@uv run ruff check --fix .

lint:
	@uv run ruff check .
	@uv run ruff format --check .
