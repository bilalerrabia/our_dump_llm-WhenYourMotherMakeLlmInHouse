.PHONY: all test

all:
	@clear
	@uv sync
	@uv run python3 main.py

test:
	@uv run pytest -q
