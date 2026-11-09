"""The command line: every pipeline stage exposed as a `fire` command.

    uv run python main.py --help
    uv run python main.py train-tokenizer --num-merges 14970   # learn the BPE merges
    uv run python main.py tokenize --corpus data              # corpus -> token file
    uv run python main.py train --preset debug                # == train-llm
    uv run python main.py pipeline                            # all of it, in order
    uv run our-llm pipeline            # same thing via the console script

Why a class instead of a dict: Fire takes each *member* of the component as a
command, so a class keeps the per-command docstrings and type defaults in the
help output, while the bodies stay in :mod:`our_llm.tasks` (one file per stage).
The methods here are `staticmethod`s — pure re-exports, no state, no logic.

Fire rules worth knowing:

* a method named ``train_tokenizer`` is the command ``train-tokenizer``; both
  spellings work on the command line.
* ``--num-merges`` and ``--num_merges`` are the same flag (dashes normalise to
  underscores).
* booleans take ``--force`` / ``--noforce``; ``--nowrite`` also works for `write`.
* ``kw: type = default`` in a stage function becomes ``--kw TYPE`` in ``--help``.
* Fire's own flags go after a separator: ``main.py tokenize -- --help``.
* Fire **silently ignores** an unknown ``--flag``. A typo'd flag would then run
  the stage with its defaults (and `train` would happily start a long run), so
  :func:`main` validates flags against the chosen command's signature first.
"""

from __future__ import annotations

import difflib
import inspect
import sys

import fire

from our_llm.tasks import COMMANDS

#: Flags Fire itself understands; never validated against a command signature.
FIRE_FLAGS = frozenset(
    {"help", "h", "verbose", "v", "interactive", "i", "trace", "completion", "separator"}
)


class OurLLM:
    """A GPT built from scratch: corpus -> BPE -> tokens -> GPT -> training -> sampling."""


# One Fire command per stage, bound outside the class body: inside it the names
# would be rebound as the body runs, so `train_llm = staticmethod(train)` placed
# after `train = staticmethod(train)` would wrap the *wrapper* instead of the
# function. `COMMANDS` stays the single source of truth (`tasks/__init__.py`),
# and Fire reads the command list straight off the class.
for _name, _task in COMMANDS.items():
    setattr(OurLLM, _name, staticmethod(_task))
del _name, _task


def _canonical_command(token: str) -> str | None:
    """Map a typed command (`train-tokenizer`, `train_tokenizer`) to its key."""
    if token in COMMANDS:
        return token
    key = token.replace("-", "_")
    return key if key in COMMANDS else None


def _split_command(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return ``(command, flag_tokens)`` from raw argv.

    A token after ``--flag`` (with no ``=``) is its value, so it is not the
    command; everything after a bare ``--`` belongs to Fire and is skipped.
    """
    command: str | None = None
    flags: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            break
        if token.startswith("-"):
            flags.append(token)
            if "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 1  # its value
        elif command is None:
            command = _canonical_command(token)
        index += 1
    return command, flags


def _valid_flags(command: str) -> tuple[set[str], list[str]]:
    """Accepted flag names (all spellings) and the canonical ones for messages.

    Fire accepts ``--num_merges``, ``--num-merges`` and ``--nummerges``
    interchangeably, and ``--nofoo`` for any boolean, so all of those are valid.
    """
    accepted: set[str] = set()
    canonical: list[str] = []
    for parameter in inspect.signature(COMMANDS[command]).parameters.values():
        name = parameter.name
        canonical.append(name)
        accepted.update({name, name.replace("_", "-"), name.replace("_", "")})
        is_bool = parameter.annotation in (bool, "bool") or isinstance(parameter.default, bool)
        if is_bool:
            accepted.add(f"no{name.replace('_', '')}")
    return accepted, canonical


def validate_flags(argv: list[str]) -> None:
    """Reject ``--flags`` that the chosen command does not have.

    Fire would drop them silently and run with defaults; for `train` that is the
    difference between "1 epoch, debug" and "10 epochs on the tiny model".
    """
    command, flags = _split_command(argv)
    if command is None:
        return

    accepted, canonical = _valid_flags(command)
    squashed = {name.replace("-", "") for name in accepted}
    for token in flags:
        name = token.lstrip("-").split("=", 1)[0].replace("-", "_")
        if name.replace("_", "") in FIRE_FLAGS:
            continue
        if name.replace("_", "") in squashed:
            continue
        close = difflib.get_close_matches(name.replace("_", "-"), canonical, n=1)
        hint = f" did you mean --{close[0]}?" if close else ""
        raise SystemExit(
            f"error: '{command}' has no flag --{name}.{hint}\n"
            f"       valid flags: {', '.join('--' + name for name in canonical)}"
        )


def main(argv: list[str] | None = None) -> None:
    """Run the CLI. ``argv=None`` means "use ``sys.argv[1:]``" (Fire's default)."""
    args = list(sys.argv[1:] if argv is None else argv)
    validate_flags(args)
    try:
        fire.Fire(OurLLM, command=args, name="our-llm")
    except NotImplementedError as exc:
        # Stages that exist only as a signature yet (M4/M5) say so instead of
        # printing a traceback.
        print(f"not implemented yet: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, FileNotFoundError) as exc:
        # Bad arguments (unknown preset/stage, missing corpus): a message, not a stack.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
