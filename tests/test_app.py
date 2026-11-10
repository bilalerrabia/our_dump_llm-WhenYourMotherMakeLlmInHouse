"""Tests for the fire-based entry point (`our_llm.app`).

The pieces worth pinning down: every task is exposed as a Fire command, Fire's
aliases are all accepted, and a *typo'd* flag is rejected instead of being
silently ignored by Fire (which would run a stage with its defaults).
"""

import pytest

from our_llm.app import FIRE_FLAGS, OurLLM, _split_command, _valid_flags, main, validate_flags
from our_llm.tasks import COMMANDS


class TestFireComponent:
    def test_every_task_is_a_command(self):
        for name, task in COMMANDS.items():
            assert hasattr(OurLLM, name), name
            assert getattr(OurLLM, name) is task

    def test_nothing_else_is_public(self):
        public = {name for name in dir(OurLLM) if not name.startswith("_")}
        assert public == set(COMMANDS)

    def test_no_state_and_no_instantiation_needed(self):
        assert OurLLM.__dict__.get("__init__") is None  # plain object constructor


class TestSplitCommand:
    @pytest.mark.parametrize(
        "argv,command",
        [
            (["tokenize"], "tokenize"),
            (["tokenize", "--num_merges", "40"], "tokenize"),
            (["--stages", "tokenize,train", "pipeline"], "pipeline"),
            (["train-tokenizer"], "train_tokenizer"),
            (["train_tokenizer"], "train_tokenizer"),
            (["train-tokenizer", "--num-merges", "40"], "train_tokenizer"),
            (["train-llm"], "train_llm"),
            (["--", "--help"], None),
            ([], None),
            (["nonsense"], None),
        ],
    )
    def test_finds_the_command(self, argv, command):
        assert _split_command(argv)[0] == command

    def test_flag_values_are_not_mistaken_for_the_command(self):
        command, flags = _split_command(["--corpus", "tokenize", "eval"])
        assert command == "eval"
        assert flags == ["--corpus"]

    def test_fire_flags_after_the_separator_are_ignored(self):
        assert _split_command(["tokenize", "--", "--help"]) == ("tokenize", [])


class TestValidFlags:
    def test_all_spellings_and_negations_are_accepted(self):
        accepted, canonical = _valid_flags("tokenize")
        assert canonical[0] == "corpus"
        for spelling in ("num_merges", "num-merges", "nummerges", "noretrain", "nowrite"):
            assert spelling in accepted, spelling

    def test_a_non_bool_gets_no_negation(self):
        accepted, _ = _valid_flags("tokenize")
        assert "novalratio" not in accepted
        assert "noval_ratio" not in accepted


class TestValidateFlags:
    @pytest.mark.parametrize(
        "argv",
        [
            ["tokenize"],
            ["tokenize", "--corpus", "data"],
            ["tokenize", "--num-merges=40"],
            ["tokenize", "--nowrite"],
            ["pipeline", "--stages", "tokenize", "--preset", "debug"],
            ["pipeline", "--retrain-vocab", "--stages", "all"],
            ["train-tokenizer", "--num-merges", "40"],
            ["train-tokenizer", "--noretrain"],  # `--retrain` negated
            ["train_tokenizer", "--quiet"],
            ["train-llm", "--preset", "debug"],
            ["tokenize", "--", "--help"],
            ["--", "--help"],
            [],  # no command: Fire prints the command list
        ],
    )
    def test_accepts_good_input(self, argv):
        validate_flags(argv)

    def test_typo_is_rejected_with_a_suggestion(self):
        with pytest.raises(SystemExit) as info:
            validate_flags(["train", "--prets", "tiny"])
        message = str(info.value.code)
        assert "no flag --prets" in message
        assert "did you mean --preset?" in message

    def test_a_flag_from_another_command_is_rejected(self):
        # The exact footgun: `--nowrite` exists on tokenize but not on pipeline.
        with pytest.raises(SystemExit):
            validate_flags(["pipeline", "--nowrite"])

    def test_message_lists_the_valid_flags(self):
        with pytest.raises(SystemExit) as info:
            validate_flags(["eval", "--nope"])
        assert "--ckpt" in str(info.value.code)

    def test_fire_own_flags_pass(self):
        for name in FIRE_FLAGS:
            validate_flags(["tokenize", f"--{name}"])


class TestMain:
    def test_not_implemented_stage_exits_one_without_a_traceback(self, capsys):
        # `sample` is the one stage left for M5; it must not print a traceback.
        with pytest.raises(SystemExit) as info:
            main(["sample", "--ckpt", "runs/missing.pt"])
        assert info.value.code == 1
        assert "not implemented yet" in capsys.readouterr().err

    def test_bad_option_value_exits_one(self, capsys):
        with pytest.raises(SystemExit) as info:
            main(["pipeline", "--stages", "nope"])
        assert info.value.code == 1
        assert "unknown stage" in capsys.readouterr().err

    def test_help_exits_cleanly(self, capsys):
        # Fire signals help/version with FireExit(0), a SystemExit subclass.
        with pytest.raises(SystemExit) as info:
            main(["--", "--help"])
        assert info.value.code == 0
        # Fire prints help to stderr (the "INFO: Showing help" hint goes to stdout).
        assert "tokenize" in capsys.readouterr().err
