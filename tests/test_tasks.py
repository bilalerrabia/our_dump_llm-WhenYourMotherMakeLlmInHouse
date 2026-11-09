"""Tests for the task layer: one function per stage, plus the pipeline runner.

The stages are deliberately thin over `our_llm.data` / `our_llm.tokenizer`, so
these tests focus on the *contract*: every stage is importable on its own, the
unimplemented ones say so with `NotImplementedError` (not a print), stage
selection is order-safe, and `pipeline` reports ok/skipped/failed correctly.
"""

import inspect
import os
from pathlib import Path

import pytest

from our_llm.tokenizer.bpe import BPETokenizer
from our_llm.tasks import (
    COMMANDS,
    STAGES,
    evaluate,
    pipeline,
    sample,
    tokenize,
    train,
    train_tokenizer,
)
from our_llm.tasks.pipeline import PipelineReport, StageOutcome, parse_stages

CORPUS = (
    "The monster opened its eyes and saw the night.\n\n"
    "It was on a dreary night of November that I beheld my work.\n\n"
    "Learn from me, if not by my precepts, at least by my example.\n\n"
    "Nothing is so painful to the human mind as a great and sudden change.\n"
)


@pytest.fixture()
def corpus_file(tmp_path):
    path = tmp_path / "data" / "small.txt"
    path.parent.mkdir()
    path.write_text(CORPUS, encoding="utf-8")
    return str(path.parent)


# --- the stage surface ------------------------------------------------------


class TestCommandSurface:
    def test_every_stage_is_exported_and_registered(self):
        assert set(COMMANDS) == {
            "train_tokenizer",
            "tokenize",
            "train",
            "train_llm",
            "eval",
            "sample",
            "pipeline",
        }
        assert COMMANDS["train_tokenizer"] is train_tokenizer
        assert COMMANDS["tokenize"] is tokenize
        assert COMMANDS["train"] is train
        assert COMMANDS["train_llm"] is train  # alias, one implementation
        assert COMMANDS["eval"] is evaluate
        assert COMMANDS["pipeline"] is pipeline
        assert STAGES == ("train-tokenizer", "tokenize", "train", "eval", "sample")

    def test_each_stage_lives_in_its_own_module(self):
        modules = {
            train_tokenizer.__module__,
            tokenize.__module__,
            train.__module__,
            evaluate.__module__,
            sample.__module__,
            pipeline.__module__,
        }
        assert len(modules) == 6
        assert all(module.startswith("our_llm.tasks.") for module in modules)

    def test_every_stage_has_a_docstring_and_no_required_surprises(self):
        for command in COMMANDS.values():
            assert (command.__doc__ or "").strip(), command
            signature = inspect.signature(command)
            # nothing positional-only or keyword-only: Fire maps them to flags
            assert all(
                param.kind
                in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                for param in signature.parameters.values()
            )

    def test_importing_tasks_is_cheap(self):
        # `our_llm.tasks` must not need torch: the model is only imported from M4 on.
        import subprocess
        import sys

        code = "import sys, our_llm.tasks; print('torch' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestTokenizeStage:
    def test_returns_a_summary_and_writes_the_token_file(self, corpus_file, tmp_path):
        result = tokenize(
            corpus=corpus_file,
            num_merges=40,
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(tmp_path / "processed"),
            quiet=True,
        )
        assert result.n_tokens > 0
        assert result.num_merges == 40
        assert result.vocab_size == 258 + 40
        assert result.train_docs + result.val_docs == result.n_documents
        assert result.token_file and result.token_file.endswith(".bin")
        assert not result.reused
        assert "tokens" in str(result)

    def test_second_run_reuses_the_token_file(self, corpus_file, tmp_path):
        kwargs = dict(
            corpus=corpus_file,
            num_merges=40,
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(tmp_path / "processed"),
            quiet=True,
        )
        first = tokenize(**kwargs)
        second = tokenize(**kwargs)
        assert second.reused
        assert second.n_tokens == first.n_tokens
        assert second.token_file == first.token_file

    def test_nowrite_leaves_the_disk_alone(self, corpus_file, tmp_path):
        processed = tmp_path / "processed"
        result = tokenize(
            corpus=corpus_file,
            num_merges=40,
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(processed),
            write=False,
            quiet=True,
        )
        assert result.n_tokens > 0
        assert result.token_file is None
        assert not processed.exists()

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No such corpus path"):
            tokenize(corpus=str(tmp_path / "nope"), quiet=True)


class TestTrainTokenizerStage:
    """Stage 0: the merges themselves are learned here, not by `tokenize`."""

    def test_trains_and_caches_the_merges(self, corpus_file, tmp_path):
        cache_dir = tmp_path / "cache"
        result = train_tokenizer(
            corpus=corpus_file, num_merges=40, cache_dir=str(cache_dir), quiet=True
        )
        assert result.trained
        assert result.num_merges == 40
        assert result.vocab_size == 258 + 40
        assert result.corpus_chars >= 200  # the reader trims trailing whitespace
        assert result.n_documents == 4
        assert os.path.exists(result.cache_path)
        assert cache_dir in Path(result.cache_path).parents
        assert "trained in" in str(result)

    def test_second_run_reuses_the_cache(self, corpus_file, tmp_path):
        kwargs = dict(corpus=corpus_file, num_merges=40, cache_dir=str(tmp_path / "cache"))
        first = train_tokenizer(**kwargs, quiet=True)
        second = train_tokenizer(**kwargs, quiet=True)
        assert second.trained is False
        assert second.seconds == 0.0
        assert second.cache_path == first.cache_path
        assert second.vocab_size == first.vocab_size
        assert "already trained" in str(second)

    def test_retrain_learns_them_again(self, corpus_file, tmp_path):
        kwargs = dict(corpus=corpus_file, num_merges=40, cache_dir=str(tmp_path / "cache"))
        train_tokenizer(**kwargs, quiet=True)
        again = train_tokenizer(**kwargs, retrain=True, quiet=True)
        assert again.trained

    def test_it_is_deterministic(self, corpus_file, tmp_path):
        first = train_tokenizer(
            corpus=corpus_file, num_merges=40, cache_dir=str(tmp_path / "a"), quiet=True
        )
        second = train_tokenizer(
            corpus=corpus_file, num_merges=40, cache_dir=str(tmp_path / "b"), quiet=True
        )
        assert first.vocab_size == second.vocab_size
        assert (
            BPETokenizer.load(first.cache_path).merges
            == BPETokenizer.load(second.cache_path).merges
        )

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No such corpus path"):
            train_tokenizer(corpus=str(tmp_path / "nope"), quiet=True)

    def test_tokenize_then_reports_the_trained_vocabulary(self, corpus_file, tmp_path):
        cache_dir = str(tmp_path / "cache")
        train_tokenizer(corpus=corpus_file, num_merges=40, cache_dir=cache_dir, quiet=True)
        result = tokenize(
            corpus=corpus_file,
            num_merges=40,
            cache_dir=cache_dir,
            processed_dir=str(tmp_path / "processed"),
            quiet=True,
        )
        assert result.vocab_size == 258 + 40
        assert result.num_merges == 40


class TestUnimplementedStages:
    def test_train_raises_with_the_milestone_and_the_size(self):
        with pytest.raises(NotImplementedError, match="M4") as info:
            train(preset="debug")
        assert "1,078,400 parameters" in str(info.value)

    def test_evaluate_raises(self):
        with pytest.raises(NotImplementedError, match="M4"):
            evaluate(ckpt="runs/best.pt")

    def test_sample_raises(self):
        with pytest.raises(NotImplementedError, match="M5"):
            sample(ckpt="runs/best.pt")


# --- stage selection --------------------------------------------------------


class TestParseStages:
    def test_all_is_every_stage_in_order(self):
        assert parse_stages("all") == list(STAGES)
        assert parse_stages(None) == list(STAGES)

    def test_order_is_always_the_pipeline_order(self):
        assert parse_stages("sample,tokenize") == ["tokenize", "sample"]
        assert parse_stages(["eval", "train"]) == ["train", "eval"]

    def test_accepts_a_comma_string_with_spaces(self):
        assert parse_stages("tokenize, train") == ["tokenize", "train"]

    def test_aliases(self):
        assert parse_stages("evaluate") == ["eval"]
        assert parse_stages("tokenizer") == ["train-tokenizer"]
        assert parse_stages("vocab") == ["train-tokenizer"]
        assert parse_stages("bpe") == ["train-tokenizer"]
        assert parse_stages("llm") == ["train"]
        assert parse_stages("train_llm") == ["train"]

    def test_both_spellings_of_the_tokenizer_stage(self):
        assert parse_stages("train-tokenizer") == ["train-tokenizer"]
        assert parse_stages("train_tokenizer") == ["train-tokenizer"]
        assert parse_stages("Train_Tokenizer") == ["train-tokenizer"]

    def test_unknown_stage_names_the_choices(self):
        with pytest.raises(ValueError, match="unknown stage"):
            parse_stages("fine_tune")

    def test_empty_string_means_nothing(self):
        assert parse_stages("") == []


# --- the runner -------------------------------------------------------------


class TestPipeline:
    def test_train_tokenizer_stage_trains_the_merges(self, corpus_file, tmp_path):
        report = pipeline(
            corpus=corpus_file,
            num_merges=40,
            stages="train-tokenizer",
            cache_dir=str(tmp_path / "cache"),
            quiet=True,
        )
        assert [outcome.name for outcome in report] == ["train-tokenizer"]
        assert report[0].status == "ok"
        assert report[0].value.trained
        assert report[0].value.vocab_size == 258 + 40

    def test_train_everything_learns_merges_then_tokens(self, corpus_file, tmp_path):
        """`--stages all` is the two training stages plus the data in between."""
        report = pipeline(
            corpus=corpus_file,
            num_merges=40,
            preset="debug",
            stages="tokenizer,tokenize,train",
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(tmp_path / "processed"),
            run_dir=str(tmp_path / "runs"),
            quiet=True,
        )
        assert [outcome.name for outcome in report] == ["train-tokenizer", "tokenize", "train"]
        assert report[0].value.trained is True  # stage 0 really trained the BPE
        assert report[1].status == "ok"
        assert report[1].value.vocab_size == report[0].value.vocab_size
        assert report[2].status == "skipped"  # the GPT loop lands in M4

    def test_retrain_vocab_forces_the_merges_again(self, corpus_file, tmp_path):
        kwargs = dict(
            corpus=corpus_file,
            num_merges=40,
            stages="train-tokenizer",
            cache_dir=str(tmp_path / "cache"),
            quiet=True,
        )
        pipeline(**kwargs)
        warm = pipeline(**kwargs)
        assert warm[0].value.trained is False
        forced = pipeline(**kwargs, retrain_vocab=True)
        assert forced[0].value.trained is True

    def test_tokenize_only(self, corpus_file, tmp_path):
        report = pipeline(
            corpus=corpus_file,
            num_merges=40,
            stages="tokenize",
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(tmp_path / "processed"),
            quiet=True,
        )
        assert isinstance(report, PipelineReport)
        assert [outcome.name for outcome in report] == ["tokenize"]
        assert report[0].status == "ok"
        assert report.ok == 1 and report.failed == 0

    def test_full_run_skips_the_unbuilt_stages(self, corpus_file, tmp_path):
        report = pipeline(
            corpus=corpus_file,
            num_merges=40,
            preset="debug",
            stages="all",
            cache_dir=str(tmp_path / "cache"),
            processed_dir=str(tmp_path / "processed"),
            run_dir=str(tmp_path / "runs"),
            quiet=True,
        )
        statuses = {outcome.name: outcome.status for outcome in report}
        assert statuses["train-tokenizer"] == "ok"
        assert statuses["tokenize"] == "ok"
        # train is not built yet, so eval/sample have no checkpoint to consume
        assert statuses["train"] == "skipped"
        assert statuses["eval"] == "skipped"
        assert statuses["sample"] == "skipped"
        assert report.failed == 0
        assert len(report) == len(STAGES)
        assert "skipped" in str(report)

    def test_a_failing_stage_stops_the_run(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as info:
            pipeline(corpus=str(tmp_path / "does-not-exist"), stages="all", quiet=True)
        assert info.value.code == 1
        report = capsys.readouterr().out
        assert "failed" in report
        # only the first stage ran: nothing after the failure was attempted
        assert "pipeline: 0 ok, 0 skipped, 1 failed" in report

    def test_report_counts(self):
        report = PipelineReport(
            [
                StageOutcome("tokenize", "ok"),
                StageOutcome("train", "skipped"),
                StageOutcome("eval", "failed"),
            ]
        )
        assert (report.ok, report.skipped, report.failed) == (1, 1, 1)
        assert str(report) == "pipeline: 1 ok, 1 skipped, 1 failed"
