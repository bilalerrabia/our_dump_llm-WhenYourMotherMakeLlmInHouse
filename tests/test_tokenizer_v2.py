"""Tests for the fast, versioned BPE tokenizer (``our_llm.tokenizer``)."""

import json
import os

import pytest

from our_llm.tokenizer import (
    BPETokenizer,
    EOT_TOKEN,
    MERGE_START,
    PAD_TOKEN,
    SPECIAL_IDS,
    vocab_size,
)
from our_llm.tokenizer.bpe import corpus_hash, load_or_train, tokenizer_cache_path
from src.tokenizer import Tokenizer as LegacyTokenizer

SAMPLE = (
    "The Project Gutenberg eBook of Frankenstein. Dr. Frankenstein said: "
    "'It is alive!' -- 215 pounds, 你好, 🚀\n\nYour journey starts with one step."
)

LEGACY_CACHE = "tokenizer_vocab_merges.json"


def slow_encode(tokenizer: BPETokenizer, text: str) -> list[int]:
    """The original O(num_merges) algorithm: replay every merge over all words."""
    words = tokenizer.encoder(text)
    for new_id, pair in sorted(tokenizer.merges.items()):
        words = BPETokenizer.replace(words, pair, new_id)
    return [token for word in words for token in word]


@pytest.fixture
def trained() -> BPETokenizer:
    return BPETokenizer().train(SAMPLE * 20, num_merges=40, show_progress=False)


class TestRoundTrip:
    def test_ascii(self, trained):
        text = "hello world, this is a test!"
        assert trained.decode(trained.encode(text)) == text

    def test_utf8(self, trained):
        text = "héllo wörld — 你好 مرحبا 🚀"
        assert trained.decode(trained.encode(text)) == text

    def test_empty(self, trained):
        assert trained.encode("") == []
        assert trained.decode([]) == ""

    def test_whitespace_is_preserved(self, trained):
        text = "a\n\n  b\tc   d\n"
        assert trained.decode(trained.encode(text)) == text


class TestVocabularyLayout:
    def test_untrained_is_bytes_only(self):
        tok = BPETokenizer()
        assert tok.num_merges == 0
        assert tok.vocab_size == MERGE_START
        assert max(tok.encode("hello world")) < 256

    def test_special_ids(self):
        assert SPECIAL_IDS[EOT_TOKEN] == 256
        assert SPECIAL_IDS[PAD_TOKEN] == 257
        assert MERGE_START == 258

    def test_merges_start_after_specials(self, trained):
        assert trained.num_merges == 40
        assert min(trained.merges) == MERGE_START
        assert trained.vocab_size == MERGE_START + 40
        assert trained.vocab_size == vocab_size(40)

    def test_ids_stay_inside_vocab(self, trained):
        for token_id in trained.encode(SAMPLE):
            assert 0 <= token_id < trained.vocab_size


class TestSpecialTokens:
    def test_not_recognised_unless_allowed(self, trained):
        text = f"a{EOT_TOKEN}b"
        plain = trained.encode(text)
        allowed = trained.encode(text, allowed_special=[EOT_TOKEN])
        assert SPECIAL_IDS[EOT_TOKEN] not in plain
        assert SPECIAL_IDS[EOT_TOKEN] in allowed
        assert len(allowed) < len(plain)

    def test_round_trip_with_specials(self, trained):
        text = f"first doc{EOT_TOKEN}second doc"
        ids = trained.encode(text, allowed_special=[EOT_TOKEN])
        assert trained.decode(ids) == text
        assert trained.decode(ids, skip_special=True) == "first docsecond doc"


class TestFastEncode:
    def test_matches_slow_replay(self, trained):
        assert trained.encode(SAMPLE) == slow_encode(trained, SAMPLE)

    def test_matches_slow_replay_on_large_text(self, trained):
        text = SAMPLE * 50
        assert trained.encode(text) == slow_encode(trained, text)

    def test_memoisation_is_transparent(self, trained):
        first = trained.encode(SAMPLE)
        second = trained.encode(SAMPLE)
        assert first == second


class TestTraining:
    def test_matches_legacy_trainer(self):
        """The rewrite must learn exactly the same merges as `src/tokenizer.py`."""
        text = SAMPLE * 30
        legacy = LegacyTokenizer()
        legacy.train(text, num_merges=25)  # NB: the legacy train() returns ids, not self
        new = BPETokenizer().train(text, num_merges=25, show_progress=False)

        shifted = {
            new_id + 2: (
                left + 2 if left >= 256 else left,
                right + 2 if right >= 256 else right,
            )
            for new_id, (left, right) in legacy.map_replace.items()
        }
        assert new.merges == shifted

        legacy_ids = [i + 2 if i >= 256 else i for i in legacy.encode(text)]
        assert new.encode(text) == legacy_ids

    def test_train_is_deterministic(self):
        a = BPETokenizer().train(SAMPLE, num_merges=20, show_progress=False)
        b = BPETokenizer().train(SAMPLE, num_merges=20, show_progress=False)
        assert a.merges == b.merges

    def test_stops_when_no_pairs_left(self):
        tok = BPETokenizer().train("aaaa", num_merges=100, show_progress=False)
        assert 0 < tok.num_merges < 100


class TestPersistence:
    def test_save_load_round_trip(self, trained, tmp_path):
        path = str(tmp_path / "vocab.json")
        trained.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.merges == trained.merges
        assert loaded.encode(SAMPLE) == trained.encode(SAMPLE)
        assert loaded.decode(loaded.encode(SAMPLE)) == SAMPLE

    def test_saved_file_is_self_describing(self, trained, tmp_path):
        path = str(tmp_path / "vocab.json")
        trained.save(path)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["_meta"]["merge_start"] == MERGE_START
        assert payload["_meta"]["num_merges"] == 40
        assert payload["_meta"]["vocab_size"] == 298

    def test_legacy_file_is_migrated(self, trained, tmp_path):
        """A pre-special-token cache (merges from 256) loads into the new layout."""
        legacy_path = str(tmp_path / "legacy.json")
        legacy = LegacyTokenizer()
        legacy.train(SAMPLE * 30, num_merges=25)
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump({str(k): list(v) for k, v in legacy.map_replace.items()}, handle)

        migrated = BPETokenizer.load(legacy_path)
        assert min(migrated.merges) == MERGE_START
        assert migrated.num_merges == 25
        assert migrated.decode(migrated.encode(SAMPLE)) == SAMPLE
        for left, right in migrated.merges.values():
            assert 0 <= left < migrated.vocab_size
            assert 0 <= right < migrated.vocab_size

    @pytest.mark.skipif(not os.path.exists(LEGACY_CACHE), reason="no legacy cache in repo")
    def test_repo_legacy_cache_loads_and_round_trips(self):
        tok = BPETokenizer.load(LEGACY_CACHE)
        assert tok.num_merges == 14_970
        assert tok.vocab_size == MERGE_START + 14_970
        text = SAMPLE * 10
        assert tok.decode(tok.encode(text)) == text


class TestCacheKeying:
    def test_path_depends_on_corpus_and_merges(self, tmp_path):
        a = tokenizer_cache_path(str(tmp_path), corpus_hash("aaa"), 100)
        b = tokenizer_cache_path(str(tmp_path), corpus_hash("bbb"), 100)
        c = tokenizer_cache_path(str(tmp_path), corpus_hash("aaa"), 200)
        assert a != b and a != c
        assert "aaa" not in os.path.basename(a)  # hashed, not literal

    def test_load_or_train_reuses_the_cache(self, tmp_path):
        text = SAMPLE * 20
        first = load_or_train(text, 30, cache_dir=str(tmp_path), show_progress=False)
        path = tokenizer_cache_path(str(tmp_path), corpus_hash(text), 30)
        assert os.path.exists(path)
        mtime = os.path.getmtime(path)

        second = load_or_train(text, 30, cache_dir=str(tmp_path), show_progress=False)
        assert second.merges == first.merges
        assert os.path.getmtime(path) == mtime  # nothing rewritten

    def test_different_merge_count_retrains(self, tmp_path):
        text = SAMPLE * 20
        load_or_train(text, 30, cache_dir=str(tmp_path), show_progress=False)
        other = load_or_train(text, 10, cache_dir=str(tmp_path), show_progress=False)
        assert other.num_merges == 10

    def test_a_cache_is_reused_even_when_training_stopped_early(self, tmp_path):
        """`num_merges` is a ceiling: the corpus runs out of pairs before it.

        Comparing `loaded.num_merges` to the request made every run retrain the
        same vocabulary (68 s for the real corpus); the *requested* count is what
        says whether a cache is the right one.
        """
        text = SAMPLE * 20
        first = load_or_train(text, 10_000, cache_dir=str(tmp_path), show_progress=False)
        assert first.num_merges < 10_000  # stopped early ...
        assert first.satisfies(10_000)  # ... but this is the cache for that request

        path = tokenizer_cache_path(str(tmp_path), corpus_hash(text), 10_000)
        mtime = os.path.getmtime(path)
        second = load_or_train(text, 10_000, cache_dir=str(tmp_path), show_progress=False)
        assert second.merges == first.merges
        assert os.path.getmtime(path) == mtime  # loaded, not retrained

    def test_the_requested_count_survives_a_save_load_round_trip(self, tmp_path):
        tokenizer = BPETokenizer().train(SAMPLE * 20, 10_000, show_progress=False)
        path = str(tmp_path / "vocab.json")
        tokenizer.save(path)
        assert BPETokenizer.load(path).requested_merges == 10_000


class TestReadableVocab:
    def test_contains_bytes_specials_and_merges(self, trained):
        vocab = trained.to_readable_vocab()
        assert vocab["A"] == ord("A")
        assert vocab[EOT_TOKEN] == SPECIAL_IDS[EOT_TOKEN]
        assert len(vocab) == trained.vocab_size
