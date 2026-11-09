"""Tests for the on-disk token cache (`our_llm.data.tokens`)."""

import json
import os

import numpy as np
import pytest

from our_llm.data.corpus import Corpus
from our_llm.data.tokens import (
    EncodedCorpus,
    check_compatible,
    doc_spans,
    encode_corpus,
    load_meta,
    meta_path,
    open_tokens,
    sha16,
    token_file_path,
    tokens_dtype,
    vocab_fingerprint,
    write_tokens,
)
from our_llm.tokenizer import BPETokenizer, EOT_TOKEN, MERGE_START

DOCS = [
    "The monster opened its eyes.",
    "It was a dreary night of November.",
    "Learn from me, if not by my precepts.",
]


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    text = "\n\n".join(DOCS) * 8
    return BPETokenizer().train(text, num_merges=60, show_progress=False)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(docs=list(DOCS), sources=["a.txt", "b.txt", "c.txt"])


@pytest.fixture
def encoded(tokenizer, corpus) -> EncodedCorpus:
    return encode_corpus(tokenizer, corpus)


@pytest.fixture
def written(encoded, corpus, tokenizer, tmp_path):
    path = token_file_path(str(tmp_path), corpus.fingerprint, tokenizer.num_merges)
    meta = write_tokens(path, encoded, corpus=corpus, tokenizer=tokenizer)
    return path, meta, encoded


class TestDtype:
    def test_uint16_for_our_vocabulary(self):
        assert tokens_dtype(15_228) == np.dtype(np.uint16)

    def test_boundary(self):
        assert tokens_dtype(2**16) == np.dtype(np.uint16)
        assert tokens_dtype(2**16 + 1) == np.dtype(np.uint32)

    def test_huge_vocab_falls_back_to_uint32(self):
        assert tokens_dtype(100_257) == np.dtype(np.uint32)


class TestEncodeCorpus:
    def test_every_document_is_terminated_by_eot(self, encoded, tokenizer, corpus):
        eot = tokenizer.special_ids[EOT_TOKEN]
        assert len(encoded.doc_spans) == len(corpus.docs)
        for index in range(len(corpus.docs)):
            assert encoded.document(index)[-1] == eot

    def test_spans_tile_the_array_without_gaps(self, encoded):
        expected_start = 0
        for start, end in encoded.doc_spans:
            assert start == expected_start
            assert end > start
            expected_start = end
        assert expected_start == encoded.n_tokens

    def test_documents_round_trip_to_the_original_text(self, encoded, tokenizer, corpus):
        for index, doc in enumerate(corpus.docs):
            assert encoded.decode_document(tokenizer, index) == doc

    def test_dtype_matches_the_vocabulary(self, encoded, tokenizer):
        assert encoded.ids.dtype == tokens_dtype(tokenizer.vocab_size)
        assert encoded.ids.max() < tokenizer.vocab_size

    def test_no_eot_option(self, tokenizer, corpus):
        encoded = encode_corpus(tokenizer, corpus, add_eot=False)
        assert encoded.doc_spans[-1][1] == encoded.n_tokens
        assert len(encoded.ids) == sum(
            len(tokenizer.encode(doc)) for doc in corpus.docs
        )

    def test_empty_corpus(self, tokenizer):
        encoded = encode_corpus(tokenizer, Corpus())
        assert encoded.n_tokens == 0
        assert encoded.doc_spans == []
        assert encoded.ids.dtype == tokens_dtype(tokenizer.vocab_size)


class TestNaming:
    def test_merge_count_is_part_of_the_name(self, tmp_path):
        a = token_file_path(str(tmp_path), "abc", 1000)
        b = token_file_path(str(tmp_path), "abc", 2000)
        assert a != b
        assert os.path.basename(a) == "tokens-abc-m1000.bin"

    def test_sidecar_sits_next_to_the_binary(self, tmp_path):
        path = token_file_path(str(tmp_path), "abc", 10)
        assert meta_path(path) == path.replace(".bin", ".meta.json")


class TestRoundTripOnDisk:
    def test_memmap_and_read_back_agree(self, written):
        path, meta, encoded = written
        mapped = open_tokens(path)
        loaded = open_tokens(path, mmap=False)
        assert isinstance(mapped, np.memmap)
        assert np.array_equal(mapped, encoded.ids)
        assert np.array_equal(loaded, encoded.ids)
        assert mapped.dtype == np.dtype(meta["dtype"])

    def test_memmap_is_read_only(self, written):
        path, _, _ = written
        mapped = open_tokens(path)
        with pytest.raises(ValueError):
            mapped[0] = 1

    def test_meta_is_json_and_carries_the_fingerprints(self, written, corpus, tokenizer):
        path, meta, _ = written
        on_disk = load_meta(path)
        assert on_disk == json.loads(json.dumps(meta))
        assert on_disk["corpus_hash"] == corpus.fingerprint
        assert on_disk["vocab_fingerprint"] == vocab_fingerprint(tokenizer)
        assert on_disk["vocab_size"] == tokenizer.vocab_size
        assert on_disk["num_merges"] == tokenizer.num_merges
        assert on_disk["n_tokens"] == meta["n_tokens"]
        assert on_disk["n_documents"] == len(DOCS)
        assert on_disk["eot_id"] == tokenizer.special_ids[EOT_TOKEN]

    def test_sources_are_stored_once_with_a_per_document_index(self, written, corpus):
        path, meta, _ = written
        assert meta["sources"] == sorted(set(corpus.sources))
        assert len(meta["doc_source"]) == len(corpus.docs)
        assert [meta["sources"][i] for i in meta["doc_source"]] == corpus.sources

    def test_sidecar_stays_short(self, written):
        """One line per key: the spans/indices must not explode into diff noise."""
        from our_llm.data.tokens import meta_path

        path, _, _ = written
        with open(meta_path(path), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        assert len(lines) == len(written[1]) + 2  # "{", one line per key, "}"
        assert any(line.lstrip().startswith('"doc_spans":') for line in lines)

    def test_file_size_is_exactly_n_tokens_times_itemsize(self, written):
        path, meta, _ = written
        assert os.path.getsize(path) == meta["n_tokens"] * np.dtype(meta["dtype"]).itemsize

    def test_doc_spans_survive_the_round_trip(self, written):
        path, meta, encoded = written
        assert doc_spans(meta) == encoded.doc_spans

    def test_writing_creates_missing_directories(self, encoded, corpus, tokenizer, tmp_path):
        path = os.path.join(str(tmp_path), "deep", "nested", "tokens.bin")
        write_tokens(path, encoded, corpus=corpus, tokenizer=tokenizer)
        assert os.path.exists(path) and os.path.exists(meta_path(path))


class TestCompatibility:
    def test_matching_vocabulary_passes(self, written, tokenizer):
        path, meta, _ = written
        check_compatible(load_meta(path), tokenizer)

    def test_different_vocabulary_raises(self, written):
        path, meta, _ = written
        other = BPETokenizer().train("\n\n".join(DOCS), 10, show_progress=False)
        with pytest.raises(ValueError, match="different vocabulary"):
            check_compatible(load_meta(path), other)

    def test_stale_version_raises(self, written, tokenizer):
        path, meta, _ = written
        stale = dict(load_meta(path), version=0)
        with pytest.raises(ValueError, match="version"):
            check_compatible(stale, tokenizer)

    def test_vocab_size_mismatch_raises(self, written, tokenizer):
        path, meta, _ = written
        tampered = dict(load_meta(path), vocab_size=tokenizer.vocab_size + 1)
        with pytest.raises(ValueError, match="vocab_size"):
            check_compatible(tampered, tokenizer)

    def test_fingerprint_tracks_the_merges(self, tokenizer):
        same = BPETokenizer(dict(tokenizer.merges))
        assert vocab_fingerprint(same) == vocab_fingerprint(tokenizer)

        altered = dict(tokenizer.merges)
        altered[MERGE_START] = (32, 32)
        assert vocab_fingerprint(BPETokenizer(altered)) != vocab_fingerprint(tokenizer)

    def test_sha16_is_16_chars(self):
        assert len(sha16("anything")) == 16
        assert sha16("anything") != sha16("anything else")
