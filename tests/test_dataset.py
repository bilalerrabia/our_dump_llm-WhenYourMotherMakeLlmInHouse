"""Tests for the torch sliding-window dataset, the split, and the loaders."""

import os

import numpy as np
import pytest
import torch

from our_llm.config import DataConfig
from our_llm.data.corpus import Corpus
from our_llm.data.dataset import (
    GPTDataset,
    build_loaders,
    gather,
    make_loader,
    prepare,
    split_documents,
)
from our_llm.data.tokens import encode_corpus, open_tokens, token_file_path
from our_llm.tokenizer import BPETokenizer, EOT_TOKEN

IDS = list(range(100, 130))  # 30 distinct "tokens"

DOCS = [
    "The monster opened its eyes and saw the night.",
    "It was on a dreary night of November that I beheld my work.",
    "Learn from me, if not by my precepts, at least by my example.",
    "Nothing is so painful to the human mind as a great and sudden change.",
]


# --- sliding window ---------------------------------------------------------


class TestGPTDataset:
    def test_targets_are_inputs_shifted_by_one(self):
        dataset = GPTDataset(IDS, context_length=4)
        for index in range(len(dataset)):
            inputs, targets = dataset[index]
            assert torch.equal(inputs[1:], targets[:-1])

    def test_windows_are_contiguous_slices(self):
        dataset = GPTDataset(IDS, context_length=4, stride=4)
        inputs, targets = dataset[0]
        assert inputs.tolist() == IDS[0:4]
        assert targets.tolist() == IDS[1:5]

        inputs, targets = dataset[2]
        assert inputs.tolist() == IDS[8:12]
        assert targets.tolist() == IDS[9:13]

    def test_stride_one_yields_every_position(self):
        dataset = GPTDataset(IDS, context_length=4, stride=1)
        assert len(dataset) == len(IDS) - 4
        assert dataset[1][0].tolist() == IDS[1:5]

    def test_default_stride_is_the_context_length(self):
        assert GPTDataset(IDS, context_length=5).stride == 5

    def test_sample_count(self):
        # 30 ids, window 4 -> starts 0..25 with stride 1 -> 26 samples
        assert len(GPTDataset(IDS, 4, 1)) == 26
        # stride 4 -> floor((30-4-1)/4)+1 = 7
        assert len(GPTDataset(IDS, 4, 4)) == 7
        # stride 5 (== context) -> 6 full non-overlapping windows of 5 ids
        assert len(GPTDataset(IDS, 5, 5)) == 5

    def test_short_input_has_no_samples(self):
        assert len(GPTDataset([1, 2, 3], context_length=4)) == 0
        assert len(GPTDataset([], context_length=4)) == 0

    def test_exactly_context_plus_one_gives_one_sample(self):
        dataset = GPTDataset(list(range(5)), context_length=4)
        assert len(dataset) == 1
        inputs, targets = dataset[0]
        assert inputs.tolist() == [0, 1, 2, 3]
        assert targets.tolist() == [1, 2, 3, 4]

    def test_tensor_contract(self):
        inputs, targets = GPTDataset(IDS, context_length=6)[0]
        for tensor in (inputs, targets):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.long
            assert tensor.shape == (6,)

    def test_out_of_range_index(self):
        dataset = GPTDataset(IDS, context_length=4)
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            GPTDataset(IDS, context_length=0)
        with pytest.raises(ValueError):
            GPTDataset(IDS, context_length=4, stride=0)

    def test_works_on_a_memmap(self, tmp_path):
        array = np.arange(50, dtype=np.uint16)
        path = str(tmp_path / "ids.bin")
        array.tofile(path)
        mapped = np.memmap(path, dtype=np.uint16, mode="r", shape=(50,))
        dataset = GPTDataset(mapped, context_length=4)
        inputs, targets = dataset[0]
        assert inputs.dtype == torch.long  # widened from uint16
        assert inputs.tolist() == [0, 1, 2, 3]


# --- splitting --------------------------------------------------------------


class TestSplitDocuments:
    def test_disjoint_and_complete(self):
        train, val = split_documents(10, val_ratio=0.2, seed=1)
        assert sorted(train + val) == list(range(10))
        assert not set(train) & set(val)

    def test_ratio_is_respected(self):
        train, val = split_documents(10, val_ratio=0.1)
        assert len(val) == 1 and len(train) == 9
        train, val = split_documents(10, val_ratio=0.3)
        assert len(val) == 3 and len(train) == 7

    def test_deterministic_for_a_seed(self):
        assert split_documents(20, 0.25, seed=7) == split_documents(20, 0.25, seed=7)

    def test_different_seeds_pick_different_documents(self):
        a = split_documents(50, 0.2, seed=1)
        b = split_documents(50, 0.2, seed=2)
        assert a != b

    def test_single_document_all_train(self):
        assert split_documents(1, 0.1) == ([0], [])

    def test_zero_val_ratio(self):
        train, val = split_documents(5, 0.0)
        assert val == [] and len(train) == 5

    def test_never_empties_the_training_set(self):
        train, val = split_documents(2, 0.9)
        assert train and len(val) == 1

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            split_documents(5, 1.5)
        with pytest.raises(ValueError):
            split_documents(5, -0.1)


class TestGather:
    def test_concatenates_in_the_order_given(self):
        ids = np.arange(10, dtype=np.uint16)
        spans = [(0, 3), (3, 6), (6, 10)]
        assert gather(ids, spans, [2, 0]).tolist() == [6, 7, 8, 9, 0, 1, 2]

    def test_empty_selection(self):
        ids = np.arange(4, dtype=np.uint16)
        assert len(gather(ids, [(0, 4)], [])) == 0

    def test_preserves_dtype(self):
        ids = np.arange(4, dtype=np.uint16)
        assert gather(ids, [(0, 4)], [0]).dtype == np.uint16


# --- loaders ----------------------------------------------------------------


class TestLoaders:
    def test_batch_shapes(self):
        loader = make_loader(IDS * 4, context_length=4, batch_size=3, stride=4)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (3, 4)
        assert targets.shape == (3, 4)
        assert inputs.dtype == torch.long

    def test_drop_last(self):
        ids = IDS * 2  # 60 ids -> 74 samples at stride 1
        kept = len(list(make_loader(ids, 4, 8, stride=1, drop_last=True)))
        allb = len(list(make_loader(ids, 4, 8, stride=1, drop_last=False)))
        assert kept == allb or kept == allb - 1
        total = sum(x.shape[0] for x, _ in make_loader(ids, 4, 8, stride=1, drop_last=True))
        assert total % 8 == 0

    def test_shuffle_is_seeded(self):
        def first_batch(shuffle: bool):
            loader = make_loader(
                IDS * 4, context_length=4, batch_size=2, shuffle=shuffle, seed=5
            )
            return next(iter(loader))[0].tolist()

        assert first_batch(True) == first_batch(True)
        assert first_batch(False) == [IDS[0:4], IDS[4:8]]

    def test_empty_stream_raises_with_a_reason(self):
        with pytest.raises(ValueError, match="no samples"):
            make_loader([1, 2, 3], context_length=8, batch_size=2)


# --- prepare (end to end on a tiny corpus) ----------------------------------


@pytest.fixture
def tiny_config(tmp_path) -> DataConfig:
    corpus_dir = tmp_path / "data"
    corpus_dir.mkdir()
    (corpus_dir / "one.txt").write_text(DOCS[0] * 5, encoding="utf-8")
    (corpus_dir / "two.txt").write_text(DOCS[1] * 5, encoding="utf-8")
    (corpus_dir / "three.txt").write_text(DOCS[2] * 5, encoding="utf-8")
    return DataConfig(
        corpus_path=str(corpus_dir),
        cache_dir=str(tmp_path / "cache"),
        processed_dir=str(tmp_path / "processed"),
        num_merges=80,
        val_ratio=0.34,
        batch_size=2,
        seed=0,
    )


class TestPrepare:
    def test_writes_a_token_file_and_reads_it_back(self, tiny_config):
        prepared = prepare(tiny_config)
        assert prepared.path.endswith(".bin")
        assert prepared.meta["n_documents"] == 3

        ids = open_tokens(prepared.path)
        assert ids.dtype == np.dtype(prepared.meta["dtype"])
        assert len(ids) == prepared.meta["n_tokens"]
        assert prepared.n_train_tokens + prepared.n_val_tokens == len(ids)

    def test_second_call_reuses_the_cache(self, tiny_config):
        first = prepare(tiny_config)
        mtime = os.path.getmtime(first.path)
        second = prepare(tiny_config)
        assert os.path.getmtime(first.path) == mtime
        assert second.meta["n_tokens"] == first.meta["n_tokens"]

    def test_force_re_encodes(self, tiny_config):
        first = prepare(tiny_config)
        second = prepare(tiny_config, force=True)
        assert second.path == first.path
        assert np.array_equal(second.train_ids, first.train_ids)

    def test_split_keeps_documents_whole(self, tiny_config):
        prepared = prepare(tiny_config)
        assert not set(prepared.train_docs) & set(prepared.val_docs)
        assert sorted(prepared.train_docs + prepared.val_docs) == [0, 1, 2]
        assert prepared.train_docs and prepared.val_docs  # 2/1 split

    def test_no_document_appears_on_both_sides(self, tiny_config):
        """Leakage guard: the train and val token ranges must not intersect."""
        prepared = prepare(tiny_config)
        spans = [(int(s), int(e)) for s, e in prepared.meta["doc_spans"]]
        train_ranges = {spans[i] for i in prepared.train_docs}
        val_ranges = {spans[i] for i in prepared.val_docs}

        assert train_ranges.isdisjoint(val_ranges)
        assert prepared.n_train_tokens == sum(e - s for s, e in train_ranges)
        assert prepared.n_val_tokens == sum(e - s for s, e in val_ranges)

        # ...and the arrays really are exactly those ranges, in order.
        ids = open_tokens(prepared.path)
        assert np.array_equal(
            prepared.val_ids,
            np.concatenate([ids[s:e] for s, e in (spans[i] for i in prepared.val_docs)]),
        )

    def test_document_split_avoids_what_a_token_split_would_leak(self, tiny_config):
        """Why the split is by document: a percentage cut lands mid-document."""
        prepared = prepare(tiny_config)
        docs = prepared.corpus.docs

        val_texts = [docs[i] for i in prepared.val_docs]
        train_text = "\n\n".join(docs[i] for i in prepared.train_docs)
        assert not any(text in train_text for text in val_texts)

        joined = "\n\n".join(docs)
        cut = int(len(joined) * 0.9)
        assert joined[cut - 1 : cut + 1] != "\n\n"  # the naive cut is inside text

    def test_token_file_name_depends_on_vocab_size(self, tiny_config):
        prepared = prepare(tiny_config)
        expected = token_file_path(
            tiny_config.processed_dir,
            prepared.corpus.structure_fingerprint,
            prepared.tokenizer.num_merges,
        )
        assert prepared.path == expected

    def test_encoded_documents_decode_back_to_the_corpus(self, tiny_config):
        prepared = prepare(tiny_config)
        ids = open_tokens(prepared.path)
        spans = [(int(s), int(e)) for s, e in prepared.meta["doc_spans"]]
        eot = prepared.tokenizer.special_ids[EOT_TOKEN]
        for index, (start, end) in enumerate(spans):
            text = prepared.tokenizer.decode(ids[start:end].tolist())
            assert text.endswith(EOT_TOKEN)
            assert ids[end - 1] == eot
            assert text.removesuffix(EOT_TOKEN) == prepared.corpus.docs[index]

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare(DataConfig(corpus_path=str(tmp_path / "nope")))


class TestBuildLoaders:
    def test_train_and_val_loaders(self, tiny_config):
        train, val, prepared = build_loaders(tiny_config, context_length=8)
        assert len(train) > 0
        assert val is not None and len(val) > 0

        inputs, targets = next(iter(train))
        assert inputs.shape[1] == 8
        assert targets.shape == inputs.shape
        assert inputs.min() >= 0 and inputs.max() < prepared.vocab_size

    def test_validation_loader_is_optional(self, tiny_config):
        only_one = DataConfig(
            corpus_path=tiny_config.corpus_path,
            cache_dir=tiny_config.cache_dir,
            processed_dir=tiny_config.processed_dir,
            num_merges=tiny_config.num_merges,
            val_ratio=0.0,
        )
        _, val, prepared = build_loaders(only_one, context_length=8)
        assert val is None
        assert prepared.val_docs == []

    def test_val_loader_is_deterministic(self, tiny_config):
        _, val_a, _ = build_loaders(tiny_config, context_length=8)
        _, val_b, _ = build_loaders(tiny_config, context_length=8)
        assert val_a is not None and val_b is not None
        a = torch.cat([x for x, _ in val_a])
        b = torch.cat([x for x, _ in val_b])
        assert torch.equal(a, b)


class TestCorpusDocBoundaries:
    def test_windows_do_not_mix_documents_when_each_doc_is_long_enough(self):
        corpus = Corpus(docs=["aaaa bbbb cccc dddd", "eeee ffff gggg hhhh"])
        tokenizer = BPETokenizer().train(
            " ".join(corpus.docs) * 5, 20, show_progress=False
        )
        encoded = encode_corpus(tokenizer, corpus)
        eot = tokenizer.special_ids[EOT_TOKEN]
        # Each document ends with EOT, so a window that crosses the boundary has
        # exactly one EOT in it -- never two documents' text without a marker.
        assert encoded.ids.tolist().count(eot) == len(corpus.docs)
