"""Tests for DatasetPreparation and create_dataloader."""

import random

import pytest

from src.data_loader import DatasetPreparation, create_dataloader


class TestDatasetPreparation:
    def test_window_content(self):
        ds = DatasetPreparation([0, 1, 2, 3, 4, 5], context_window=4, stride=4)
        # range(0, 6-4, 4) -> single window starting at 0
        assert ds.input_ids == [[0, 1, 2, 3]]
        assert ds[0] == ([0, 1, 2, 3], [1, 2, 3, 4])

    def test_number_of_windows(self):
        ds = DatasetPreparation(list(range(10)), context_window=4, stride=2)
        # range(0, 10-4, 2) -> starts at 0,2,4 -> 3 windows
        assert len(ds) == 3

    def test_stride_shifts_the_window(self):
        ds = DatasetPreparation(list(range(10)), context_window=3, stride=3)
        assert ds[1] == ([3, 4, 5], [4, 5, 6])

    def test_input_is_target_shifted_by_one(self):
        ids = list(range(20))
        ds = DatasetPreparation(ids, context_window=4, stride=2)
        for inp, tgt in ds:
            assert tgt == inp[1:] + [inp[-1] + 1]

    def test_too_short_input(self):
        assert len(DatasetPreparation([1, 2], context_window=4, stride=1)) == 0

    def test_exact_length(self):
        # range(0, 4-4, 1) is empty -> zero windows
        assert len(DatasetPreparation([1, 2, 3, 4], context_window=4, stride=1)) == 0


class TestCreateDataloader:
    def test_batch_sizes(self):
        ds = DatasetPreparation(list(range(20)), context_window=4, stride=1)
        batches = list(create_dataloader(ds, batch_size=3, shuffle=False, drop_last=True))
        assert all(len(inp) == 3 for inp, _ in batches)

    def test_drop_last_true_discards_tail(self):
        ds = DatasetPreparation(list(range(20)), context_window=4, stride=1)
        batches = list(create_dataloader(ds, batch_size=3, shuffle=False, drop_last=True))
        # 16 samples -> 5 full batches, last 1 sample dropped
        assert len(batches) == 5

    def test_drop_last_false_keeps_tail(self):
        ds = DatasetPreparation(list(range(20)), context_window=4, stride=1)
        batches = list(create_dataloader(ds, batch_size=3, shuffle=False, drop_last=False))
        assert [len(inp) for inp, _ in batches] == [3, 3, 3, 3, 3, 1]

    def test_no_shuffle_preserves_order(self):
        ids = list(range(12))
        ds = DatasetPreparation(ids, context_window=4, stride=4)
        (inputs, targets), *_ = create_dataloader(ds, batch_size=2, shuffle=False)
        assert inputs == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert targets == [[1, 2, 3, 4], [5, 6, 7, 8]]

    def test_seeded_shuffle_is_reproducible(self):
        ds = DatasetPreparation(list(range(20)), context_window=4, stride=2)
        random.seed(42)
        a = [b[0] for b in create_dataloader(ds, batch_size=2, shuffle=True)]
        random.seed(42)
        b = [b[0] for b in create_dataloader(ds, batch_size=2, shuffle=True)]
        assert a == b

    def test_empty_dataset(self):
        ds = DatasetPreparation([], context_window=4, stride=1)
        assert list(create_dataloader(ds, batch_size=2, shuffle=False)) == []