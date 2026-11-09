"""Tests for the pure-Python embedding layer and sinusoidal PE."""

import math

import pytest

from src.embedding import Embedding, add_embeddings, get_shape, sinusoidal_positional_encoding


class TestGetShape:
    def test_1d(self):
        assert get_shape([1, 2, 3]) == (3,)

    def test_2d(self):
        assert get_shape([[1, 2], [3, 4]]) == (2, 2)

    def test_3d(self):
        assert get_shape([[[1]], [[2]]]) == (2, 1, 1)

    def test_empty(self):
        assert get_shape([]) == (0,)


class TestEmbedding:
    def test_1d_lookup_returns_rows(self):
        layer = Embedding(vocab_size=10, output_dim=4)
        out = layer([0, 3, 5])
        assert out == [layer.weight[0], layer.weight[3], layer.weight[5]]

    def test_2d_lookup_preserves_batch_shape(self):
        layer = Embedding(10, 4)
        out = layer([[0, 1], [2, 3]])
        assert get_shape(out) == (2, 2, 4)

    def test_1d_returns_2d(self):
        assert get_shape(Embedding(10, 4)([1, 2])) == (2, 4)

    def test_rejects_invalid_input(self):
        layer = Embedding(10, 4)
        with pytest.raises(ValueError, match="1D or 2D"):
            layer([1.5, 2.5])  # floats, not ints
        with pytest.raises(ValueError):
            layer([])  # empty list is not 1D nor 2D per implementation
        with pytest.raises(ValueError):
            layer("abc")  # not a list at all

    def test_weights_are_deterministic_given_seed(self):
        import random

        random.seed(123)
        a = Embedding(5, 3).weight
        random.seed(123)
        b = Embedding(5, 3).weight
        assert a == b


class TestAddEmbeddings:
    def test_broadcast_add_shapes(self):
        token = [[[1.0] * 4] * 3]  # (1, 3, 4)
        pos = [[10.0] * 4] * 3     # (3, 4)
        out = add_embeddings(token, pos)
        assert get_shape(out) == (1, 3, 4)
        assert out[0][0] == [11.0] * 4

    def test_broadcast_add_values(self):
        token = [[[1.0, 2.0], [3.0, 4.0]]]  # (1, 2, 2)
        pos = [[0.5, 0.5], [1.0, 1.0]]      # (2, 2)
        out = add_embeddings(token, pos)
        assert out == [[[1.5, 2.5], [4.0, 5.0]]]

    def test_does_not_mutate_inputs(self):
        token = [[[1.0, 1.0]]]
        pos = [[1.0, 1.0]]
        add_embeddings(token, pos)
        assert token == [[[1.0, 1.0]]]  # unchanged

    def test_different_batch_entries_get_same_pos(self):
        # every batch element must add the SAME positional row per sequence index
        token = [[[1.0]], [[2.0]]]              # (2, 1, 1)
        pos = [[10.0]]                          # (1, 1)
        out = add_embeddings(token, pos)
        assert out == [[[11.0]], [[12.0]]]


class TestSinusoidalPE:
    def test_shape(self):
        assert get_shape(sinusoidal_positional_encoding(6, 8)) == (6, 8)

    def test_position_zero_is_alternating_0_1(self):
        pe = sinusoidal_positional_encoding(1, 8)
        assert pe[0] == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_even_dims_are_sin_odd_are_cos(self):
        pe = sinusoidal_positional_encoding(2, 4)[1]  # row for position 1
        for i in range(0, 4, 2):
            rate = 10000 ** (-i / 4)
            assert math.isclose(pe[i], math.sin(1 * rate))
            assert math.isclose(pe[i + 1], math.cos(1 * rate))

    def test_values_bounded_in_minus_one_one(self):
        pe = sinusoidal_positional_encoding(50, 16)
        assert all(-1 <= v <= 1 for row in pe for v in row)

    def test_unique_per_position(self):
        pe = sinusoidal_positional_encoding(20, 8)
        rows = [tuple(r) for r in pe]
        assert len(set(rows)) == 20  # all positions distinct

    def test_base_parameter_changes_rates(self):
        # with base=1.0 all rates are 1 -> sin(pos), cos(pos) pattern
        pe = sinusoidal_positional_encoding(2, 4, base=1.0)
        assert math.isclose(pe[1][0], math.sin(1.0))
        assert math.isclose(pe[1][1], math.cos(1.0))
        # base=10000 default gives much smaller angle for dim 2-3
        pe2 = sinusoidal_positional_encoding(2, 4)[1]
        assert not math.isclose(pe2[0], pe2[2])