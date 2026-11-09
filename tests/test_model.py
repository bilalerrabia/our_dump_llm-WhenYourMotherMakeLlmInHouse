"""Tests for the M3 model: layer parity, causality, sizes and shapes.

The point of these tests is *equivalence*: the hand-written LayerNorm/GELU must
match ``torch.nn``, one attention head must match the legacy single-head
``CausalAttention``, the fast sdpa path must match the manual path, and the live
parameter count must match the architecture-derived one in ``GPTConfig``.
"""

import math

import pytest
import torch
from torch import nn

from our_llm.config import GPTConfig, get_preset
from our_llm.model import (
    GELU,
    GPTModel,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    sinusoidal_positional_encoding,
)
from our_llm.utils import count_parameters
from src.attentions.causel_attention import CausalAttention


# --- layers -----------------------------------------------------------------


class TestLayerNorm:
    def test_matches_torch_layer_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8) * 3 + 1
        ours = LayerNorm(8)
        ref = nn.LayerNorm(8, eps=1e-5)
        with torch.no_grad():
            ours.scale.copy_(torch.randn(8))
            ours.shift.copy_(torch.randn(8))
            ref.weight.copy_(ours.scale)
            ref.bias.copy_(ours.shift)
        assert torch.allclose(ours(x), ref(x), atol=1e-6)

    def test_each_token_is_normalised_across_the_embedding(self):
        x = torch.randn(3, 7, 16) * 5 + 2
        out = LayerNorm(16)(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(3, 7), atol=1e-5)
        assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(3, 7), atol=1e-4)

    def test_has_exactly_two_learnable_vectors(self):
        layer = LayerNorm(16)
        assert count_parameters(layer) == 32
        assert layer.scale.shape == layer.shift.shape == (16,)


class TestGELU:
    def test_matches_torch_tanh_approximation(self):
        x = torch.linspace(-6, 6, 101)
        ref = nn.GELU(approximate="tanh")
        assert torch.allclose(GELU()(x), ref(x), atol=1e-6)

    def test_zero_maps_to_zero_and_large_positive_is_identity(self):
        out = GELU()(torch.tensor([0.0, 10.0, -1.0]))
        assert out[0].item() == 0.0
        assert out[1].item() == pytest.approx(10.0, abs=1e-5)
        # unlike ReLU, a little negative signal survives
        assert -0.2 < out[2].item() < 0.0


class TestFeedForward:
    def test_shape_is_preserved(self):
        assert FeedForward(32)(torch.randn(2, 5, 32)).shape == (2, 5, 32)

    def test_parameter_count_is_two_wide_linears(self):
        dim = 32
        assert count_parameters(FeedForward(dim)) == (dim * 4 * dim + 4 * dim) + (
            4 * dim * dim + dim
        )

    def test_drop_rate_zero_is_deterministic(self):
        ff = FeedForward(16)
        x = torch.randn(2, 4, 16)
        assert torch.equal(ff(x), ff(x))


# --- attention --------------------------------------------------------------


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_in=32, d_out=32, context_length=8, dropout=0.0, num_heads=4)
        assert mha(torch.randn(2, 6, 32)).shape == (2, 6, 32)

    def test_d_out_must_divide_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_in=32, d_out=30, context_length=8, dropout=0.0, num_heads=4)

    def test_sequence_longer_than_context_is_rejected(self):
        mha = MultiHeadAttention(d_in=8, d_out=8, context_length=4, dropout=0.0, num_heads=2)
        with pytest.raises(ValueError, match="context_length"):
            mha(torch.randn(1, 5, 8))

    def test_causal_mask_blocks_the_future(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(d_in=16, d_out=16, context_length=6, dropout=0.0, num_heads=2)
        mha.eval()
        x = torch.randn(1, 6, 16)
        changed = x.clone()
        changed[:, 4:] = torch.randn(1, 2, 16)  # only the last tokens differ
        out, out_changed = mha(x), mha(changed)
        # positions 0..3 may not depend on positions 4..5
        assert torch.allclose(out[:, :4], out_changed[:, :4], atol=1e-6)
        assert not torch.allclose(out[:, 4:], out_changed[:, 4:])

    def test_mask_is_a_non_persistent_buffer(self):
        mha = MultiHeadAttention(d_in=8, d_out=8, context_length=4, dropout=0.0, num_heads=2)
        assert "mask" in dict(mha.named_buffers())
        assert "mask" not in mha.state_dict()  # derived state, not checkpoint state
        assert mha.mask.dtype == torch.bool

    def test_one_head_matches_the_legacy_single_head_attention(self):
        torch.manual_seed(123)
        dim, context = 8, 4
        mha = MultiHeadAttention(dim, dim, context, dropout=0.0, num_heads=1, qkv_bias=True)
        legacy = CausalAttention(dim, dim, context, dropout=0.0, qkv_bias=True)
        with torch.no_grad():
            for ours, theirs in (
                (mha.W_query, legacy.W_query),
                (mha.W_key, legacy.W_key),
                (mha.W_value, legacy.W_value),
            ):
                theirs.weight.copy_(ours.weight)
                theirs.bias.copy_(ours.bias)
            mha.out_proj.weight.copy_(torch.eye(dim))  # legacy has no out_proj
            mha.out_proj.bias.zero_()

        x = torch.randn(2, context, dim)
        assert torch.allclose(mha(x), legacy(x), atol=1e-6)

    def test_sdpa_path_matches_the_manual_path(self):
        torch.manual_seed(7)
        kwargs = dict(d_in=32, d_out=32, context_length=8, dropout=0.0, num_heads=4)
        manual = MultiHeadAttention(**kwargs).eval()
        fast = MultiHeadAttention(**kwargs, use_sdpa=True).eval()
        fast.load_state_dict(manual.state_dict())
        x = torch.randn(3, 8, 32)
        assert torch.allclose(manual(x), fast(x), atol=1e-6)

    def test_qkv_bias_adds_one_bias_per_projection(self):
        dim = 16
        plain = MultiHeadAttention(dim, dim, 8, 0.0, 2)
        biased = MultiHeadAttention(dim, dim, 8, 0.0, 2, qkv_bias=True)
        assert count_parameters(biased) - count_parameters(plain) == 3 * dim

    def test_eval_mode_has_no_dropout(self):
        mha = MultiHeadAttention(d_in=16, d_out=16, context_length=4, dropout=0.5, num_heads=2)
        mha.eval()
        x = torch.randn(2, 4, 16)
        assert torch.equal(mha(x), mha(x))


# --- transformer block ------------------------------------------------------


class TestTransformerBlock:
    def test_shape_is_preserved(self):
        cfg = GPTConfig(emb_dim=32, n_heads=4, n_layers=1, context_length=8, drop_rate=0.0)
        block = TransformerBlock(cfg).eval()
        assert block(torch.randn(2, 6, 32)).shape == (2, 6, 32)

    def test_residual_means_zero_weights_are_the_identity(self):
        cfg = GPTConfig(emb_dim=16, n_heads=2, n_layers=1, context_length=4, drop_rate=0.0)
        block = TransformerBlock(cfg).eval()
        with torch.no_grad():
            block.att.out_proj.weight.zero_()
            block.att.out_proj.bias.zero_()
            block.ff.layers[2].weight.zero_()
            block.ff.layers[2].bias.zero_()
        x = torch.randn(2, 4, 16)
        assert torch.allclose(block(x), x, atol=1e-6)


# --- the whole model --------------------------------------------------------


class TestGPTModel:
    def test_forward_shape_is_batch_seq_vocab(self):
        cfg = get_preset("debug")
        model = GPTModel(cfg).eval()
        logits = model(torch.randint(0, cfg.vocab_size, (2, 16)))
        assert logits.shape == (2, 16, cfg.vocab_size)

    @pytest.mark.parametrize("preset", ["debug", "tiny"])
    def test_parameter_count_matches_the_config(self, preset):
        cfg = get_preset(preset)
        model = GPTModel(cfg)
        assert count_parameters(model) == cfg.num_parameters()

    def test_tiny_is_pinned_at_13m(self):
        model = GPTModel(get_preset("tiny"))
        assert model.num_parameters() == 13_039_872

    def test_tying_shares_one_tensor(self):
        cfg = get_preset("debug")
        model = GPTModel(cfg)
        assert model.out_head.weight is model.tok_emb.weight
        assert model.out_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()

    def test_untied_head_costs_an_extra_vocab_times_emb(self):
        cfg = get_preset("debug")
        tied = count_parameters(GPTModel(cfg))
        untied_cfg = GPTConfig.from_dict({**cfg.to_dict(), "tie_embeddings": False})
        assert count_parameters(GPTModel(untied_cfg)) - tied == cfg.vocab_size * cfg.emb_dim

    def test_sinusoidal_positions_cost_no_parameters(self):
        cfg = get_preset("debug")
        learned = count_parameters(GPTModel(cfg))
        sin_cfg = GPTConfig.from_dict({**cfg.to_dict(), "pos_emb": "sinusoidal"})
        sin_model = GPTModel(sin_cfg)
        assert count_parameters(sin_model) == sin_cfg.num_parameters()
        assert learned - count_parameters(sin_model) == cfg.context_length * cfg.emb_dim
        assert "pos_emb" not in sin_model.state_dict()  # deterministic, not stored
        assert sin_model(torch.zeros(1, 8, dtype=torch.long)).shape == (1, 8, cfg.vocab_size)

    def test_sinusoidal_encoding_is_banded_and_deterministic(self):
        pe = sinusoidal_positional_encoding(4, 8)
        assert pe.shape == (4, 8)
        assert torch.equal(pe, sinusoidal_positional_encoding(4, 8))
        assert torch.allclose(pe[0, 0::2], torch.zeros(4)) and torch.allclose(pe[0, 1::2], torch.ones(4))
        assert not torch.allclose(pe[0], pe[1])  # different positions differ

    def test_context_length_is_a_hard_limit(self):
        cfg = get_preset("debug")
        model = GPTModel(cfg)
        with pytest.raises(ValueError, match="context_length"):
            model(torch.zeros(1, cfg.context_length + 1, dtype=torch.long))

    def test_wrong_rank_is_rejected(self):
        cfg = get_preset("debug")
        with pytest.raises(ValueError, match=r"\(batch, seq\)"):
            GPTModel(cfg)(torch.zeros(4, dtype=torch.long))

    def test_init_scale(self):
        torch.manual_seed(0)
        cfg = GPTConfig(emb_dim=64, n_heads=4, n_layers=4, context_length=16, drop_rate=0.0)
        model = GPTModel(cfg)
        assert model.tok_emb.weight.std().item() == pytest.approx(0.02, abs=0.002)
        # residual projections are shrunk by 1/sqrt(2 * n_layers)
        expected = 0.02 / math.sqrt(2 * cfg.n_layers)
        assert model.trf_blocks[0].att.out_proj.weight.std().item() == pytest.approx(
            expected, rel=0.1
        )

    def test_gradients_reach_every_parameter(self):
        cfg = get_preset("debug")
        model = GPTModel(cfg)
        inputs = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), inputs.flatten()
        )
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_same_seed_same_logits(self):
        cfg = get_preset("debug")
        torch.manual_seed(1)
        a = GPTModel(cfg).eval()
        torch.manual_seed(1)
        b = GPTModel(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        assert torch.equal(a(idx), b(idx))
