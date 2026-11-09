"""Tests for the model presets and the architecture-derived parameter count."""

import pytest

from our_llm.config import PRESETS, GPTConfig, get_preset
from our_llm.tokenizer.special import MERGE_START


class TestGPTConfig:
    def test_default_vocab_matches_repo_tokenizer(self):
        # MERGE_START + 14,970 merges from the cached vocabulary.
        assert GPTConfig().vocab_size == 15_228

    def test_head_dim(self):
        cfg = GPTConfig(emb_dim=384, n_heads=6)
        assert cfg.head_dim == 64

    def test_emb_dim_must_divide_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            GPTConfig(emb_dim=100, n_heads=3)

    def test_pos_emb_is_validated(self):
        with pytest.raises(ValueError, match="pos_emb"):
            GPTConfig(pos_emb="rope")

    def test_is_frozen(self):
        with pytest.raises(Exception):
            GPTConfig().emb_dim = 128  # type: ignore[misc]

    def test_with_vocab_only_changes_vocab(self):
        cfg = GPTConfig()
        new = cfg.with_vocab(50_257)
        assert new.vocab_size == 50_257
        assert new.emb_dim == cfg.emb_dim
        assert new.n_layers == cfg.n_layers

    def test_dict_round_trip(self):
        cfg = get_preset("tiny")
        assert GPTConfig.from_dict(cfg.to_dict()) == cfg

    def test_from_dict_ignores_unknown_keys(self):
        cfg = GPTConfig.from_dict({"emb_dim": 64, "n_heads": 4, "whatever": 1})
        assert cfg.emb_dim == 64


class TestParameterCount:
    def test_tiny_is_the_13m_target(self):
        # Pinned by hand: tok 15,228*384 + pos 256*384 + 4 blocks * 1,773,312
        # + final norm 768, embeddings tied.
        assert get_preset("tiny").num_parameters() == 13_039_872

    def test_debug_preset(self):
        assert get_preset("debug").num_parameters() == 1_078_400

    def test_gpt2_small_is_124m(self):
        assert get_preset("gpt2-small").num_parameters() == 124_412_160

    def test_tying_saves_vocab_times_emb(self):
        config = dict(vocab_size=MERGE_START + 14_970, context_length=256,
                      emb_dim=384, n_heads=6, n_layers=4)
        tied = GPTConfig(**config, tie_embeddings=True)
        untied = GPTConfig(**config, tie_embeddings=False)
        assert untied.num_parameters() - tied.num_parameters() == 15_228 * 384

    def test_more_layers_adds_one_block(self):
        a = GPTConfig(n_layers=2)
        b = GPTConfig(n_layers=3)
        per_block = b.num_parameters() - a.num_parameters()
        dim = a.emb_dim
        # QKV 3*dim^2 + out_proj (dim^2 + dim) + FFN (8*dim^2 + 5*dim) + 2 norms (4*dim)
        assert per_block == 12 * dim * dim + 10 * dim


class TestPresets:
    def test_all_presets_build(self):
        for name in PRESETS:
            cfg = get_preset(name)
            assert cfg.vocab_size > 0 and cfg.context_length > 0

    def test_unknown_preset_raises_with_hint(self):
        with pytest.raises(KeyError, match="Known"):
            get_preset("nonexistent")

    def test_debug_is_the_cheapest(self):
        params = {name: cfg.num_parameters() for name, cfg in PRESETS.items()}
        assert params["debug"] < params["tiny"] < params["gpt2-small"]
