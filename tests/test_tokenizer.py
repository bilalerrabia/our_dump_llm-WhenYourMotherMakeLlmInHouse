"""Tests for the BPE Tokenizer."""

import json

import pytest

from src.tokenizer import Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


class TestRoundTrip:
    def test_ascii_round_trip(self, tok):
        text = "hello world, this is a test!"
        assert tok.decoder(tok.encode(text)) == text

    def test_utf8_round_trip(self, tok):
        text = "héllo wörld — 你好 مرحبا 🚀"
        assert tok.decoder(tok.encode(text)) == text

    def test_empty_string(self, tok):
        assert tok.encode("") == []
        assert tok.decoder([]) == ""

    def test_ids_stay_in_vocab(self, tok):
        # Untrained tokenizer: every id must be a raw byte id (0-255)
        text = "the quick brown fox jumps over the lazy dog"
        for tid in tok.encode(text):
            assert 0 <= tid <= 255

    def test_after_training_round_trip(self, tok):
        text = "aaabdaaabac" * 10
        tok.train(text, num_merges=20)
        assert tok.decoder(tok.encode(text)) == text


class TestTrain:
    def test_vocab_grows_by_num_merges(self, tok):
        tok.train("aaaa bbbb aaaa bbbb", num_merges=5)
        assert len(tok.map_replace) == 5

    def test_first_merge_is_most_frequent_pair(self, tok):
        tok.train("aaaa", num_merges=1)
        # 'a' appears 4x -> (97,97) is the only/most frequent pair
        assert tok.map_replace[256] == (97, 97)

    def test_train_is_deterministic(self):
        t1, t2 = Tokenizer(), Tokenizer()
        text = "the cat sat on the mat. the cat sat."
        r1 = t1.train(text, num_merges=15)
        r2 = t2.train(text, num_merges=15)
        assert r1 == r2
        assert t1.map_replace == t2.map_replace

    def test_train_returns_encoded_ids(self, tok):
        text = "hello"
        assert tok.train(text, num_merges=0) == [104, 101, 108, 108, 111]


class TestEncodeDecode:
    def test_encode_matches_manual_merge_application(self, tok):
        # encode() must equal flattening the words after applying every
        # learned merge in the order it was learned
        text = "abcabc abcabc"
        tok.train(text, num_merges=5)
        words = tok.encoder(text)
        for new_id, pair in sorted(tok.map_replace.items()):
            words = tok.replace(words, pair, new_id)
        assert tok.encode(text) == [t for w in words for t in w]

    def test_unknown_text_uses_base_bytes(self, tok):
        # Never-trained tokenizer: every id must be a raw byte id (0-255)
        ids = tok.encode("xyz 123 !?")
        assert all(0 <= tid <= 255 for tid in ids)

    def test_decoder_raises_on_missing_merge(self, tok):
        # id 300 has no merge rule
        with pytest.raises(ValueError, match="Missing merge for token 300"):
            tok.decoder([300])


class TestHelpers:
    def test_pairs_counter(self, tok):
        assert tok.pairs_counter([[1, 1, 2, 2]]) == {(1, 1): 1, (1, 2): 1, (2, 2): 1}

    def test_pairs_counter_respects_word_boundaries(self, tok):
        # 'ab' and 'cd' are separate words: (98,99) must NOT be counted
        assert tok.pairs_counter([[97, 98], [99, 100]]) == {
            (97, 98): 1,
            (99, 100): 1,
        }

    def test_replace(self, tok):
        words = [[1, 2, 1, 2, 3]]
        assert tok.replace(words, (1, 2), 9) == [[9, 9, 3]]

    def test_replace_no_match(self, tok):
        assert tok.replace([[1, 3, 2]], (1, 2), 9) == [[1, 3, 2]]

    def test_replace_overlapping_pairs_left_to_right(self, tok):
        # "aaa" with pair (97,97) -> one merge consumed, third byte untouched
        assert tok.replace([[97, 97, 97]], (97, 97), 256) == [[256, 97]]


class TestPersistence:
    def test_save_then_load(self, tok, tmp_path):
        tok.train("the cat sat on the mat", num_merges=10)
        vocab_path = tmp_path / "vocab.json"
        tok.save_vocab(str(vocab_path))

        merges_path = tmp_path / "vocab_merges.json"
        assert merges_path.exists()  # derived filename

        tok2 = Tokenizer()
        tok2.load_vocab(str(vocab_path))
        assert tok2.map_replace == tok.map_replace

    def test_save_vocab_writes_readable_json(self, tok, tmp_path):
        tok.train("hello", num_merges=2)
        vocab_path = tmp_path / "vocab.json"
        tok.save_vocab(str(vocab_path))

        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        # base byte entries exist, e.g. 'a' -> 97
        assert data["a"] == 97

    def test_load_vocab_missing_file_raises(self, tok, tmp_path):
        with pytest.raises(FileNotFoundError):
            tok.load_vocab(str(tmp_path / "nope.json"))


class TestPreTokenizer:
    def test_encoder_splits_words_and_keeps_spaces(self, tok):
        # ' cat' -> space is part of the token, GPT-2 style
        assert tok.encoder("a cat") == [[97], [32, 99, 97, 116]]

    def test_encoder_empty(self, tok):
        assert tok.encoder("") == []