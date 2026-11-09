"""Tests for corpus discovery and reading."""

import os

import pytest

from our_llm.data.corpus import DATA_EXTENSIONS, Corpus, find_data_files, read_corpus


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


class TestFindDataFiles:
    def test_filters_by_extension(self, tmp_path):
        write(tmp_path / "a.txt", "a")
        write(tmp_path / "b.md", "b")
        write(tmp_path / "c.bin", "c")
        write(tmp_path / "no_extension", "d")
        found = find_data_files(str(tmp_path))
        assert [os.path.basename(p) for p in found] == ["a.txt", "b.md"]

    def test_single_file_is_returned_as_is(self, tmp_path):
        path = write(tmp_path / "a.txt", "a")
        assert find_data_files(str(path)) == [str(path)]

    def test_recurses_and_sorts(self, tmp_path):
        (tmp_path / "sub").mkdir()
        write(tmp_path / "z.txt", "z")
        write(tmp_path / "sub" / "a.txt", "a")
        found = find_data_files(str(tmp_path))
        assert found == sorted(found)
        assert len(found) == 2

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_data_files(str(tmp_path / "nope"))

    def test_case_insensitive_extension(self, tmp_path):
        write(tmp_path / "A.TXT", "a")
        assert len(find_data_files(str(tmp_path))) == 1


class TestReadCorpus:
    def test_documents_are_joined_with_a_blank_line(self, tmp_path):
        write(tmp_path / "a.txt", "alpha")
        write(tmp_path / "b.txt", "beta")
        corpus = read_corpus(str(tmp_path))
        assert corpus.docs == ["alpha", "beta"]
        assert corpus.text == "alpha\n\nbeta"
        assert len(corpus) == len(corpus.text)

    def test_extensionless_duplicate_is_ignored(self, tmp_path):
        """The `data/gg` regression: no extension -> not part of the corpus."""
        write(tmp_path / "dd.txt", "the real corpus")
        write(tmp_path / "gg", "the real corpus")
        corpus = read_corpus(str(tmp_path))
        assert len(corpus.docs) == 1

    def test_empty_files_are_skipped(self, tmp_path):
        write(tmp_path / "a.txt", "   \n")
        write(tmp_path / "b.txt", "text")
        corpus = read_corpus(str(tmp_path))
        assert corpus.docs == ["text"]

    def test_fingerprint_is_stable_and_content_dependent(self, tmp_path):
        write(tmp_path / "a.txt", "alpha")
        a = read_corpus(str(tmp_path))
        b = read_corpus(str(tmp_path))
        assert a.sha256 == b.sha256
        assert len(a.fingerprint) == 16

        write(tmp_path / "a.txt", "alpha!")
        c = read_corpus(str(tmp_path))
        assert c.sha256 != a.sha256

    def test_empty_directory_is_falsy(self, tmp_path):
        assert not read_corpus(str(tmp_path))
        assert read_corpus(str(tmp_path)).text == ""


class TestCorpusDataclass:
    def test_defaults(self):
        corpus = Corpus(docs=["a"], sources=["f.txt"])
        assert corpus.separator == "\n\n"
        assert corpus.text == "a"
        assert bool(corpus) is True

    def test_default_extensions_cover_txt_and_md(self):
        assert ".txt" in DATA_EXTENSIONS and ".md" in DATA_EXTENSIONS

    def test_structure_fingerprint_tracks_document_boundaries(self):
        one_doc = Corpus(docs=["alpha\n\nbeta"])
        two_docs = Corpus(docs=["alpha", "beta"])
        assert one_doc.text == two_docs.text  # identical characters...
        assert one_doc.fingerprint == two_docs.fingerprint
        assert one_doc.structure_fingerprint != two_docs.structure_fingerprint  # ...different layout

    def test_structure_fingerprint_is_order_sensitive(self):
        assert (
            Corpus(docs=["a", "b"]).structure_fingerprint
            != Corpus(docs=["b", "a"]).structure_fingerprint
        )


class TestGranularity:
    def test_paragraphs_are_split_on_blank_lines(self, tmp_path):
        write(tmp_path / "a.txt", "one\n\ntwo\n \nthree\n")
        corpus = read_corpus(str(tmp_path), granularity="paragraph")
        assert corpus.docs == ["one", "two", "three"]

    def test_paragraphs_strip_surrounding_whitespace(self, tmp_path):
        write(tmp_path / "a.txt", "  lead\ntrail  \n\n\n\nnext\n")
        assert read_corpus(str(tmp_path), granularity="paragraph").docs == ["lead\ntrail", "next"]

    def test_file_granularity_keeps_the_file_whole(self, tmp_path):
        write(tmp_path / "a.txt", "one\n\ntwo")
        assert read_corpus(str(tmp_path), granularity="file").docs == ["one\n\ntwo"]

    def test_both_granularities_agree_on_the_text_they_keep(self, tmp_path):
        write(tmp_path / "a.txt", "one\n\ntwo\n\nthree")
        as_file = read_corpus(str(tmp_path), granularity="file")
        as_paras = read_corpus(str(tmp_path), granularity="paragraph")
        assert as_file.text == as_paras.text

    def test_the_vocabulary_key_ignores_granularity(self, tmp_path):
        """The BPE cache must not be invalidated by a layout change."""
        write(tmp_path / "a.txt", "one\n\ntwo\n\nthree")
        as_file = read_corpus(str(tmp_path), granularity="file")
        as_paras = read_corpus(str(tmp_path), granularity="paragraph")
        assert as_file.files_hash == as_paras.files_hash
        # unchanged by file granularity: files_hash is just the text there
        assert as_file.files_hash == as_file.fingerprint

    def test_the_token_file_key_does_change(self, tmp_path):
        write(tmp_path / "a.txt", "one\n\ntwo\n\nthree")
        as_file = read_corpus(str(tmp_path), granularity="file")
        as_paras = read_corpus(str(tmp_path), granularity="paragraph")
        assert as_file.structure_fingerprint != as_paras.structure_fingerprint

    def test_sources_are_repeated_per_paragraph(self, tmp_path):
        write(tmp_path / "a.txt", "one\n\ntwo")
        write(tmp_path / "b.txt", "three")
        corpus = read_corpus(str(tmp_path), granularity="paragraph")
        assert corpus.docs == ["one", "two", "three"]
        assert corpus.sources == [str(tmp_path / "a.txt")] * 2 + [str(tmp_path / "b.txt")]

    def test_unknown_granularity_raises(self, tmp_path):
        write(tmp_path / "a.txt", "one")
        with pytest.raises(ValueError, match="granularity"):
            read_corpus(str(tmp_path), granularity="sentence")
