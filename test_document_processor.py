"""Tests for DocumentProcessor."""

import os
import tempfile

import pytest

from src.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Provide a fresh DocumentProcessor instance."""
    return DocumentProcessor()


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------

class TestCleanText:
    def test_collapses_multiple_spaces(self, processor):
        assert processor._clean_text("hello   world") == "hello world"

    def test_collapses_tabs(self, processor):
        assert processor._clean_text("hello\t\tworld") == "hello world"

    def test_collapses_excessive_newlines(self, processor):
        result = processor._clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_leading_trailing_whitespace(self, processor):
        assert processor._clean_text("  hello  ") == "hello"

    def test_preserves_double_newlines(self, processor):
        result = processor._clean_text("a\n\nb")
        assert result == "a\n\nb"


# ------------------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------------------

class TestGetMetadata:
    def test_word_count(self, processor):
        meta = processor.get_metadata("Hello world foo bar")
        assert meta["word_count"] == 4

    def test_char_count(self, processor):
        meta = processor.get_metadata("abc")
        assert meta["char_count"] == 3

    def test_sentence_count(self, processor):
        meta = processor.get_metadata("Hello. World. Foo!")
        assert meta["sentence_count"] >= 2

    def test_paragraph_count(self, processor):
        meta = processor.get_metadata("Para one.\n\nPara two.\n\nPara three.")
        assert meta["paragraph_count"] == 3

    def test_empty_text_metadata(self, processor):
        meta = processor.get_metadata("")
        assert meta["word_count"] == 0
        assert meta["char_count"] == 0


# ------------------------------------------------------------------
# Text chunking
# ------------------------------------------------------------------

class TestChunkText:
    def test_short_text_single_chunk(self, processor):
        text = "Short sentence."
        chunks = processor.chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].strip() == text

    def test_long_text_produces_multiple_chunks(self, processor):
        text = "Word. " * 1500  # ~9000 chars
        chunks = processor.chunk_text(text, chunk_size=1000)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, processor):
        text = "Hello world. " * 500
        chunks = processor.chunk_text(text, chunk_size=500)
        # First chunk should be ≤ chunk_size (roughly, may exceed slightly
        # when the last added sentence pushes it over).
        assert len(chunks[0]) < 800  # generous upper bound

    def test_empty_text(self, processor):
        chunks = processor.chunk_text("")
        # Should return a single empty-ish chunk or empty list.
        assert isinstance(chunks, list)


# ------------------------------------------------------------------
# File-based extraction
# ------------------------------------------------------------------

class TestExtractText:
    def test_extract_txt(self, processor):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write("Hello from a text file.\nSecond line.")
            tmp_path = tmp.name
        try:
            text = processor.extract_text(tmp_path)
            assert "Hello from a text file." in text
            assert "Second line." in text
        finally:
            os.unlink(tmp_path)

    def test_extract_unsupported_raises(self, processor):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="w"
        ) as tmp:
            tmp.write("a,b,c")
            tmp_path = tmp.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                processor.extract_text(tmp_path)
        finally:
            os.unlink(tmp_path)


# ------------------------------------------------------------------
# Bytes-based extraction (Streamlit uploads)
# ------------------------------------------------------------------

class TestExtractTextFromBytes:
    def test_extract_txt_bytes(self, processor):
        content = b"Hello from bytes."
        text = processor.extract_text_from_bytes(content, "txt")
        assert "Hello from bytes." in text

    def test_extract_unsupported_bytes_raises(self, processor):
        with pytest.raises(ValueError, match="Unsupported"):
            processor.extract_text_from_bytes(b"data", "csv")
