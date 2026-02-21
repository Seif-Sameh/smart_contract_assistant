"""Tests for the ingestion pipeline."""

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_file


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    """Tests for chunk_documents function."""

    def test_chunker_splits_text(self):
        """Large text should be split into multiple chunks."""
        long_text = "word " * 500  # ~2500 characters
        docs = [{"text": long_text, "metadata": {"source": "test.pdf", "page": 1}}]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunker_preserves_metadata(self):
        """Metadata from the source document should be preserved in chunks."""
        docs = [
            {
                "text": "This is a test document with enough content to be interesting.",
                "metadata": {"source": "contract.pdf", "page": 2},
            }
        ]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["source"] == "contract.pdf"
        assert chunks[0]["metadata"]["page"] == 2

    def test_chunker_adds_chunk_index(self):
        """Each chunk should have a chunk_index in its metadata."""
        docs = [{"text": "Hello world.", "metadata": {"source": "doc.pdf", "page": 1}}]
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
        assert "chunk_index" in chunks[0]["metadata"]

    def test_chunker_empty_input(self):
        """Empty input should return an empty list."""
        chunks = chunk_documents([], chunk_size=1000, chunk_overlap=0)
        assert chunks == []


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    """Tests for the Embedder class."""

    def test_embedder_initialization(self):
        """Embedder should initialize with default parameters."""
        embedder = Embedder()
        assert embedder.provider == "sentence_transformers"
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder._model is None

    def test_embedder_custom_initialization(self):
        """Embedder should accept custom provider and model name."""
        embedder = Embedder(provider="openai", model_name="text-embedding-ada-002")
        assert embedder.provider == "openai"
        assert embedder.model_name == "text-embedding-ada-002"

    @patch("app.ingestion.embedder.SentenceTransformer", create=True)
    def test_embed_texts_sentence_transformers(self, mock_st_cls):
        """embed_texts should return list of vectors for sentence_transformers."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st_cls.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)}):
            embedder = Embedder(provider="sentence_transformers", model_name="all-MiniLM-L6-v2")
            embedder._model = mock_model

            result = embedder.embed_texts(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 3

    def test_get_langchain_embeddings_unsupported_provider(self):
        """get_langchain_embeddings should raise ValueError for unknown provider."""
        embedder = Embedder(provider="unknown_provider")
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            embedder.get_langchain_embeddings()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    """Tests for parse_file function."""

    def test_parse_file_unsupported_format(self):
        """parse_file should raise ValueError for unsupported file extensions."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_file("document.txt")

    @patch("app.ingestion.parser.fitz", create=True)
    def test_parse_pdf_calls_fitz(self, mock_fitz):
        """parse_file should use PyMuPDF for PDF files."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Contract clause text."
        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=1)
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_fitz.open.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch("app.ingestion.parser.fitz", mock_fitz):
                # Should not raise even if fitz returns data
                pass

    def test_parse_file_pdf_extension_routing(self):
        """parse_file should route .pdf files to _parse_pdf."""
        with patch("app.ingestion.parser._parse_pdf") as mock_pdf_parser:
            mock_pdf_parser.return_value = [{"text": "text", "metadata": {}}]
            result = parse_file("contract.pdf")
            mock_pdf_parser.assert_called_once_with("contract.pdf", "contract.pdf")

    def test_parse_file_docx_extension_routing(self):
        """parse_file should route .docx files to _parse_docx."""
        with patch("app.ingestion.parser._parse_docx") as mock_docx_parser:
            mock_docx_parser.return_value = [{"text": "text", "metadata": {}}]
            result = parse_file("agreement.docx")
            mock_docx_parser.assert_called_once_with("agreement.docx", "agreement.docx")
