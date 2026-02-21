"""Tests for the retrieval pipeline."""

from unittest.mock import MagicMock

import pytest

from app.retrieval.retriever import DocumentRetriever


class TestDocumentRetriever:
    """Tests for DocumentRetriever class."""

    def _make_retriever(self, search_results=None):
        """Helper to build a DocumentRetriever with a mocked VectorStoreManager."""
        mock_vsm = MagicMock()
        mock_vsm.similarity_search.return_value = search_results or []
        return DocumentRetriever(vector_store_manager=mock_vsm, k=5), mock_vsm

    def test_retriever_returns_results(self):
        """retrieve() should return results from the vector store."""
        expected = [
            {"text": "Clause 1 text.", "metadata": {"source": "contract.pdf", "page": 1}, "score": 0.9},
            {"text": "Clause 2 text.", "metadata": {"source": "contract.pdf", "page": 2}, "score": 0.85},
        ]
        retriever, mock_vsm = self._make_retriever(expected)

        results = retriever.retrieve("What are the payment terms?")

        mock_vsm.similarity_search.assert_called_once_with("What are the payment terms?", k=5)
        assert len(results) == 2
        assert results[0]["text"] == "Clause 1 text."

    def test_retriever_empty_store(self):
        """retrieve() should return empty list when store has no documents."""
        retriever, _ = self._make_retriever([])
        results = retriever.retrieve("any query")
        assert results == []

    def test_format_context_formats_properly(self):
        """format_context() should produce a well-structured context string."""
        retriever, _ = self._make_retriever()

        results = [
            {"text": "Payment is due in 30 days.", "metadata": {"source": "contract.pdf", "page": 3}, "score": 0.9},
            {"text": "Termination requires 60 days notice.", "metadata": {"source": "contract.pdf", "page": 7}, "score": 0.8},
        ]

        context = retriever.format_context(results)

        assert "Source: contract.pdf" in context
        assert "Page: 3" in context
        assert "Payment is due in 30 days." in context
        assert "Page: 7" in context
        assert "Termination requires 60 days notice." in context

    def test_format_context_empty_results(self):
        """format_context() should handle empty results gracefully."""
        retriever, _ = self._make_retriever()
        context = retriever.format_context([])
        assert "No relevant context found." in context

    def test_retriever_uses_configured_k(self):
        """retrieve() should pass the configured k to similarity_search."""
        retriever, mock_vsm = self._make_retriever()
        retriever.k = 3
        retriever.retrieve("query")
        mock_vsm.similarity_search.assert_called_once_with("query", k=3)
