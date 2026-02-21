"""Tests for the evaluation pipeline."""

from unittest.mock import MagicMock

import pytest

from app.evaluation.evaluator import RAGEvaluator


class TestRAGEvaluator:
    """Tests for RAGEvaluator class."""

    def _make_evaluator(self, retrieve_results=None, chain_answer="Test answer."):
        """Build a RAGEvaluator with mocked retriever and chain."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = retrieve_results or [
            {"text": "Relevant text.", "metadata": {"source": "doc.pdf", "page": 1}, "score": 0.9}
        ]

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": chain_answer,
            "sources": [],
            "conversation_history": [],
        }

        return RAGEvaluator(retriever=mock_retriever, chain=mock_chain), mock_retriever, mock_chain

    def test_evaluate_retrieval_returns_metrics(self):
        """evaluate_retrieval should return correct metric keys."""
        evaluator, mock_retriever, _ = self._make_evaluator()

        queries = ["What are the payment terms?", "When does the contract expire?"]
        metrics = evaluator.evaluate_retrieval(queries)

        assert "avg_num_retrieved" in metrics
        assert "queries_evaluated" in metrics
        assert metrics["queries_evaluated"] == 2
        assert mock_retriever.retrieve.call_count == 2

    def test_evaluate_retrieval_empty_queries(self):
        """evaluate_retrieval with no queries should return zero metrics."""
        evaluator, _, _ = self._make_evaluator()
        metrics = evaluator.evaluate_retrieval([])
        assert metrics["avg_num_retrieved"] == 0.0
        assert metrics["queries_evaluated"] == 0

    def test_evaluate_retrieval_avg_calculation(self):
        """avg_num_retrieved should reflect the average docs returned."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = [
            [{"text": "a"}, {"text": "b"}],  # 2 results
            [{"text": "c"}],                 # 1 result
        ]
        mock_chain = MagicMock()
        evaluator = RAGEvaluator(retriever=mock_retriever, chain=mock_chain)

        metrics = evaluator.evaluate_retrieval(["q1", "q2"])
        assert metrics["avg_num_retrieved"] == 1.5  # (2+1)/2

    def test_evaluate_answers_returns_metrics(self):
        """evaluate_answers should return correct metric keys."""
        evaluator, _, _ = self._make_evaluator(chain_answer="This is the answer to your question.")

        qa_pairs = [
            {"question": "What is clause 3?", "expected_answer": "Clause 3 covers liability."},
            {"question": "What is the term?", "expected_answer": "The term is 2 years."},
        ]
        metrics = evaluator.evaluate_answers(qa_pairs)

        assert "avg_answer_length" in metrics
        assert "total_evaluated" in metrics
        assert "results" in metrics
        assert metrics["total_evaluated"] == 2
        assert len(metrics["results"]) == 2

    def test_evaluate_answers_empty_pairs(self):
        """evaluate_answers with no pairs should return zero metrics."""
        evaluator, _, _ = self._make_evaluator()
        metrics = evaluator.evaluate_answers([])
        assert metrics["avg_answer_length"] == 0.0
        assert metrics["total_evaluated"] == 0
        assert metrics["results"] == []

    def test_run_full_evaluation(self):
        """run_full_evaluation should combine retrieval and answer metrics."""
        evaluator, _, _ = self._make_evaluator()

        qa_pairs = [{"question": "What is clause 1?", "expected_answer": "It covers payments."}]
        result = evaluator.run_full_evaluation(qa_pairs)

        assert "retrieval_metrics" in result
        assert "answer_metrics" in result
        assert result["retrieval_metrics"]["queries_evaluated"] == 1
        assert result["answer_metrics"]["total_evaluated"] == 1
