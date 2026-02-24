"""Tests for the evaluation pipeline."""

from unittest.mock import MagicMock, patch

import pandas as pd
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

    def test_run_full_evaluation_without_ragas(self):
        """run_full_evaluation without include_ragas should not contain ragas_metrics."""
        evaluator, _, _ = self._make_evaluator()
        qa_pairs = [{"question": "What is clause 1?", "expected_answer": "It covers payments."}]
        result = evaluator.run_full_evaluation(qa_pairs, include_ragas=False)
        assert "ragas_metrics" not in result

    # ------------------------------------------------------------------ #
    # RAGAS tests                                                          #
    # ------------------------------------------------------------------ #

    def _make_ragas_mock(self, metric_names=None):
        """Return a mock RAGAS evaluate result with the given metric columns."""
        if metric_names is None:
            metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
        data = {"user_input": ["q1"], "response": ["ans1"]}
        for name in metric_names:
            data[name] = [0.8]
        return MagicMock(to_pandas=MagicMock(return_value=pd.DataFrame(data)))

    def _make_ragas_metric_mocks(self, names=None):
        """Build lightweight metric mock objects with a .name attribute."""
        if names is None:
            names = ["faithfulness", "answer_relevancy", "context_precision"]
        metrics = []
        for name in names:
            m = MagicMock()
            m.name = name
            metrics.append(m)
        return metrics

    @patch("app.evaluation.evaluator.RAGEvaluator.evaluate_with_ragas")
    def test_run_full_evaluation_with_ragas(self, mock_ragas):
        """run_full_evaluation with include_ragas=True should include ragas_metrics."""
        mock_ragas.return_value = {
            "scores": {"faithfulness": 0.9},
            "results": [],
            "total_evaluated": 1,
        }
        evaluator, _, _ = self._make_evaluator()
        qa_pairs = [{"question": "What is clause 1?", "expected_answer": "It covers payments."}]

        result = evaluator.run_full_evaluation(qa_pairs, include_ragas=True)

        assert "ragas_metrics" in result
        mock_ragas.assert_called_once()

    def test_evaluate_with_ragas_uses_provided_contexts(self):
        """evaluate_with_ragas should use pre-fetched contexts when supplied."""
        evaluator, mock_retriever, mock_chain = self._make_evaluator(
            chain_answer="Payment is due in 30 days."
        )
        metric_mocks = self._make_ragas_metric_mocks()
        ragas_result = self._make_ragas_mock()

        with patch("ragas.evaluate", return_value=ragas_result) as mock_eval, \
             patch("ragas.SingleTurnSample") as mock_sample, \
             patch("ragas.EvaluationDataset") as mock_ds:

            mock_sample.side_effect = lambda **kw: kw  # return kwargs dict as stand-in

            evaluator.evaluate_with_ragas(
                qa_pairs=[{
                    "question": "What are the payment terms?",
                    "expected_answer": "Net 30.",
                    "contexts": ["Payment is due in 30 days."],
                }],
                metrics=metric_mocks,
            )

        # retriever.retrieve should NOT have been called since contexts were supplied
        mock_retriever.retrieve.assert_not_called()

    def test_evaluate_with_ragas_auto_retrieves_contexts(self):
        """evaluate_with_ragas should auto-retrieve contexts when not supplied."""
        evaluator, mock_retriever, mock_chain = self._make_evaluator(
            chain_answer="Payment is due in 30 days."
        )
        metric_mocks = self._make_ragas_metric_mocks()
        ragas_result = self._make_ragas_mock()

        with patch("ragas.evaluate", return_value=ragas_result), \
             patch("ragas.SingleTurnSample") as mock_sample, \
             patch("ragas.EvaluationDataset"):

            mock_sample.side_effect = lambda **kw: kw

            evaluator.evaluate_with_ragas(
                qa_pairs=[{"question": "What are the payment terms?"}],
                metrics=metric_mocks,
            )

        # retriever.retrieve SHOULD have been called
        mock_retriever.retrieve.assert_called_once_with("What are the payment terms?")

    def test_evaluate_with_ragas_returns_correct_structure(self):
        """evaluate_with_ragas should return scores, results, and total_evaluated."""
        evaluator, _, _ = self._make_evaluator(chain_answer="Payment is due in 30 days.")
        metric_mocks = self._make_ragas_metric_mocks(["faithfulness"])
        ragas_result = self._make_ragas_mock(["faithfulness"])

        with patch("ragas.evaluate", return_value=ragas_result), \
             patch("ragas.SingleTurnSample") as mock_sample, \
             patch("ragas.EvaluationDataset"):

            mock_sample.side_effect = lambda **kw: kw

            output = evaluator.evaluate_with_ragas(
                qa_pairs=[{
                    "question": "What is the liability cap?",
                    "expected_answer": "2x annual contract value.",
                    "contexts": ["The liability cap is 2x the annual contract value."],
                }],
                metrics=metric_mocks,
            )

        assert "scores" in output
        assert "results" in output
        assert "total_evaluated" in output
        assert output["total_evaluated"] == 1
        assert "faithfulness" in output["scores"]
        assert output["scores"]["faithfulness"] == pytest.approx(0.8)

    def test_evaluate_with_ragas_no_reference(self):
        """evaluate_with_ragas should work when expected_answer is absent."""
        evaluator, _, _ = self._make_evaluator(chain_answer="Answer without reference.")
        metric_mocks = self._make_ragas_metric_mocks(["answer_relevancy"])
        ragas_result = self._make_ragas_mock(["answer_relevancy"])

        with patch("ragas.evaluate", return_value=ragas_result), \
             patch("ragas.SingleTurnSample") as mock_sample, \
             patch("ragas.EvaluationDataset"):

            captured_kwargs = {}

            def capture(**kw):
                captured_kwargs.update(kw)
                return kw

            mock_sample.side_effect = capture

            evaluator.evaluate_with_ragas(
                qa_pairs=[{
                    "question": "What are termination clauses?",
                    "contexts": ["30-day written notice required."],
                }],
                metrics=metric_mocks,
            )

        # "reference" should not be set when expected_answer is absent
        assert "reference" not in captured_kwargs
