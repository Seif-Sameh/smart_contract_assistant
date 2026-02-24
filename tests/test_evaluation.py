"""Tests for the evaluation pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.evaluator import RAGEvaluator, RAGASEvaluator


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


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator class."""

    def _make_evaluator(self, retrieve_results=None, chain_answer="Generated answer."):
        """Build a RAGASEvaluator with mocked retriever and chain."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = retrieve_results or [
            {"text": "Context text.", "metadata": {"source": "doc.pdf", "page": 1}, "score": 0.9}
        ]

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": chain_answer,
            "sources": [],
            "conversation_history": [],
        }

        return RAGASEvaluator(retriever=mock_retriever, chain=mock_chain), mock_retriever, mock_chain

    def _make_ragas_result(self, questions):
        """Build a mock RAGAS EvaluationResult with per-question DataFrame."""
        import pandas as pd

        per_question_scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.75,
        }

        rows = [
            {"question": q, **per_question_scores}
            for q in questions
        ]
        df = pd.DataFrame(rows)

        mock_result = MagicMock()
        mock_result.__getitem__.side_effect = per_question_scores.__getitem__
        mock_result.to_pandas.return_value = df
        return mock_result

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_returns_scores_and_aggregate(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should return 'scores' list and 'aggregate' dict."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["What is clause 1?"])

        evaluator, _, _ = self._make_evaluator()
        qa_pairs = [{"question": "What is clause 1?", "ground_truth": "Clause 1 covers payment."}]
        result = evaluator.evaluate(qa_pairs)

        assert "scores" in result
        assert "aggregate" in result
        assert len(result["scores"]) == 1

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_per_question_metric_keys(self, mock_from_dict, mock_ragas_eval):
        """Each entry in 'scores' should contain all four metric keys."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["What is clause 1?"])

        evaluator, _, _ = self._make_evaluator()
        result = evaluator.evaluate([{"question": "What is clause 1?", "ground_truth": "It covers payment."}])

        score = result["scores"][0]
        assert "question" in score
        assert "faithfulness" in score
        assert "answer_relevancy" in score
        assert "context_precision" in score
        assert "context_recall" in score

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_aggregate_metric_keys(self, mock_from_dict, mock_ragas_eval):
        """'aggregate' should contain all four metric keys."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["Q1"])

        evaluator, _, _ = self._make_evaluator()
        result = evaluator.evaluate([{"question": "Q1", "ground_truth": "A1"}])

        aggregate = result["aggregate"]
        assert "faithfulness" in aggregate
        assert "answer_relevancy" in aggregate
        assert "context_precision" in aggregate
        assert "context_recall" in aggregate

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_correct_metric_values(self, mock_from_dict, mock_ragas_eval):
        """Scores and aggregate values should match the mocked RAGAS result."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["What is clause 1?"])

        evaluator, _, _ = self._make_evaluator()
        result = evaluator.evaluate([{"question": "What is clause 1?", "ground_truth": "Payment."}])

        assert result["scores"][0]["faithfulness"] == pytest.approx(0.9)
        assert result["aggregate"]["faithfulness"] == pytest.approx(0.9)
        assert result["aggregate"]["answer_relevancy"] == pytest.approx(0.85)
        assert result["aggregate"]["context_precision"] == pytest.approx(0.8)
        assert result["aggregate"]["context_recall"] == pytest.approx(0.75)

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_calls_retriever_for_each_question(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should call retriever.retrieve() once per QA pair."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["Q1", "Q2"])

        evaluator, mock_retriever, _ = self._make_evaluator()
        qa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        evaluator.evaluate(qa_pairs)

        assert mock_retriever.retrieve.call_count == 2

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_skips_retriever_when_contexts_provided(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should NOT call retriever when 'contexts' key is present."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["Q1"])

        evaluator, mock_retriever, _ = self._make_evaluator()
        qa_pairs = [
            {
                "question": "Q1",
                "ground_truth": "A1",
                "contexts": ["Pre-provided context."],
            }
        ]
        evaluator.evaluate(qa_pairs)

        mock_retriever.retrieve.assert_not_called()

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_calls_chain_for_each_question(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should call chain.invoke() once per QA pair."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["Q1", "Q2"])

        evaluator, _, mock_chain = self._make_evaluator()
        qa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        evaluator.evaluate(qa_pairs)

        assert mock_chain.invoke.call_count == 2

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_accepts_expected_answer_alias(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should accept 'expected_answer' as alias for 'ground_truth'."""
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(["Q1"])

        evaluator, _, _ = self._make_evaluator()
        # Should not raise even when 'ground_truth' is absent
        result = evaluator.evaluate([{"question": "Q1", "expected_answer": "A1"}])

        assert "scores" in result

    @patch("ragas.evaluate")
    @patch("datasets.Dataset.from_dict")
    def test_evaluate_multiple_pairs(self, mock_from_dict, mock_ragas_eval):
        """evaluate() should return one score entry per QA pair."""
        questions = ["Q1", "Q2", "Q3"]
        mock_from_dict.return_value = MagicMock()
        mock_ragas_eval.return_value = self._make_ragas_result(questions)

        evaluator, _, _ = self._make_evaluator()
        qa_pairs = [{"question": q, "ground_truth": f"A{i}"} for i, q in enumerate(questions, 1)]
        result = evaluator.evaluate(qa_pairs)

        assert len(result["scores"]) == 3

