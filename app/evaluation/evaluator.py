"""Evaluation pipeline for RAG system performance measurement."""

from typing import Dict, List, Optional


class RAGEvaluator:
    """Evaluates retrieval and generation quality of the RAG pipeline."""

    def __init__(self, retriever, chain) -> None:
        """Initialize the RAGEvaluator.

        Args:
            retriever: DocumentRetriever instance to evaluate.
            chain: RAGChain instance to evaluate.
        """
        self.retriever = retriever
        self.chain = chain

    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth_docs: Optional[List[List[str]]] = None,
    ) -> Dict:
        """Evaluate retrieval performance across a set of queries.

        Args:
            queries: List of query strings to evaluate.
            ground_truth_docs: Optional list of expected document sources per query.

        Returns:
            Dict with keys:
                - "avg_num_retrieved": Average number of docs retrieved per query.
                - "queries_evaluated": Total number of queries evaluated.
        """
        total_retrieved = 0

        for query in queries:
            results = self.retriever.retrieve(query)
            total_retrieved += len(results)

        avg_retrieved = total_retrieved / len(queries) if queries else 0.0

        return {
            "avg_num_retrieved": avg_retrieved,
            "queries_evaluated": len(queries),
        }

    def evaluate_answers(self, qa_pairs: List[Dict]) -> Dict:
        """Evaluate answer quality against expected answers.

        Args:
            qa_pairs: List of dicts with "question" and "expected_answer" keys.

        Returns:
            Dict with keys:
                - "avg_answer_length": Average character length of generated answers.
                - "total_evaluated": Total number of QA pairs evaluated.
                - "results": List of dicts with question, expected, and generated answers.
        """
        results = []
        total_length = 0

        for pair in qa_pairs:
            question = pair["question"]
            expected = pair.get("expected_answer", "")

            response = self.chain.invoke(question)
            generated = response.get("answer", "")
            total_length += len(generated)

            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
            })

        avg_length = total_length / len(qa_pairs) if qa_pairs else 0.0

        return {
            "avg_answer_length": avg_length,
            "total_evaluated": len(qa_pairs),
            "results": results,
        }

    def run_full_evaluation(self, qa_pairs: List[Dict]) -> Dict:
        """Run the complete evaluation pipeline.

        Args:
            qa_pairs: List of dicts with "question" and "expected_answer" keys.

        Returns:
            Dict combining retrieval and answer quality metrics.
        """
        queries = [pair["question"] for pair in qa_pairs]
        retrieval_metrics = self.evaluate_retrieval(queries)
        answer_metrics = self.evaluate_answers(qa_pairs)

        return {
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
        }
