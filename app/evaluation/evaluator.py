"""Evaluation pipeline for RAG system performance measurement."""

from typing import Any, Dict, List, Optional


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


class RAGASEvaluator:
    """Evaluates RAG quality using RAGAS metrics.

    Measures faithfulness, answer relevancy, context precision, and context
    recall for each QA pair, plus aggregate averages across all pairs.
    """

    def __init__(
        self,
        retriever: Any,
        chain: Any,
        llm: Any = None,
        embeddings: Any = None,
    ) -> None:
        """Initialize the RAGASEvaluator.

        Args:
            retriever: DocumentRetriever instance with a ``retrieve(query)``
                method that returns ``List[Dict]`` with "text", "metadata",
                and "score" keys.
            chain: RAGChain instance with an ``invoke(question)`` method that
                returns a dict containing at least an "answer" key.
            llm: Optional LangChain LLM to use for RAGAS metric computation.
                Defaults to the RAGAS built-in LLM (OpenAI) when not provided.
            embeddings: Optional LangChain embeddings for RAGAS metric
                computation.  Defaults to the RAGAS built-in embeddings when
                not provided.
        """
        self.retriever = retriever
        self.chain = chain
        self.llm = llm
        self.embeddings = embeddings

    def evaluate(self, qa_pairs: List[Dict]) -> Dict:
        """Run RAGAS evaluation on a list of QA pairs.

        For each pair the method:
        1. Retrieves relevant context chunks via ``self.retriever`` (unless
           contexts are pre-provided in the pair dict).
        2. Generates an answer via ``self.chain``.
        3. Collects question / answer / contexts / ground-truth into a
           HuggingFace ``Dataset``.
        4. Calls ``ragas.evaluate()`` with faithfulness, answer_relevancy,
           context_precision, and context_recall metrics.

        Args:
            qa_pairs: List of dicts with:
                - "question" (str): The query string.
                - "ground_truth" (str): The expected/reference answer.
                  ``"expected_answer"`` is also accepted as an alias.
                - "contexts" (List[str], optional): Pre-provided context
                  strings.  When absent the retriever is called instead.

        Returns:
            Dict with:
                - "scores": List of per-question metric dicts, each containing
                  "question", "faithfulness", "answer_relevancy",
                  "context_precision", and "context_recall".
                - "aggregate": Dict of average metric scores across all
                  questions, with the same metric keys.
        """
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics.collections import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[str] = []

        for pair in qa_pairs:
            question = pair["question"]
            ground_truth = pair.get("ground_truth", pair.get("expected_answer", ""))

            if "contexts" in pair:
                retrieved_contexts = list(pair["contexts"])
            else:
                results = self.retriever.retrieve(question)
                retrieved_contexts = [r["text"] for r in results]

            response = self.chain.invoke(question)
            generated_answer = response.get("answer", "")

            questions.append(question)
            answers.append(generated_answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        kwargs: Dict[str, Any] = {}
        if self.llm is not None:
            kwargs["llm"] = self.llm
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings

        result = ragas_evaluate(dataset, metrics=metrics, **kwargs)

        result_df = result.to_pandas()
        scores = []
        for _, row in result_df.iterrows():
            scores.append(
                {
                    "question": row["question"],
                    "faithfulness": row.get("faithfulness"),
                    "answer_relevancy": row.get("answer_relevancy"),
                    "context_precision": row.get("context_precision"),
                    "context_recall": row.get("context_recall"),
                }
            )

        aggregate = {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
        }

        return {
            "scores": scores,
            "aggregate": aggregate,
        }
