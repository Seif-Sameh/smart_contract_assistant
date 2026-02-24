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

    def evaluate_with_ragas(
        self,
        qa_pairs: List[Dict],
        metrics: Optional[List[Any]] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ) -> Dict:
        """Evaluate the RAG pipeline using RAGAS metrics.

        Each entry in ``qa_pairs`` must contain a ``"question"`` key.
        Optionally it may contain:
        - ``"expected_answer"`` – reference answer used for metrics such as
          ``context_recall`` and ``answer_correctness``.
        - ``"contexts"`` – pre-fetched list of context strings. When absent,
          contexts are retrieved automatically via ``self.retriever``.

        Args:
            qa_pairs: List of dicts describing each evaluation sample.
            metrics: RAGAS metric instances to evaluate. Defaults to
                ``[faithfulness, answer_relevancy, context_precision]``.
            llm: Optional LangChain-compatible LLM to use for RAGAS scoring.
                When *None*, RAGAS will use its default LLM.
            embeddings: Optional LangChain-compatible embeddings model to use
                for RAGAS scoring. When *None*, RAGAS will use its default.

        Returns:
            Dict with keys:
                - ``"scores"``: Per-metric aggregate scores (metric_name → float).
                - ``"results"``: Per-sample dicts containing question, response,
                  and individual metric scores.
                - ``"total_evaluated"``: Number of samples evaluated.
        """
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.metrics.collections import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ragas is required for RAGAS evaluation. "
                "Install it with: pip install ragas"
            ) from exc

        if metrics is None:
            metrics = [faithfulness, answer_relevancy, context_precision]

        samples = []
        for pair in qa_pairs:
            question = pair["question"]
            reference = pair.get("expected_answer", None)

            # Retrieve contexts if not provided
            if "contexts" in pair:
                contexts = list(pair["contexts"])
            else:
                retrieved = self.retriever.retrieve(question)
                contexts = [doc.get("text", "") for doc in retrieved]

            # Generate answer
            response = self.chain.invoke(question)
            answer = response.get("answer", "")

            sample_kwargs: Dict[str, Any] = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
            if reference is not None:
                sample_kwargs["reference"] = reference

            samples.append(SingleTurnSample(**sample_kwargs))

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        # Build per-sample results list
        result_df = result.to_pandas()
        sample_results = []
        for i, pair in enumerate(qa_pairs):
            row = result_df.iloc[i]
            sample_result: Dict[str, Any] = {
                "question": pair["question"],
                "response": row.get("response", ""),
            }
            for metric in metrics:
                metric_name = metric.name
                sample_result[metric_name] = row.get(metric_name, None)
            sample_results.append(sample_result)

        # Aggregate scores per metric
        scores: Dict[str, float] = {}
        for metric in metrics:
            metric_name = metric.name
            if metric_name in result_df.columns and not result_df[metric_name].isna().all():
                scores[metric_name] = float(result_df[metric_name].mean())

        return {
            "scores": scores,
            "results": sample_results,
            "total_evaluated": len(qa_pairs),
        }

    def run_full_evaluation(
        self,
        qa_pairs: List[Dict],
        include_ragas: bool = False,
        ragas_llm: Optional[Any] = None,
        ragas_embeddings: Optional[Any] = None,
    ) -> Dict:
        """Run the complete evaluation pipeline.

        Args:
            qa_pairs: List of dicts with "question" and "expected_answer" keys.
            include_ragas: When *True*, also run RAGAS-based evaluation and
                include the results under the ``"ragas_metrics"`` key.
            ragas_llm: Optional LLM to pass to :meth:`evaluate_with_ragas`.
            ragas_embeddings: Optional embeddings to pass to
                :meth:`evaluate_with_ragas`.

        Returns:
            Dict combining retrieval and answer quality metrics. When
            ``include_ragas`` is *True*, also contains a ``"ragas_metrics"``
            key with the RAGAS evaluation results.
        """
        queries = [pair["question"] for pair in qa_pairs]
        retrieval_metrics = self.evaluate_retrieval(queries)
        answer_metrics = self.evaluate_answers(qa_pairs)

        full_result: Dict[str, Any] = {
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
        }

        if include_ragas:
            full_result["ragas_metrics"] = self.evaluate_with_ragas(
                qa_pairs,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )

        return full_result
