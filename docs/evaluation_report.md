# Evaluation Report — Smart Contract Assistant

## Executive Summary

This report presents the evaluation results of the Smart Contract Assistant RAG system. The system was assessed across two primary dimensions: retrieval quality (how well relevant document chunks are surfaced) and answer quality (how accurately and faithfully the LLM responds based on retrieved context).

---

## Methodology

Evaluation was conducted using a curated set of question-answer pairs derived from sample contract documents. The RAG pipeline was evaluated end-to-end using the `RAGEvaluator` class (`app/evaluation/evaluator.py`).

**Evaluation steps:**
1. Upload contract documents to the vector store.
2. For each test question, retrieve top-k document chunks.
3. Generate an answer using the RAG chain.
4. Compare generated answers against expected answers.

---

## Retrieval Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Average Docs Retrieved | Mean number of chunks returned per query | ≥ 3 |
| Queries Evaluated | Total number of test queries run | — |
| Relevance Rate | % of retrieved chunks judged relevant | ≥ 80% |
| Recall@k | % of ground-truth docs appearing in top-k results | ≥ 70% |

### Notes on Retrieval

- Retrieval quality is sensitive to chunk size and overlap settings.
- Smaller chunks improve precision; larger chunks improve recall.
- Recommended settings: `chunk_size=1000`, `chunk_overlap=200`, `top_k=5`.

---

## Answer Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | % of answer claims supported by retrieved context | ≥ 90% |
| Answer Relevance | % of answers addressing the asked question | ≥ 85% |
| Completeness | % of expected answer points covered | ≥ 75% |
| Average Answer Length | Mean character count of generated answers | 200–800 |

---

## RAGAS Metrics

RAGAS (Retrieval Augmented Generation Assessment) is now integrated into the evaluation pipeline. It provides LLM-judge-based metrics for automatic quality measurement.

### Available Metrics

| Metric | Description | Requires Reference? |
|--------|-------------|---------------------|
| `faithfulness` | Fraction of answer claims supported by retrieved context | No |
| `answer_relevancy` | How well the answer addresses the asked question | No |
| `context_precision` | Fraction of retrieved chunks that are relevant to the question | No |
| `context_recall` | Fraction of ground-truth answer that is covered by retrieved context | Yes |
| `answer_correctness` | Semantic similarity of generated answer to the reference answer | Yes |

### Usage

```python
from app.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator(retriever=retriever, chain=chain)

qa_pairs = [
    {
        "question": "What are the payment terms?",
        "expected_answer": "Net 30 days from invoice.",       # optional
        "contexts": ["Payment is due 30 days after invoice."],  # optional
    },
]

# Standalone RAGAS evaluation
ragas_results = evaluator.evaluate_with_ragas(qa_pairs)
print(ragas_results["scores"])  # {"faithfulness": 0.92, ...}

# Combined evaluation (retrieval + answers + RAGAS)
full_results = evaluator.run_full_evaluation(qa_pairs, include_ragas=True)
print(full_results["ragas_metrics"]["scores"])
```

> **Note:** RAGAS metrics rely on an LLM judge. Pass a `llm=` argument to
> `evaluate_with_ragas()` or `run_full_evaluation()` to use the same LLM
> already configured for the RAG pipeline. When no LLM is provided, RAGAS
> falls back to its own default.

---

| Question | Expected Answer (Summary) | Generated Answer (Summary) | Faithful? |
|----------|--------------------------|---------------------------|-----------|
| What are the payment terms? | Net 30 days from invoice | — | — |
| When does the contract expire? | December 31, 2025 | — | — |
| What is the liability cap? | 2x annual contract value | — | — |
| What triggers termination? | 30-day written notice | — | — |

> **Note:** Populate this table with actual evaluation results after running `RAGEvaluator.run_full_evaluation()`.

---

## Limitations

1. **No ground-truth dataset**: The current evaluator measures proxy metrics (answer length, retrieval count) rather than semantic similarity to gold-standard answers.
2. **LLM variability**: Generated answers may vary between runs due to LLM temperature settings.
3. **Embedding quality**: Retrieval performance depends on the quality of the chosen embedding model.
4. **Context window limits**: Very long contracts may require summarization before retrieval to fit within token limits.
5. **No hallucination detection**: The system does not automatically detect when the LLM generates unsupported claims. Use `evaluate_with_ragas()` with the `faithfulness` metric to measure how well answers are grounded in retrieved context.

---

## Recommendations

1. **Add semantic similarity metrics**: Integrate RAGAS or DeepEval for automated faithfulness and relevance scoring. ✅ RAGAS is now integrated via `RAGEvaluator.evaluate_with_ragas()` and the `include_ragas` flag on `run_full_evaluation()`.
2. **Build a gold-standard test set**: Manually annotate 50–100 contract QA pairs for rigorous evaluation.
3. **Tune chunking parameters**: Experiment with chunk sizes between 500–2000 tokens and measure retrieval recall.
4. **Increase top-k for complex queries**: Use `top_k=8` or higher for multi-clause questions.
5. **Add re-ranking**: Implement a cross-encoder re-ranker to improve precision of retrieved chunks.
6. **Monitor production metrics**: Log retrieval counts, answer lengths, and user feedback ratings in production.
