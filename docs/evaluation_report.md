# Evaluation Report — Smart Contract Assistant

## Executive Summary

This report presents the evaluation results of the Smart Contract Assistant RAG system. The system was assessed across two primary dimensions: retrieval quality (how well relevant document chunks are surfaced) and answer quality (how accurately and faithfully the LLM responds based on retrieved context).

RAGAS automated scoring is now available alongside the original proxy metrics to give a more complete picture of system quality.

---

## Methodology

Evaluation was conducted using a curated set of question-answer pairs derived from sample contract documents. The RAG pipeline was evaluated end-to-end using the `RAGEvaluator` class (`app/evaluation/evaluator.py`) for proxy metrics and the `RAGASEvaluator` class for semantic quality metrics via RAGAS.

**Evaluation steps:**
1. Upload contract documents to the vector store.
2. For each test question, retrieve top-k document chunks.
3. Generate an answer using the RAG chain.
4. Compare generated answers against expected answers.
5. *(RAGAS)* Score faithfulness, answer relevancy, context precision, and context recall automatically.

---

## Running RAGAS Evaluation

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

RAGAS requires an LLM for its internal scoring. The script uses the same LLM configured in `app/config.py` (defaulting to Groq). Ensure the relevant API key is set in `.env` (e.g., `GROQ_API_KEY=...`).

### Running the evaluation script

```bash
# Basic run — prints results to stdout
python scripts/run_ragas_evaluation.py

# Save detailed per-question results to JSON
python scripts/run_ragas_evaluation.py --output ragas_evaluation_results.json

# Load custom QA pairs from a JSON file
python scripts/run_ragas_evaluation.py --qa-file path/to/qa_pairs.json
```

The QA pairs JSON file must be a list of objects with `"question"` and `"ground_truth"` keys:

```json
[
  {
    "question": "What are the payment terms?",
    "ground_truth": "Payment is due within 30 days of invoice."
  }
]
```

### Using `RAGASEvaluator` programmatically

```python
from app.evaluation.evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(retriever=retriever, chain=rag_chain)

results = evaluator.evaluate([
    {"question": "What are the payment terms?", "ground_truth": "Net 30 days."},
])

print(results["aggregate"])   # {'faithfulness': 0.92, 'answer_relevancy': 0.88, ...}
print(results["scores"])      # per-question list
```

---

## RAGAS Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Fraction of answer claims that are grounded in the retrieved context. A score of 1.0 means every claim in the answer can be attributed to a retrieved chunk. | ≥ 0.90 |
| **Answer Relevancy** | Measures how well the generated answer addresses the question. Computed by regenerating candidate questions from the answer and measuring similarity to the original. | ≥ 0.85 |
| **Context Precision** | Measures whether the most relevant context chunks are ranked higher (precision@k). A score of 1.0 means all highly relevant chunks appear first. | ≥ 0.80 |
| **Context Recall** | Measures how much of the ground-truth information is covered by the retrieved context. A score of 1.0 means every piece of the ground-truth answer is found in the retrieved chunks. | ≥ 0.75 |

### How to interpret the results

- **Faithfulness < 0.70**: The LLM is hallucinating — generating claims not supported by the retrieved context. Reduce LLM temperature, tighten the system prompt, or increase `top_k`.
- **Answer Relevancy < 0.70**: Answers are off-topic or verbose. Review the prompt template and ensure the question is clearly passed to the chain.
- **Context Precision < 0.70**: Low-relevance chunks are ranked above high-relevance ones. Consider enabling re-ranking (`rerank_enabled=True`) or tuning chunk size.
- **Context Recall < 0.70**: The retriever is missing important information. Increase `top_k`, decrease `chunk_size`, or review the embedding model.

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

## Results

| Question | Expected Answer (Summary) | Generated Answer (Summary) | Faithful? |
|----------|--------------------------|---------------------------|-----------|
| What are the payment terms? | Net 30 days from invoice | — | — |
| When does the contract expire? | December 31, 2025 | — | — |
| What is the liability cap? | 2x annual contract value | — | — |
| What triggers termination? | 30-day written notice | — | — |

> **Note:** Populate this table with actual evaluation results after running `scripts/run_ragas_evaluation.py`.

---

## Limitations

1. **LLM variability**: Generated answers may vary between runs due to LLM temperature settings.
2. **Embedding quality**: Retrieval performance depends on the quality of the chosen embedding model.
3. **Context window limits**: Very long contracts may require summarization before retrieval to fit within token limits.
4. **RAGAS LLM dependency**: RAGAS metrics themselves require an LLM API call, so evaluation incurs additional API cost.

---

## Recommendations

1. **Build a gold-standard test set**: Manually annotate 50–100 contract QA pairs for rigorous evaluation.
2. **Tune chunking parameters**: Experiment with chunk sizes between 500–2000 tokens and measure retrieval recall.
3. **Increase top-k for complex queries**: Use `top_k=8` or higher for multi-clause questions.
4. **Add re-ranking**: Implement a cross-encoder re-ranker to improve precision of retrieved chunks.
5. **Monitor production metrics**: Log retrieval counts, answer lengths, and user feedback ratings in production.

