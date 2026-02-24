#!/usr/bin/env python3
"""Runner script for RAGAS evaluation of the Smart Contract Assistant RAG pipeline.

Usage
-----
From the project root::

    python scripts/run_ragas_evaluation.py
    python scripts/run_ragas_evaluation.py --output ragas_evaluation_results.json
    python scripts/run_ragas_evaluation.py --qa-file path/to/qa_pairs.json

The QA JSON file must contain a list of objects with "question" and
"ground_truth" keys (and optionally "contexts").
"""

import argparse
import json
import os
import sys

# Ensure the project root is importable when running the script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.evaluation.evaluator import RAGASEvaluator
from app.generation.chain import RAGChain
from app.generation.llm import get_llm
from app.ingestion.embedder import Embedder
from app.retrieval.retriever import DocumentRetriever
from app.vectorstore.store import VectorStoreManager

# ---------------------------------------------------------------------------
# Default sample QA pairs (used when no --qa-file is provided)
# ---------------------------------------------------------------------------

DEFAULT_QA_PAIRS = [
    {
        "question": "What are the payment terms in the contract?",
        "ground_truth": "Payment is due within 30 days of invoice.",
    },
    {
        "question": "When does the contract expire?",
        "ground_truth": "The contract expires on December 31, 2025.",
    },
    {
        "question": "What is the liability cap?",
        "ground_truth": "The liability cap is two times the annual contract value.",
    },
    {
        "question": "What triggers contract termination?",
        "ground_truth": "Either party may terminate with 30 days written notice.",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    """Print a formatted summary of RAGAS evaluation results to stdout."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Aggregate Scores:")
    aggregate = results["aggregate"]
    for metric, score in aggregate.items():
        score_val = score if score is not None else 0.0
        bar = "█" * int(score_val * 20)
        print(f"  {metric:<25} {score_val:.3f}  {bar}")

    print("\n📋 Per-Question Scores:")
    print("-" * 60)
    for item in results["scores"]:
        print(f"\n  Q: {item['question']}")
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            val = item.get(metric)
            val_str = f"{val:.3f}" if val is not None else "N/A"
            print(f"    {metric:<25} {val_str}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_file: str = None, qa_file: str = None) -> None:
    """Initialize app components, run RAGAS evaluation, and print results.

    Args:
        output_file: Optional path to save detailed JSON results.
        qa_file: Optional path to a JSON file containing QA pairs.
    """
    settings = get_settings()

    # Load QA pairs
    if qa_file:
        with open(qa_file) as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} QA pair(s) from {qa_file}")
    else:
        qa_pairs = DEFAULT_QA_PAIRS
        print(f"Using {len(qa_pairs)} default sample QA pair(s)")

    print("\nInitializing app components...")

    # Embeddings
    embedder = Embedder(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
    )
    lc_embeddings = embedder.get_langchain_embeddings()

    # Vector store
    vector_store_manager = VectorStoreManager(
        store_type=settings.vector_store_type,
        persist_directory=settings.vector_store_persist_dir,
        embeddings=lc_embeddings,
    )

    # Retriever
    retriever = DocumentRetriever(
        vector_store_manager=vector_store_manager,
        k=settings.top_k,
        rerank=settings.rerank_enabled,
        rerank_multiplier=settings.rerank_multiplier,
    )

    # LLM
    llm = get_llm(
        provider=settings.llm_provider,
        model_name=settings.model_name,
        groq_api_key=settings.groq_api_key,
    )

    # RAG chain
    rag_chain = RAGChain(llm=llm, retriever=retriever)

    # RAGAS evaluator — pass the same LLM so RAGAS uses the configured model
    evaluator = RAGASEvaluator(retriever=retriever, chain=rag_chain, llm=llm)

    print(f"Running RAGAS evaluation on {len(qa_pairs)} QA pair(s)...")
    results = evaluator.evaluate(qa_pairs)

    _print_results(results)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Smart Contract Assistant."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Path to save detailed JSON results (e.g., ragas_evaluation_results.json)",
    )
    parser.add_argument(
        "--qa-file",
        "-q",
        default=None,
        metavar="FILE",
        help="Path to a JSON file with QA pairs (list of {question, ground_truth})",
    )
    args = parser.parse_args()
    main(output_file=args.output, qa_file=args.qa_file)
