"""Retrieval pipeline for fetching relevant document chunks."""

from typing import Dict, List

from app.vectorstore.store import VectorStoreManager


class DocumentRetriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(self, vector_store_manager: VectorStoreManager, k: int = 5) -> None:
        """Initialize the DocumentRetriever.

        Args:
            vector_store_manager: VectorStoreManager instance to search.
            k: Number of top results to retrieve.
        """
        self.vector_store_manager = vector_store_manager
        self.k = k

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve the most relevant document chunks for a query.

        Args:
            query: The search query string.

        Returns:
            List of dicts with "text", "metadata", and "score" keys.
        """
        return self.vector_store_manager.similarity_search(query, k=self.k)

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks as a context string for the LLM.

        Args:
            results: List of dicts with "text" and "metadata" keys.

        Returns:
            Formatted context string with source citations.
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            text = result.get("text", "")
            context_parts.append(f"[{i}] Source: {source}, Page: {page}\n{text}")

        return "\n\n".join(context_parts)
