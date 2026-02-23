"""Retrieval pipeline with reranking for multi-document search."""

from typing import Dict, List, Optional

from app.vectorstore.store import VectorStoreManager

# Number of characters used as the deduplication key for candidate chunks.
# Using the first 200 characters provides a reliable fingerprint while
# keeping comparison cost low.
_DEDUP_TEXT_PREFIX_LENGTH = 200


class DocumentRetriever:
    """Retrieves and reranks relevant document chunks across all uploaded documents."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        k: int = 5,
        rerank: bool = True,
        rerank_multiplier: int = 3,
    ) -> None:
        """Initialize the DocumentRetriever.

        Args:
            vector_store_manager: VectorStoreManager instance to search.
            k: Number of top results to retrieve.
            rerank: Whether to apply reranking on retrieved candidates.
            rerank_multiplier: Factor by which to over-fetch candidates for reranking.
        """
        self.vector_store_manager = vector_store_manager
        self.k = k
        self.rerank = rerank
        self.rerank_multiplier = rerank_multiplier

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve the most relevant document chunks with optional reranking.

        Args:
            query: The search query string.

        Returns:
            List of dicts with "text", "metadata", and "score" keys.
        """
        if self.rerank:
            # Over-fetch candidates
            candidates = self.vector_store_manager.similarity_search(
                query, k=self.k * self.rerank_multiplier
            )
            if not candidates:
                return []
            # Rerank candidates
            reranked = self._rerank_candidates(query, candidates)
            return reranked[: self.k]
        else:
            return self.vector_store_manager.similarity_search(query, k=self.k)

    def _rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates by computing cosine similarity between query and each chunk embedding.

        Args:
            query: The search query string.
            candidates: List of candidate dicts with "text" key.

        Returns:
            Deduplicated list of candidates sorted by rerank score descending.
        """
        embeddings = self.vector_store_manager.embeddings
        if embeddings is None:
            return candidates

        import numpy as np

        # Get query embedding
        query_embedding = np.array(embeddings.embed_query(query))

        # Get embeddings for all candidate texts
        candidate_texts = [c["text"] for c in candidates]
        candidate_embeddings = np.array(embeddings.embed_documents(candidate_texts))

        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarities = candidate_norms @ query_norm

        # Attach rerank score and sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(similarities[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Remove duplicates (same text from different chunks)
        seen_texts: set = set()
        unique_candidates = []
        for c in candidates:
            text_key = c["text"][:_DEDUP_TEXT_PREFIX_LENGTH]  # Use prefix as dedup key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_candidates.append(c)

        return unique_candidates

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks as context string with multi-document source citations.

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
            score = result.get("rerank_score", result.get("score", 0))
            text = result.get("text", "")
            context_parts.append(
                f"[{i}] Source: {source}, Page: {page}, Relevance: {score:.2f}\n{text}"
            )

        return "\n\n".join(context_parts)
