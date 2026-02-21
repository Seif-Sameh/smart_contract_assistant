"""Vector store management for document retrieval."""

from typing import Dict, List, Optional

from langchain.schema import Document


class VectorStoreManager:
    """Manages a vector store for document storage and similarity search."""

    def __init__(
        self,
        store_type: str = "chroma",
        persist_directory: str = "./data/vectorstore",
        embeddings=None,
    ) -> None:
        """Initialize the VectorStoreManager.

        Args:
            store_type: Type of vector store, one of "chroma" or "faiss".
            persist_directory: Directory to persist the vector store.
            embeddings: LangChain embeddings object to use.
        """
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self._store = None

    def _get_or_create_store(self, texts: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None):
        """Get existing store or create a new one.

        Args:
            texts: Optional list of texts to initialize the store with.
            metadatas: Optional list of metadata dicts corresponding to texts.

        Returns:
            Initialized vector store instance.
        """
        if self._store is not None:
            return self._store

        if self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma

            if texts:
                self._store = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    persist_directory=self.persist_directory,
                )
            else:
                self._store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if texts:
                self._store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                )
            else:
                self._store = None  # FAISS requires texts to initialize
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self._store

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add text chunks to the vector store.

        Args:
            chunks: List of dicts with "text" and "metadata" keys.
        """
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        if self._store is None:
            self._get_or_create_store(texts=texts, metadatas=metadatas)
        else:
            self._store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search against stored documents.

        Args:
            query: Query string to search for.
            k: Number of top results to return.

        Returns:
            List of dicts with "text", "metadata", and "score" keys.
        """
        if self._store is None:
            return []

        results = self._store.similarity_search_with_score(query, k=k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self._store = None

        if self.store_type == "chroma":
            import os
            import shutil

            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

    def get_retriever(self, k: int = 5):
        """Return a LangChain retriever for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            LangChain retriever object, or None if store is empty.
        """
        if self._store is None:
            return None
        return self._store.as_retriever(search_kwargs={"k": k})
