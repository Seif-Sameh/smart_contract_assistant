"""Vector store management for document retrieval."""

import os
import shutil
import uuid
from typing import Dict, List, Optional


class VectorStoreManager:
    """Manages a vector store for document storage and similarity search."""

    def __init__(
        self,
        store_type: str = "chroma",
        persist_directory: str = "./data/vectorstore",
        embeddings=None,
    ) -> None:
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self._store = None
        self._collection = None
        self._client = None

    def _get_or_create_store(self):
        """Get existing store or create a new one."""
<<<<<<< HEAD
        if self._store is not None:
            return self._store
=======
        if self._collection is not None:
            return self._collection
>>>>>>> d3d5d87 (.)

        if self.store_type == "chroma":
            import chromadb

            os.makedirs(self.persist_directory, exist_ok=True)
<<<<<<< HEAD

            # Use chromadb's PersistentClient directly to avoid Pydantic type inference issues
=======
            # Use PersistentClient directly — bypasses Pydantic Settings issue
>>>>>>> d3d5d87 (.)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            self._store = self._collection
        elif self.store_type == "faiss":
<<<<<<< HEAD
            # FAISS requires texts to initialize — store is set up on first add
=======
>>>>>>> d3d5d87 (.)
            pass
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self._collection

    def add_documents(self, chunks: List[Dict]) -> None:
        """Add text chunks to the vector store."""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        if self.store_type == "chroma":
            self._get_or_create_store()

<<<<<<< HEAD
            # Sanitize metadata: Chroma only supports str, int, float, bool values
=======
            # Sanitize metadata — Chroma only supports str, int, float, bool
>>>>>>> d3d5d87 (.)
            sanitized_metadatas = []
            for meta in metadatas:
                sanitized = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized[k] = v
                    else:
                        sanitized[k] = str(v)
                sanitized_metadatas.append(sanitized)

<<<<<<< HEAD
            # Generate embeddings
            embeddings_list = self.embeddings.embed_documents(texts)

            # Generate unique IDs
=======
            # Generate embeddings manually
            embeddings_list = self.embeddings.embed_documents(texts)
>>>>>>> d3d5d87 (.)
            ids = [str(uuid.uuid4()) for _ in texts]

            self._collection.add(
                documents=texts,
                embeddings=embeddings_list,
                metadatas=sanitized_metadatas,
                ids=ids,
            )
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if self._store is None:
                self._store = FAISS.from_texts(
<<<<<<< HEAD
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
=======
                    texts=texts, embedding=self.embeddings, metadatas=metadatas,
>>>>>>> d3d5d87 (.)
                )
            else:
                self._store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search against stored documents."""
        if self.store_type == "chroma":
            if self._collection is None:
                self._get_or_create_store()
            if self._collection is None or self._collection.count() == 0:
                return []

            query_embedding = self.embeddings.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

<<<<<<< HEAD
        Returns:
            List of dicts with "text", "metadata", and "score" keys.
        """
        if self.store_type == "chroma":
            if self._collection is None:
                self._get_or_create_store()
            if self._collection is None or self._collection.count() == 0:
                return []

            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            output = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0
                    # Convert cosine distance to similarity score.
                    # Chroma returns cosine distance in [0, 2] where 0 is identical
                    # and 2 is opposite, so 1.0 - distance maps to [-1, 1] similarity.
                    score = 1.0 - distance
                    output.append({
                        "text": doc_text,
                        "metadata": metadata,
                        "score": float(score),
                    })
            return output

=======
            output = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {}
                    )
                    distance = (
                        results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else 0.0
                    )
                    score = 1.0 - distance  # cosine distance → similarity
                    output.append({
                        "text": doc_text,
                        "metadata": metadata,
                        "score": float(score),
                    })
            return output

>>>>>>> d3d5d87 (.)
        elif self.store_type == "faiss":
            if self._store is None:
                return []
            results = self._store.similarity_search_with_score(query, k=k)
            return [
<<<<<<< HEAD
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
=======
                {"text": doc.page_content, "metadata": doc.metadata, "score": float(score)}
>>>>>>> d3d5d87 (.)
                for doc, score in results
            ]

        return []

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self._store = None
        self._collection = None
        self._client = None
<<<<<<< HEAD

=======
>>>>>>> d3d5d87 (.)
        if self.store_type == "chroma":
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

    def get_retriever(self, k: int = 5):
<<<<<<< HEAD
        """Return a LangChain retriever for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            LangChain retriever object, or None if store is empty.
        """
        # For Chroma direct client, we don't use LangChain retriever
        # The similarity_search method is used directly instead
        return None
=======
        """Not used with direct ChromaDB client — use similarity_search instead."""
        return None
>>>>>>> d3d5d87 (.)
