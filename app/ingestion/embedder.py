"""Generate text embeddings using SentenceTransformers or OpenAI."""

from typing import List


class Embedder:
    """Embedding model wrapper supporting multiple providers."""

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the embedder.

        Args:
            provider: Embedding provider, one of "sentence_transformers" or "openai".
            model_name: Name of the embedding model to use.
        """
        self.provider = provider
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Lazily load the embedding model."""
        if self._model is not None:
            return

        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        elif self.provider == "openai":
            from openai import OpenAI

            self._model = OpenAI()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        self._load_model()

        if self.provider == "sentence_transformers":
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        elif self.provider == "openai":
            response = self._model.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            return [item.embedding for item in response.data]
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_langchain_embeddings(self):
        """Return a LangChain-compatible embeddings object.

        Returns:
            HuggingFaceEmbeddings or OpenAIEmbeddings instance.
        """
        if self.provider == "sentence_transformers":
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=self.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
