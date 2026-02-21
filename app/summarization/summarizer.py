"""Document summarization using LangChain summarization chains."""

from typing import Dict, List

from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


class DocumentSummarizer:
    """Summarizes document chunks using map-reduce or refine strategies."""

    def __init__(self, llm) -> None:
        """Initialize the DocumentSummarizer.

        Args:
            llm: LangChain LLM object to use for summarization.
        """
        self.llm = llm

    def summarize(self, chunks: List[Dict], strategy: str = "map_reduce") -> str:
        """Summarize document chunks.

        Args:
            chunks: List of dicts with "text" and "metadata" keys.
            strategy: Summarization strategy, one of "map_reduce" or "refine".

        Returns:
            Summary string.

        Raises:
            ValueError: If an unsupported strategy is specified.
        """
        if strategy not in ("map_reduce", "refine"):
            raise ValueError(f"Unsupported strategy: {strategy}. Use 'map_reduce' or 'refine'.")

        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]

        chain = load_summarize_chain(self.llm, chain_type=strategy)
        result = chain.invoke({"input_documents": documents})

        if isinstance(result, dict):
            return result.get("output_text", str(result))
        return str(result)
