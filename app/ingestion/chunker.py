"""Split documents into smaller overlapping text chunks."""

from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of dicts with "text" and "metadata" keys.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        List of dicts with "text" and "metadata" keys.
        Each metadata dict includes a "chunk_index" field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        split_texts = splitter.split_text(text)
        for idx, chunk_text in enumerate(split_texts):
            chunk_metadata = {**metadata, "chunk_index": idx}
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    return chunks
