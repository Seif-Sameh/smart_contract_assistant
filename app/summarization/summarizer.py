<<<<<<< HEAD
"""Document summarization using LangChain."""
=======
"""Document summarization using LangChain LCEL chains."""
>>>>>>> d3d5d87 (.)

from typing import Dict, List

from langchain_core.documents import Document
<<<<<<< HEAD
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
=======
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
>>>>>>> d3d5d87 (.)


MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY:",
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Write a concise summary of the following summaries:\n\n{text}\n\nCONCISE SUMMARY:",
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_summary", "text"],
    template=(
        "Your job is to produce a final summary.\n"
        "We have provided an existing summary up to a certain point:\n"
        "{existing_summary}\n\n"
        "We have the opportunity to refine the existing summary with some more context below.\n"
        "{text}\n\n"
        "Given the new context, refine the original summary. "
        "If the context isn't useful, return the original summary."
    ),
)


class DocumentSummarizer:
    """Summarizes document chunks using map-reduce or refine strategies."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def summarize(self, chunks: List[Dict], strategy: str = "map_reduce") -> str:
        if strategy not in ("map_reduce", "refine"):
            raise ValueError(f"Unsupported strategy: {strategy}. Use 'map_reduce' or 'refine'.")

        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]

        if not documents:
            return "No content to summarize."

        if strategy == "map_reduce":
            return self._map_reduce(documents)
        else:
            return self._refine(documents)

    def _map_reduce(self, documents: List[Document]) -> str:
<<<<<<< HEAD
        # Map step: summarize each document individually
        chain = MAP_PROMPT | self.llm | StrOutputParser()
        summaries = [chain.invoke({"text": doc.page_content}) for doc in documents]

        # Reduce step: combine all summaries
=======
        chain = MAP_PROMPT | self.llm | StrOutputParser()
        summaries = [chain.invoke({"text": doc.page_content}) for doc in documents]
>>>>>>> d3d5d87 (.)
        combined = "\n\n".join(summaries)
        reduce_chain = REDUCE_PROMPT | self.llm | StrOutputParser()
        return reduce_chain.invoke({"text": combined})

    def _refine(self, documents: List[Document]) -> str:
<<<<<<< HEAD
        if not documents:
            return "No content to summarize."
        # Start with first document summary
        first_chain = MAP_PROMPT | self.llm | StrOutputParser()
        summary = first_chain.invoke({"text": documents[0].page_content})

        # Refine with each subsequent document
=======
        first_chain = MAP_PROMPT | self.llm | StrOutputParser()
        summary = first_chain.invoke({"text": documents[0].page_content})
>>>>>>> d3d5d87 (.)
        refine_chain = REFINE_PROMPT | self.llm | StrOutputParser()
        for doc in documents[1:]:
            summary = refine_chain.invoke({
                "existing_summary": summary,
                "text": doc.page_content,
            })
<<<<<<< HEAD

        return summary
=======
        return summary
>>>>>>> d3d5d87 (.)
