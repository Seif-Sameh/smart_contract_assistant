"""RAG chain combining retrieval and generation."""

from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.retrieval.retriever import DocumentRetriever

PROMPT_TEMPLATE = """You are an expert assistant for analyzing contracts and documents.
Use ONLY the following context to answer the question. 
If the answer is not in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer with specific references to the relevant sections/pages:"""


class RAGChain:
    """Retrieval-Augmented Generation chain for document Q&A."""

    _PROMPT = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=PROMPT_TEMPLATE,
    )

    def __init__(self, llm, retriever: DocumentRetriever) -> None:
        """Initialize the RAGChain.

        Args:
            llm: LangChain LLM object for answer generation.
            retriever: DocumentRetriever instance for context retrieval.
        """
        self.llm = llm
        self.retriever = retriever

    def invoke(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate an answer for the given question using retrieved context.

        Args:
            question: The user's question.
            conversation_history: Optional list of previous messages as dicts
                with "role" and "content" keys.

        Returns:
            Dict with keys:
                - "answer": Generated answer string.
                - "sources": List of source dicts used as context.
                - "conversation_history": Updated conversation history.
        """
        if conversation_history is None:
            conversation_history = []

        # Retrieve relevant context
        retrieved_docs = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved_docs)

        # Format conversation history
        history_str = self._format_history(conversation_history)

        # Build the prompt
        prompt_text = self._PROMPT.format(
            context=context,
            history=history_str,
            question=question,
        )

        # Generate answer
        response = self.llm.invoke(prompt_text)

        # Extract answer text from response
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        # Update conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "conversation_history": updated_history,
        }

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for inclusion in the prompt.

        Args:
            history: List of dicts with "role" and "content" keys.

        Returns:
            Formatted history string.
        """
        if not history:
            return "No previous conversation."

        lines = []
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)
