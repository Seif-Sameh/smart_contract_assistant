"""LangServe-compatible runnables for the Smart Contract Assistant."""

import os
from typing import Dict, Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from app.conversation.state import ConversationState
from app.generation.guardrails import apply_disclaimer, check_safety
from app.ingestion.chunker import chunk_documents
from app.ingestion.parser import parse_file
from app.summarization.summarizer import DocumentSummarizer


# ---------------------------------------------------------------------------
# Chat runnable
# ---------------------------------------------------------------------------


class ChatInput(BaseModel):
    """Input schema for the RAG chat runnable."""

    question: str
    session_id: Optional[str] = None


class ChatOutput(BaseModel):
    """Output schema for the RAG chat runnable."""

    answer: str
    sources: list
    session_id: str


def create_rag_runnable(rag_chain, conversation_states: Dict):
    """Create a LangServe-compatible runnable from the RAG chain.

    Args:
        rag_chain: Initialized RAGChain instance.
        conversation_states: Shared dict mapping session_id -> ConversationState.

    Returns:
        RunnableLambda typed with ChatInput / ChatOutput.
    """

    def _invoke(input_data) -> dict:
        # Accept both dict and Pydantic model
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
            session_id = input_data.get("session_id") or None
        else:
            question = input_data.question
            session_id = input_data.session_id or None

        # Safety check
        is_safe, reason = check_safety(question)
        if not is_safe:
            raise ValueError(f"Query blocked: {reason}")

        # Get or create conversation state
        if session_id is None:
            session_id = ConversationState().session_id
        if session_id not in conversation_states:
            conversation_states[session_id] = ConversationState(session_id=session_id)

        state = conversation_states[session_id]

        result = rag_chain.invoke(
            question=question,
            conversation_history=state.get_history(),
        )

        # Update conversation state
        state.add_message("user", question)
        state.add_message("assistant", result["answer"])

        answer = apply_disclaimer(result["answer"])

        return {
            "answer": answer,
            "sources": result["sources"],
            "session_id": session_id,
        }

    return RunnableLambda(_invoke).with_types(
        input_type=ChatInput,
        output_type=ChatOutput,
    )


# ---------------------------------------------------------------------------
# Summarize runnable
# ---------------------------------------------------------------------------


class SummarizeInput(BaseModel):
    """Input schema for the summarization runnable."""

    filename: str


class SummarizeOutput(BaseModel):
    """Output schema for the summarization runnable."""

    summary: str
    filename: str


def create_summarize_runnable(rag_chain, settings):
    """Create a LangServe-compatible runnable for document summarization.

    Args:
        rag_chain: Initialized RAGChain instance (used to access the LLM).
        settings: Application Settings instance.

    Returns:
        RunnableLambda typed with SummarizeInput / SummarizeOutput.
    """

    def _invoke(input_data) -> dict:
        if isinstance(input_data, dict):
            filename = input_data.get("filename", "")
        else:
            filename = input_data.filename

        file_path = os.path.join(settings.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{filename}' not found.")

        documents = parse_file(file_path)
        chunks = chunk_documents(documents)
        summarizer = DocumentSummarizer(llm=rag_chain.llm)
        summary = summarizer.summarize(chunks)

        return {"summary": summary, "filename": filename}

    return RunnableLambda(_invoke).with_types(
        input_type=SummarizeInput,
        output_type=SummarizeOutput,
    )
