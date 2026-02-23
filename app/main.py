"""FastAPI application entry point for the Smart Contract Assistant."""

import os
import shutil
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.conversation.state import ConversationState
from app.generation.chain import RAGChain
from app.generation.guardrails import apply_disclaimer, check_safety
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_file
from app.retrieval.retriever import DocumentRetriever
from app.summarization.summarizer import DocumentSummarizer
from app.vectorstore.store import VectorStoreManager

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

vector_store_manager: Optional[VectorStoreManager] = None
retriever: Optional[DocumentRetriever] = None
rag_chain: Optional[RAGChain] = None
conversation_states: Dict[str, ConversationState] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down application components."""
    global vector_store_manager, retriever, rag_chain

    settings = get_settings()

    # Create data directory if it doesn't exist
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.vector_store_persist_dir, exist_ok=True)

    # Initialize embeddings
    embedder = Embedder(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
    )
    lc_embeddings = embedder.get_langchain_embeddings()

    # Initialize vector store
    vector_store_manager = VectorStoreManager(
        store_type=settings.vector_store_type,
        persist_directory=settings.vector_store_persist_dir,
        embeddings=lc_embeddings,
    )

    # Initialize retriever
    retriever = DocumentRetriever(
        vector_store_manager=vector_store_manager,
        k=settings.top_k,
    )

    # Initialize LLM and RAG chain lazily (avoids requiring API keys on startup)
    try:
        from app.generation.llm import get_llm

        llm = get_llm(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            groq_api_key=settings.groq_api_key,
        )
        rag_chain = RAGChain(llm=llm, retriever=retriever)
    except Exception as e:
        import traceback
        print(f"⚠️ LLM initialization failed: {e}")
        traceback.print_exc()
        rag_chain = None

    yield

    # Cleanup (nothing required here)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Smart Contract Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    question: str
    session_id: Optional[str] = None


class SummarizeRequest(BaseModel):
    """Request body for the /summarize endpoint."""

    filename: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> Dict:
    """Return API health status."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict:
    """Upload a PDF or DOCX document, parse, chunk, and index it.

    Args:
        file: The uploaded file (PDF or DOCX).

    Returns:
        Dict with message, chunks_created count, and filename.
    """
    settings = get_settings()

    allowed_extensions = {".pdf", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save file to disk
    save_path = os.path.join(settings.data_dir, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse
    try:
        documents = parse_file(save_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    # Chunk
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Embed and store
    try:
        vector_store_manager.add_documents(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

    return {
        "message": f"Document '{file.filename}' uploaded and indexed successfully.",
        "chunks_created": len(chunks),
        "filename": file.filename,
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> Dict:
    """Answer a question based on uploaded documents.

    Args:
        request: ChatRequest with question and optional session_id.

    Returns:
        Dict with answer, sources, and session_id.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain is not initialized. Check LLM configuration.",
        )

    # Safety check
    is_safe, reason = check_safety(request.question)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Query blocked: {reason}")

    # Get or create conversation state
    session_id = request.session_id or ConversationState().session_id
    if session_id not in conversation_states:
        conversation_states[session_id] = ConversationState(session_id=session_id)

    state = conversation_states[session_id]

    try:
        result = rag_chain.invoke(
            question=request.question,
            conversation_history=state.get_history(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Update conversation state
    state.add_message("user", request.question)
    state.add_message("assistant", result["answer"])

    answer = apply_disclaimer(result["answer"])

    return {
        "answer": answer,
        "sources": result["sources"],
        "session_id": session_id,
    }


@app.post("/summarize")
async def summarize_document(request: SummarizeRequest) -> Dict:
    """Summarize a previously uploaded document.

    Args:
        request: SummarizeRequest with filename.

    Returns:
        Dict with summary and filename.
    """
    settings = get_settings()
    file_path = os.path.join(settings.data_dir, request.filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found.",
        )

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="LLM is not initialized. Check LLM configuration.",
        )

    try:
        documents = parse_file(file_path)
        chunks = chunk_documents(documents)
        summarizer = DocumentSummarizer(llm=rag_chain.llm)
        summary = summarizer.summarize(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    return {"summary": summary, "filename": request.filename}


@app.delete("/reset")
async def reset() -> Dict:
    """Clear the vector store and all conversation states.

    Returns:
        Dict with reset confirmation message.
    """
    global conversation_states

    if vector_store_manager is not None:
        vector_store_manager.clear()

    conversation_states = {}

    return {"message": "Reset successful"}
