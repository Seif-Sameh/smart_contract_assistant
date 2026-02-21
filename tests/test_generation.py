"""Tests for the generation pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from app.conversation.state import ConversationState
from app.generation.guardrails import (
    LEGAL_DISCLAIMER,
    apply_disclaimer,
    check_factuality,
    check_safety,
)


class TestGetLLM:
    """Tests for the get_llm factory function."""

    def test_get_llm_groq_default(self):
        """get_llm should return a ChatGroq instance for the groq provider."""
        mock_chat_groq = MagicMock()
        mock_instance = MagicMock()
        mock_chat_groq.return_value = mock_instance

        with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=mock_chat_groq)}):
            from importlib import reload
            import app.generation.llm as llm_module
            reload(llm_module)

            result = llm_module.get_llm(
                provider="groq",
                model_name="llama-3.1-70b-versatile",
                groq_api_key="test-key",
                temperature=0,
            )

        mock_chat_groq.assert_called_once_with(
            model="llama-3.1-70b-versatile",
            groq_api_key="test-key",
            temperature=0,
        )
        assert result is mock_instance

    def test_get_llm_groq_supported_models(self):
        """get_llm should expose all supported Groq model names."""
        from app.generation.llm import _GROQ_MODELS

        # Verify the required models are present
        required = {
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        }
        assert required.issubset(_GROQ_MODELS)
        assert len(_GROQ_MODELS) >= len(required)

    def test_get_llm_groq_unsupported_model_raises(self):
        """get_llm should raise ValueError for an unknown Groq model."""
        from app.generation.llm import get_llm

        with pytest.raises(ValueError, match="Unsupported Groq model"):
            get_llm(provider="groq", model_name="gpt-4-is-not-groq")

    def test_get_llm_unsupported_provider_raises(self):
        """get_llm should raise ValueError for an unknown provider."""
        from app.generation.llm import get_llm

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(provider="unknown_provider")

    def test_get_llm_openai_fallback(self):
        """get_llm should return a ChatOpenAI instance for the openai provider."""
        mock_chat_openai = MagicMock()
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        with patch.dict(
            "sys.modules",
            {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai)},
        ):
            from importlib import reload
            import app.generation.llm as llm_module
            reload(llm_module)

            result = llm_module.get_llm(provider="openai", model_name="gpt-4")

        mock_chat_openai.assert_called_once()
        assert result is mock_instance

class TestGuardrails:
    """Tests for safety guardrails and disclaimers."""

    def test_safety_check_safe_query(self):
        """Safe queries should pass the safety check."""
        is_safe, reason = check_safety("What are the payment terms in this contract?")
        assert is_safe is True
        assert "safe" in reason.lower()

    def test_safety_check_unsafe_query(self):
        """Harmful queries should be blocked."""
        is_safe, reason = check_safety("how to commit fraud with this contract")
        assert is_safe is False
        assert "harmful" in reason.lower()

    def test_safety_check_case_insensitive(self):
        """Safety check should be case-insensitive."""
        is_safe, reason = check_safety("HOW TO COMMIT FRAUD using contracts")
        assert is_safe is False

    def test_disclaimer_applied(self):
        """apply_disclaimer should append the legal disclaimer to the answer."""
        answer = "The contract expires on December 31, 2024."
        result = apply_disclaimer(answer)
        assert answer in result
        assert LEGAL_DISCLAIMER in result

    def test_disclaimer_content(self):
        """Disclaimer should mention legal advice."""
        result = apply_disclaimer("Some answer.")
        assert "legal" in result.lower()

    def test_check_factuality_insufficient_info(self):
        """check_factuality should return as-is when info is insufficient."""
        answer = "I don't have enough information in the document to answer that."
        result = check_factuality(answer)
        assert result == answer

    def test_check_factuality_with_answer(self):
        """check_factuality should return the answer unchanged when info is present."""
        answer = "The payment term is net 30 days per Section 4.2."
        result = check_factuality(answer)
        assert answer in result


class TestConversationState:
    """Tests for ConversationState class."""

    def test_conversation_state_tracks_history(self):
        """ConversationState should track added messages."""
        state = ConversationState(session_id="test-session")
        state.add_message("user", "What is the termination clause?")
        state.add_message("assistant", "The termination clause is in Section 8.")

        history = state.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is the termination clause?"
        assert history[1]["role"] == "assistant"

    def test_conversation_state_clear(self):
        """clear() should remove all messages from history."""
        state = ConversationState()
        state.add_message("user", "Hello")
        state.clear()
        assert state.get_history() == []

    def test_conversation_state_session_id_generated(self):
        """A session_id should be auto-generated if not provided."""
        state = ConversationState()
        assert state.session_id is not None
        assert len(state.session_id) > 0

    def test_conversation_state_custom_session_id(self):
        """Custom session_id should be preserved."""
        state = ConversationState(session_id="my-session-123")
        assert state.session_id == "my-session-123"

    def test_format_history_empty(self):
        """format_history() should return a default message when history is empty."""
        state = ConversationState()
        formatted = state.format_history()
        assert "No previous conversation" in formatted

    def test_format_history_with_messages(self):
        """format_history() should include all messages."""
        state = ConversationState()
        state.add_message("user", "Question?")
        state.add_message("assistant", "Answer.")
        formatted = state.format_history()
        assert "Question?" in formatted
        assert "Answer." in formatted


class TestRAGChain:
    """Tests for the RAGChain class."""

    def test_rag_chain_invoke(self):
        """RAGChain.invoke should return answer, sources, and updated history."""
        from app.generation.chain import RAGChain

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The payment term is 30 days."
        mock_llm.invoke.return_value = mock_response

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": "Payment due in 30 days.", "metadata": {"source": "contract.pdf", "page": 1}, "score": 0.9}
        ]
        mock_retriever.format_context.return_value = "[1] Source: contract.pdf, Page: 1\nPayment due in 30 days."

        chain = RAGChain(llm=mock_llm, retriever=mock_retriever)
        result = chain.invoke("What are the payment terms?")

        assert "answer" in result
        assert "sources" in result
        assert "conversation_history" in result
        assert result["answer"] == "The payment term is 30 days."
        assert len(result["conversation_history"]) == 2

    def test_rag_chain_invoke_with_history(self):
        """RAGChain.invoke should incorporate existing conversation history."""
        from app.generation.chain import RAGChain

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Yes, it was mentioned earlier."
        mock_llm.invoke.return_value = mock_response

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.format_context.return_value = "No relevant context found."

        chain = RAGChain(llm=mock_llm, retriever=mock_retriever)
        history = [{"role": "user", "content": "What is clause 1?"}, {"role": "assistant", "content": "Clause 1 is about payments."}]

        result = chain.invoke("Was that mentioned before?", conversation_history=history)

        assert len(result["conversation_history"]) == 4
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "Clause 1 is about payments." in prompt_arg
