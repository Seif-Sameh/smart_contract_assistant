"""Conversation state management for multi-turn dialogue."""

import uuid
from typing import Dict, List, Optional


class ConversationState:
    """Manages conversation history for a single session."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        """Initialize the conversation state.

        Args:
            session_id: Optional session identifier. If not provided, a UUID is generated.
        """
        self._session_id = session_id or str(uuid.uuid4())
        self._history: List[Dict] = []

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., "user" or "assistant").
            content: The text content of the message.
        """
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict]:
        """Return the full conversation history.

        Returns:
            List of dicts with "role" and "content" keys.
        """
        return list(self._history)

    def format_history(self) -> str:
        """Format the conversation history as a string for LLM prompts.

        Returns:
            Formatted conversation history string.
        """
        if not self._history:
            return "No previous conversation."

        lines = []
        for msg in self._history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self._history = []
