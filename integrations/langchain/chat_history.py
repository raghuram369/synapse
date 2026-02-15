"""
SynapseChatMessageHistory — LangChain chat history backed by Synapse.

Privacy-first: all messages stay local. No cloud storage.

Requirements:
    pip install langchain-core synapse-ai-memory
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from synapse import Synapse


class SynapseChatMessageHistory:
    """Persistent chat message history using Synapse as the backend.

    Unlike simple list-based histories, Synapse provides semantic recall —
    relevant past messages surface automatically based on context.

    Implements LangChain's BaseChatMessageHistory interface.

    Example::

        from synapse import Synapse
        from synapse.integrations.langchain import SynapseChatMessageHistory

        syn = Synapse("./chat_memory")
        history = SynapseChatMessageHistory(synapse=syn, session_id="user-123")

        history.add_user_message("I love Italian food")
        history.add_ai_message("Great! Do you have a favorite dish?")

        # Later — semantic recall
        relevant = history.search("What cuisine do they prefer?")
    """

    _base_resolved = False

    def __new__(cls, *args, **kwargs):
        if not cls._base_resolved:
            try:
                from langchain_core.chat_history import BaseChatMessageHistory
                cls.__bases__ = (BaseChatMessageHistory,) + tuple(
                    b for b in cls.__bases__ if b is not object
                )
                cls._base_resolved = True
            except ImportError:
                pass
        return super().__new__(cls)

    def __init__(
        self,
        synapse: Optional[Synapse] = None,
        path: str = ":memory:",
        session_id: str = "default",
    ):
        self.synapse = synapse or Synapse(path)
        self.session_id = session_id
        try:
            super().__init__()
        except TypeError:
            pass

    def _make_metadata(self, role: str) -> Dict[str, Any]:
        return {
            "role": role,
            "session_id": self.session_id,
            "source": "langchain_chat_history",
            "timestamp": time.time(),
        }

    def add_user_message(self, message: str) -> None:
        """Add a user message to history."""
        self.synapse.remember(
            message,
            memory_type="observation",
            metadata=self._make_metadata("human"),
            episode=f"chat-{self.session_id}",
        )

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to history."""
        self.synapse.remember(
            message,
            memory_type="observation",
            metadata=self._make_metadata("ai"),
            episode=f"chat-{self.session_id}",
        )

    def add_message(self, message: Any) -> None:
        """Add a LangChain BaseMessage to history."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            if isinstance(message, HumanMessage):
                self.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                self.add_ai_message(message.content)
            elif isinstance(message, SystemMessage):
                self.synapse.remember(
                    message.content,
                    memory_type="observation",
                    metadata=self._make_metadata("system"),
                    episode=f"chat-{self.session_id}",
                )
            else:
                self.synapse.remember(
                    str(message.content),
                    memory_type="observation",
                    metadata=self._make_metadata("unknown"),
                    episode=f"chat-{self.session_id}",
                )
        except ImportError:
            # Fallback: store as plain text
            self.synapse.remember(
                str(message),
                memory_type="observation",
                metadata=self._make_metadata("unknown"),
                episode=f"chat-{self.session_id}",
            )

    @property
    def messages(self) -> List[Any]:
        """Return all messages in this session, ordered by time."""
        all_memories = self.synapse.recall("", limit=10000)

        # Filter to this session
        session_memories = [
            m for m in all_memories
            if (m.metadata or {}).get("session_id") == self.session_id
        ]

        # Sort by creation time
        session_memories.sort(key=lambda m: m.created_at)

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            messages = []
            for mem in session_memories:
                role = (mem.metadata or {}).get("role", "human")
                if role == "ai":
                    messages.append(AIMessage(content=mem.content))
                elif role == "system":
                    messages.append(SystemMessage(content=mem.content))
                else:
                    messages.append(HumanMessage(content=mem.content))
            return messages
        except ImportError:
            return [{"role": (m.metadata or {}).get("role", "human"), "content": m.content}
                    for m in session_memories]

    def search(self, query: str, limit: int = 5) -> List[Any]:
        """Semantic search across chat history — Synapse's superpower.

        Unlike simple history, this finds *relevant* messages regardless
        of how long ago they were said.
        """
        results = self.synapse.recall(query, limit=limit)
        # Filter to this session
        return [m for m in results
                if (m.metadata or {}).get("session_id") == self.session_id]

    def clear(self) -> None:
        """Clear all messages in this session."""
        all_memories = self.synapse.recall("", limit=10000)
        for mem in all_memories:
            if (mem.metadata or {}).get("session_id") == self.session_id:
                self.synapse.forget(mem.id)
