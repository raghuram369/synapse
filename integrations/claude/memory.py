"""
SynapseClaudeMemory — Persistent memory wrapper for Claude conversations.

Automatically injects relevant context and saves important information.
Privacy-first: all memory stays local. No data sent to Anthropic for storage.

Requirements:
    pip install anthropic synapse-ai-memory
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from synapse import Synapse


class SynapseClaudeMemory:
    """Add persistent, semantic memory to Claude conversations.

    Wraps the Anthropic API to automatically:
    1. Recall relevant memories before each message
    2. Save important information from conversations
    3. Persist everything locally — survives restarts

    Example::

        from synapse import Synapse
        from synapse.integrations.claude import SynapseClaudeMemory

        syn = Synapse("./claude_memory")
        claude = SynapseClaudeMemory(synapse=syn)

        # First session
        response = claude.chat("My name is Alex and I love Python")

        # Later session (remembers!)
        response = claude.chat("What's my name?")
        # → "Your name is Alex!"
    """

    def __init__(
        self,
        synapse: Optional[Synapse] = None,
        path: str = ":memory:",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        system_prompt: str = "",
        recall_limit: int = 10,
        auto_save: bool = True,
        session_id: str = "default",
    ):
        self.synapse = synapse or Synapse(path)
        self.model = model
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._recall_limit = recall_limit
        self._auto_save = auto_save
        self.session_id = session_id
        self._client = None
        self._messages: List[Dict[str, str]] = []

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic is required. Install with: pip install anthropic"
                )
        return self._client

    def chat(self, message: str, **kwargs) -> str:
        """Send a message to Claude with automatic memory.

        1. Recalls relevant memories and injects as context
        2. Sends to Claude
        3. Saves the exchange to memory
        """
        # Recall relevant memories
        memories = self.synapse.recall(message, limit=self._recall_limit)
        memory_context = self._format_memories(memories)

        # Build system prompt with memory
        system = self._system_prompt or "You are a helpful assistant."
        if memory_context:
            system += f"\n\n## Relevant memories about this user:\n{memory_context}"
            system += "\n\nUse these memories naturally. Don't explicitly mention 'my memory says...' unless asked."

        # Add user message to conversation
        self._messages.append({"role": "user", "content": message})

        # Call Claude
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=self._messages,
            **kwargs,
        )

        assistant_message = response.content[0].text

        # Add to conversation
        self._messages.append({"role": "assistant", "content": assistant_message})

        # Auto-save to memory
        if self._auto_save:
            self._save_exchange(message, assistant_message)

        return assistant_message

    def remember(self, content: str, memory_type: str = "fact") -> None:
        """Explicitly save something to memory."""
        self.synapse.remember(
            content,
            memory_type=memory_type,
            metadata={
                "source": "claude",
                "session_id": self.session_id,
            },
        )

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Manually search memory."""
        memories = self.synapse.recall(query, limit=limit)
        return [
            {
                "content": m.content,
                "type": m.memory_type,
                "strength": m.effective_strength,
                "created_at": m.created_at,
            }
            for m in memories
        ]

    def forget(self, query: str) -> int:
        """Forget memories matching a query."""
        memories = self.synapse.recall(query, limit=100)
        count = 0
        for mem in memories:
            self.synapse.forget(mem.id)
            count += 1
        return count

    def reset_conversation(self) -> None:
        """Clear conversation history (keeps long-term memory)."""
        self._messages = []

    def _format_memories(self, memories) -> str:
        """Format memories for injection into system prompt."""
        if not memories:
            return ""
        lines = []
        for mem in memories:
            age_hours = (time.time() - mem.created_at) / 3600
            if age_hours < 1:
                age_str = f"{int(age_hours * 60)}m ago"
            elif age_hours < 24:
                age_str = f"{int(age_hours)}h ago"
            else:
                age_str = f"{int(age_hours / 24)}d ago"
            lines.append(f"- [{mem.memory_type}] {mem.content} ({age_str})")
        return "\n".join(lines)

    def _save_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Save conversation exchange to memory."""
        self.synapse.remember(
            user_msg,
            memory_type="observation",
            metadata={
                "source": "claude",
                "role": "user",
                "session_id": self.session_id,
            },
            episode=f"claude-{self.session_id}",
        )
        # Only save substantive assistant responses
        if len(assistant_msg) > 20:
            self.synapse.remember(
                assistant_msg[:500],  # Truncate long responses
                memory_type="observation",
                metadata={
                    "source": "claude",
                    "role": "assistant",
                    "session_id": self.session_id,
                },
                episode=f"claude-{self.session_id}",
            )
