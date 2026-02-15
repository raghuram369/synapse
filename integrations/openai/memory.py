"""
SynapseGPTMemory — Persistent memory wrapper for OpenAI chat completions.

Automatically injects relevant context and saves conversations.
Privacy-first: memory stays local. Only inference calls go to OpenAI.

Requirements:
    pip install openai synapse-ai-memory
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from synapse import Synapse


class SynapseGPTMemory:
    """Add persistent, semantic memory to OpenAI GPT conversations.

    Wraps the OpenAI chat completions API to automatically:
    1. Recall relevant memories and inject as system context
    2. Save conversation exchanges to local memory
    3. Persist across sessions — GPT finally remembers!

    Example::

        from synapse import Synapse
        from synapse.integrations.openai import SynapseGPTMemory

        syn = Synapse("./gpt_memory")
        gpt = SynapseGPTMemory(synapse=syn)

        # Session 1
        response = gpt.chat("I'm Alex, I prefer Python over JS")

        # Session 2 (remembers!)
        response = gpt.chat("What programming language do I prefer?")
    """

    def __init__(
        self,
        synapse: Optional[Synapse] = None,
        path: str = ":memory:",
        model: str = "gpt-4o-mini",
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
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
        return self._client

    def chat(self, message: str, **kwargs) -> str:
        """Send a message to GPT with automatic memory.

        1. Recalls relevant memories → injects as system context
        2. Calls OpenAI chat completions
        3. Saves the exchange to local memory
        """
        # Recall
        memories = self.synapse.recall(message, limit=self._recall_limit)
        memory_context = self._format_memories(memories)

        system = self._system_prompt or "You are a helpful assistant."
        if memory_context:
            system += f"\n\n## Relevant memories about this user:\n{memory_context}"
            system += "\n\nUse these memories naturally."

        # Build messages
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(self._messages)
        api_messages.append({"role": "user", "content": message})

        # Call OpenAI
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            **kwargs,
        )

        assistant_message = response.choices[0].message.content

        # Track conversation
        self._messages.append({"role": "user", "content": message})
        self._messages.append({"role": "assistant", "content": assistant_message})

        # Auto-save
        if self._auto_save:
            self._save_exchange(message, assistant_message)

        return assistant_message

    def remember(self, content: str, memory_type: str = "fact") -> None:
        """Explicitly save to memory."""
        self.synapse.remember(
            content, memory_type=memory_type,
            metadata={"source": "openai", "session_id": self.session_id},
        )

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory."""
        memories = self.synapse.recall(query, limit=limit)
        return [
            {"content": m.content, "type": m.memory_type, "strength": m.effective_strength}
            for m in memories
        ]

    def reset_conversation(self) -> None:
        """Clear conversation history (keeps long-term memory)."""
        self._messages = []

    def _format_memories(self, memories) -> str:
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
        self.synapse.remember(
            user_msg, memory_type="observation",
            metadata={"source": "openai", "role": "user", "session_id": self.session_id},
            episode=f"gpt-{self.session_id}",
        )
        if len(assistant_msg) > 20:
            self.synapse.remember(
                assistant_msg[:500], memory_type="observation",
                metadata={"source": "openai", "role": "assistant", "session_id": self.session_id},
                episode=f"gpt-{self.session_id}",
            )
