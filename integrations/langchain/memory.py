"""
SynapseMemory — LangChain BaseMemory implementation backed by Synapse.

Privacy-first: all memory stays local. No API calls for storage.
Works with any LangChain chain via the standard memory interface.

Requirements:
    pip install langchain-core synapse-ai-memory
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from synapse import Synapse


def _import_langchain():
    """Lazy import LangChain to avoid hard dependency."""
    try:
        from langchain_core.memory import BaseMemory
        return BaseMemory
    except ImportError:
        raise ImportError(
            "LangChain is required for SynapseMemory. "
            "Install it with: pip install langchain-core"
        )


class SynapseMemory:
    """LangChain-compatible memory backed by Synapse.

    Stores conversation history and relevant facts in a local Synapse
    database. Recall is semantic — not just last-N messages.

    Example::

        from synapse import Synapse
        from synapse.integrations.langchain import SynapseMemory

        syn = Synapse("./agent_memory")
        memory = SynapseMemory(synapse=syn)

        # Use with any LangChain chain
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    """

    # Set after dynamic base resolution
    _base_resolved = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        # Dynamically inject BaseMemory as parent if available
        if not cls._base_resolved:
            try:
                BaseMemory = _import_langchain()
                # Rebuild class with proper base
                cls.__bases__ = (BaseMemory,) + tuple(
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
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        recall_limit: int = 10,
        store_inputs: bool = True,
        store_outputs: bool = True,
        return_messages: bool = False,
    ):
        self.synapse = synapse or Synapse(path)
        self._memory_key = memory_key
        self._input_key = input_key
        self._output_key = output_key
        self._recall_limit = recall_limit
        self._store_inputs = store_inputs
        self._store_outputs = store_outputs
        self._return_messages = return_messages

        # Try calling super().__init__() for BaseMemory compatibility
        try:
            super().__init__()
        except TypeError:
            pass

    @property
    def memory_variables(self) -> List[str]:
        """Keys this memory injects into the chain."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Recall relevant memories based on the current input.

        Synapse performs semantic search — not just returning the last N
        messages, but the most *relevant* memories for this query.
        """
        # Build context from all input values
        context_parts = []
        for key, value in inputs.items():
            if isinstance(value, str):
                context_parts.append(value)
        context = " ".join(context_parts) if context_parts else ""

        memories = self.synapse.recall(context, limit=self._recall_limit)

        if self._return_messages:
            try:
                from langchain_core.messages import AIMessage, HumanMessage
                messages = []
                for mem in memories:
                    meta = mem.metadata or {}
                    role = meta.get("role", "human")
                    if role == "ai":
                        messages.append(AIMessage(content=mem.content))
                    else:
                        messages.append(HumanMessage(content=mem.content))
                return {self._memory_key: messages}
            except ImportError:
                pass

        # Default: return as formatted string
        formatted = "\n".join(f"- {m.content}" for m in memories)
        return {self._memory_key: formatted}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store the conversation turn in Synapse."""
        if self._store_inputs:
            input_key = self._input_key or next(iter(inputs), None)
            if input_key and input_key in inputs:
                value = inputs[input_key]
                if isinstance(value, str) and value.strip():
                    self.synapse.remember(
                        value,
                        memory_type="observation",
                        metadata={"role": "human", "source": "langchain"},
                    )

        if self._store_outputs:
            output_key = self._output_key or next(iter(outputs), None)
            if output_key and output_key in outputs:
                value = outputs[output_key]
                if isinstance(value, str) and value.strip():
                    self.synapse.remember(
                        value,
                        memory_type="observation",
                        metadata={"role": "ai", "source": "langchain"},
                    )

    def clear(self) -> None:
        """Clear all memories (caution: this is destructive)."""
        # Recall all and forget each
        all_memories = self.synapse.recall("", limit=10000)
        for mem in all_memories:
            self.synapse.forget(mem.id)
