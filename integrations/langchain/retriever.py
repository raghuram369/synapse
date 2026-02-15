"""
SynapseRetriever — LangChain BaseRetriever backed by Synapse.

Use Synapse as a retriever in any LangChain RAG pipeline.
Privacy-first: all retrieval is local. No vector DB API calls.

Requirements:
    pip install langchain-core synapse-ai-memory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from synapse import Synapse


class SynapseRetriever:
    """LangChain-compatible retriever using Synapse's semantic recall.

    Synapse combines BM25, concept graphs, and optional local embeddings
    for retrieval — all running locally with zero external API calls.

    Example::

        from synapse import Synapse
        from synapse.integrations.langchain import SynapseRetriever

        syn = Synapse("./knowledge_base")
        syn.remember("Python was created by Guido van Rossum")
        syn.remember("Rust was created by Graydon Hoare at Mozilla")

        retriever = SynapseRetriever(synapse=syn, k=5)

        # Use in a RAG chain
        from langchain_core.runnables import RunnablePassthrough
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
    """

    _base_resolved = False

    def __new__(cls, *args, **kwargs):
        if not cls._base_resolved:
            try:
                from langchain_core.retrievers import BaseRetriever
                cls.__bases__ = (BaseRetriever,) + tuple(
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
        k: int = 5,
        memory_type: Optional[str] = None,
        min_strength: float = 0.01,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        self.synapse = synapse or Synapse(path)
        self._k = k
        self._memory_type = memory_type
        self._min_strength = min_strength
        self._metadata_filter = metadata_filter or {}

        try:
            super().__init__()
        except TypeError:
            pass

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Any]:
        """Retrieve relevant documents from Synapse."""
        memories = self.synapse.recall(
            query,
            limit=self._k,
            memory_type=self._memory_type,
            min_strength=self._min_strength,
        )

        # Apply metadata filters
        if self._metadata_filter:
            memories = [
                m for m in memories
                if all(
                    (m.metadata or {}).get(k) == v
                    for k, v in self._metadata_filter.items()
                )
            ]

        try:
            from langchain_core.documents import Document
            return [
                Document(
                    page_content=mem.content,
                    metadata={
                        "memory_id": mem.id,
                        "memory_type": mem.memory_type,
                        "strength": mem.effective_strength,
                        "created_at": mem.created_at,
                        **(mem.metadata or {}),
                    },
                )
                for mem in memories
            ]
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Any]:
        """Async retrieval (delegates to sync — Synapse is local and fast)."""
        return self._get_relevant_documents(query, run_manager=run_manager)

    # Convenience: make it callable for LCEL chains
    def invoke(self, input: str, config=None) -> List[Any]:
        return self._get_relevant_documents(input)
