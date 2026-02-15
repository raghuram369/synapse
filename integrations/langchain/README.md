# Synapse × LangChain

**Privacy-first memory for LangChain chains and agents.** Your data stays local. No cloud vector DB. No API calls for storage.

## Install

```bash
pip install synapse-ai-memory langchain-core
```

## Components

### `SynapseMemory` — Drop-in chain memory

```python
from synapse import Synapse
from synapse.integrations.langchain import SynapseMemory

syn = Synapse("./agent_memory")
memory = SynapseMemory(synapse=syn, memory_key="context")

# Use with any chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

Unlike `ConversationBufferMemory`, Synapse recalls *semantically relevant* memories — not just the last N messages.

### `SynapseChatMessageHistory` — Persistent chat history

```python
from synapse.integrations.langchain import SynapseChatMessageHistory

history = SynapseChatMessageHistory(synapse=syn, session_id="user-123")
history.add_user_message("I love Italian food")

# Semantic search across all history
results = history.search("dietary preferences")
```

### `SynapseRetriever` — Local RAG retriever

```python
from synapse.integrations.langchain import SynapseRetriever

retriever = SynapseRetriever(synapse=syn, k=5)

# Use in LCEL chains
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm
)
```

Combines BM25 + concept graphs + optional local embeddings (Ollama). No external vector DB needed.

## Why Synapse over Chroma/Pinecone/Weaviate?

| Feature | Synapse | Cloud Vector DBs |
|---------|---------|-------------------|
| Privacy | 100% local | Data leaves your machine |
| Dependencies | Zero | SDK + API keys |
| Search | BM25 + concepts + embeddings | Embeddings only |
| Temporal decay | Built-in | Manual |
| Federation | Peer-to-peer sync | Vendor lock-in |
| Cost | Free | Per-query pricing |

## Run the example

```bash
python example.py
```
