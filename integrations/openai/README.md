# Synapse × OpenAI / ChatGPT

**Persistent memory for GPT conversations.** Your data stays local — only inference calls go to OpenAI.

## Install

```bash
pip install synapse-ai-memory openai
```

## Two Approaches

### 1. Automatic Memory (`SynapseGPTMemory`)

```python
from synapse import Synapse
from synapse.integrations.openai import SynapseGPTMemory

syn = Synapse("./gpt_memory")
gpt = SynapseGPTMemory(synapse=syn)

# Session 1
gpt.chat("I'm Alex, I prefer Python")

# Session 2 (remembers!)
response = gpt.chat("What language do I prefer?")
```

### 2. Function Calling (`synapse_functions`)

Let GPT decide when to remember/recall:

```python
from synapse.integrations.openai import synapse_functions, handle_synapse_function

response = client.chat.completions.create(
    model="gpt-4o",
    tools=synapse_functions(),
    messages=[...],
)

# Handle function calls
for tc in response.choices[0].message.tool_calls or []:
    result = handle_synapse_function(syn, tc.function.name, tc.function.arguments)
```

Or use the all-in-one helper:

```python
from synapse.integrations.openai.tool import run_with_functions

response = run_with_functions(syn, [
    {"role": "user", "content": "What do you remember about me?"}
])
```

## Functions Provided

| Function | Description |
|----------|-------------|
| `remember` | Save facts, preferences, events to local memory |
| `recall` | Semantic search across all memories |
| `forget` | Remove outdated or unwanted memories |

## Run the example

```bash
export OPENAI_API_KEY=sk-...
python example.py
```
