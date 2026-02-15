# Synapse × Claude

**Persistent memory for Claude conversations.** Your data stays local — only inference calls go to Anthropic.

## Install

```bash
pip install synapse-ai-memory anthropic
```

## Two Approaches

### 1. Automatic Memory (`SynapseClaudeMemory`)

Wraps the Anthropic API to automatically inject relevant context and save exchanges:

```python
from synapse import Synapse
from synapse.integrations.claude import SynapseClaudeMemory

syn = Synapse("./claude_memory")
claude = SynapseClaudeMemory(synapse=syn)

# Session 1
claude.chat("My name is Alex, I'm building a trading bot in Python")

# Session 2 (days later — Claude remembers!)
response = claude.chat("What was I working on?")
# → "You mentioned you're building a trading bot in Python!"
```

### 2. Tool-Based Memory (`synapse_tools`)

Let Claude decide when to remember/recall using tool_use:

```python
from synapse.integrations.claude import synapse_tools, handle_synapse_tool

# Register tools
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=synapse_tools(),
    messages=[{"role": "user", "content": "Remember I prefer dark mode"}],
)

# Handle tool calls
for block in response.content:
    if block.type == "tool_use":
        result = handle_synapse_tool(syn, block.name, block.input)
```

Or use the all-in-one helper:

```python
from synapse.integrations.claude.tool import run_with_tools

response = run_with_tools(syn, [
    {"role": "user", "content": "What do you remember about me?"}
])
```

## Tools Provided

| Tool | Description |
|------|-------------|
| `remember` | Save facts, preferences, events to local memory |
| `recall` | Semantic search across all memories |
| `forget` | Remove outdated or unwanted memories |

## Run the example

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python example.py
```
