# OpenAI Starter Kit (Memory-Backed Function Calling)

Use this as the starting point for an OpenAI / ChatGPT starter repo.

## Repo template pointers

- Repo scaffold (starter): _placeholder_ `https://github.com/synapse-ai-memory/openai-starter`
- In-repo implementation example: `integrations/openai/example.py`
- Helper APIs: `integrations/openai/memory.py`, `integrations/openai/tool.py`
- README guide source: `integrations/openai/README.md`

## What to copy into a starter repo

- `requirements.txt` with `synapse-ai-memory`, `openai`
- `app.py` that:
  - creates a persistent `Synapse` instance
  - injects `SynapseGPTMemory` or `synapse_functions`
  - persists memory path and policy defaults
- `.env.example` containing `OPENAI_API_KEY`
- deployment notes for one-command bootstrap and local secrets hygiene

## Setup flow (template)

```bash
python -m venv .venv
source .venv/bin/activate
pip install synapse-ai-memory openai
export OPENAI_API_KEY=...
python app.py
```
