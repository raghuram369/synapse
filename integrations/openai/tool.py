"""
Synapse as OpenAI function calls — let GPT remember and recall autonomously.

Defines function schemas for OpenAI's function calling API.

Requirements:
    pip install openai synapse-ai-memory
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from synapse import Synapse


# ── Function definitions for OpenAI's function calling API ────

SYNAPSE_REMEMBER_FUNCTION = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save important information to persistent memory. Use when the user "
            "shares facts, preferences, or context to remember across sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be specific and concise.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "event", "skill", "observation"],
                    "description": "Category: fact, preference, event, skill, or observation.",
                },
            },
            "required": ["content"],
        },
    },
}

SYNAPSE_RECALL_FUNCTION = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search persistent memory for relevant information about the user "
            "or past conversations. Returns semantically relevant results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

SYNAPSE_FORGET_FUNCTION = {
    "type": "function",
    "function": {
        "name": "forget",
        "description": "Remove memories matching a query. Use when information is outdated or user asks to forget.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to forget.",
                },
            },
            "required": ["query"],
        },
    },
}


def synapse_functions() -> List[Dict[str, Any]]:
    """Return all Synapse function definitions for OpenAI's API.

    Example::

        from synapse.integrations.openai import synapse_functions

        response = client.chat.completions.create(
            model="gpt-4o",
            tools=synapse_functions(),
            messages=[...],
        )
    """
    return [SYNAPSE_REMEMBER_FUNCTION, SYNAPSE_RECALL_FUNCTION, SYNAPSE_FORGET_FUNCTION]


def handle_synapse_function(
    synapse: Synapse,
    function_name: str,
    arguments: str | Dict[str, Any],
) -> str:
    """Handle a Synapse function call from GPT.

    Args:
        synapse: The Synapse instance.
        function_name: "remember", "recall", or "forget".
        arguments: JSON string or dict of function arguments.

    Returns:
        Result string to send back as tool response.
    """
    if isinstance(arguments, str):
        args = json.loads(arguments)
    else:
        args = arguments

    if function_name == "remember":
        content = args["content"]
        memory_type = args.get("memory_type", "fact")
        mem = synapse.remember(content, memory_type=memory_type, metadata={"source": "openai_tool"})
        return json.dumps({"status": "saved", "content": content, "type": memory_type, "id": mem.id})

    elif function_name == "recall":
        query = args["query"]
        limit = args.get("limit", 5)
        memories = synapse.recall(query, limit=limit)
        results = [{"content": m.content, "type": m.memory_type} for m in memories]
        return json.dumps({"memories": results, "count": len(results)})

    elif function_name == "forget":
        query = args["query"]
        memories = synapse.recall(query, limit=50)
        count = 0
        for mem in memories:
            synapse.forget(mem.id)
            count += 1
        return json.dumps({"forgotten": count, "query": query})

    return json.dumps({"error": f"Unknown function: {function_name}"})


def run_with_functions(
    synapse: Synapse,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    system: str = "You are a helpful assistant with persistent memory. Use remember/recall tools to maintain context across conversations.",
    api_key: Optional[str] = None,
    max_rounds: int = 5,
) -> str:
    """Run a GPT conversation with automatic Synapse function handling.

    Handles the full function-calling loop.

    Example::

        from synapse import Synapse
        from synapse.integrations.openai.tool import run_with_functions

        syn = Synapse("./memory")
        response = run_with_functions(syn, [
            {"role": "user", "content": "Remember I like Python, then tell me what you know"}
        ])
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai is required. Install with: pip install openai")

    client = openai.OpenAI(api_key=api_key)
    tools = synapse_functions()

    full_messages = [{"role": "system", "content": system}] + messages

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=tools,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            full_messages.append(choice.message)
            for tc in choice.message.tool_calls:
                result = handle_synapse_function(synapse, tc.function.name, tc.function.arguments)
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return choice.message.content or ""

    return "Max function call rounds reached."
