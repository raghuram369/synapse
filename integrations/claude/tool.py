"""
Synapse as a Claude tool — let Claude remember and recall via tool_use.

Defines tool schemas for Anthropic's tool_use API. Claude can decide
when to save and retrieve memories autonomously.

Requirements:
    pip install anthropic synapse-ai-memory
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from synapse import Synapse


# ── Tool definitions for Anthropic's tool_use API ─────────────

SYNAPSE_REMEMBER_TOOL = {
    "name": "remember",
    "description": (
        "Save important information to persistent memory. Use this when the user "
        "shares facts, preferences, or important context you should remember for "
        "future conversations. Memory persists across sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember. Be specific and concise.",
            },
            "memory_type": {
                "type": "string",
                "enum": ["fact", "preference", "event", "skill", "observation"],
                "description": "Category of memory. Use 'fact' for general info, 'preference' for likes/dislikes, 'event' for things that happened, 'skill' for abilities.",
                "default": "fact",
            },
        },
        "required": ["content"],
    },
}

SYNAPSE_RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Search persistent memory for relevant information. Use this when you need "
        "to check if you know something about the user, their preferences, or past "
        "conversations. Returns semantically relevant memories."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Can be a question or topic.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

SYNAPSE_FORGET_TOOL = {
    "name": "forget",
    "description": (
        "Remove memories matching a query. Use when the user asks you to forget "
        "something or when information is outdated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to forget. Memories matching this will be removed.",
            },
        },
        "required": ["query"],
    },
}


def synapse_tools() -> List[Dict[str, Any]]:
    """Return all Synapse tool definitions for Claude's tool_use API.

    Example::

        import anthropic
        from synapse.integrations.claude import synapse_tools

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            tools=synapse_tools(),
            messages=[{"role": "user", "content": "Remember that I like Python"}],
        )
    """
    return [SYNAPSE_REMEMBER_TOOL, SYNAPSE_RECALL_TOOL, SYNAPSE_FORGET_TOOL]


def handle_synapse_tool(
    synapse: Synapse,
    tool_name: str,
    tool_input: Dict[str, Any],
) -> str:
    """Handle a Synapse tool call from Claude.

    Args:
        synapse: The Synapse instance to use.
        tool_name: Name of the tool ("remember", "recall", "forget").
        tool_input: The tool input from Claude.

    Returns:
        A string result to send back as tool_result.

    Example::

        from synapse import Synapse
        from synapse.integrations.claude import synapse_tools, handle_synapse_tool

        syn = Synapse("./memory")

        # In your tool_use loop:
        for block in response.content:
            if block.type == "tool_use":
                result = handle_synapse_tool(syn, block.name, block.input)
    """
    if tool_name == "remember":
        content = tool_input["content"]
        memory_type = tool_input.get("memory_type", "fact")
        mem = synapse.remember(content, memory_type=memory_type, metadata={"source": "claude_tool"})
        return f"Remembered: '{content}' (type: {memory_type}, id: {mem.id})"

    elif tool_name == "recall":
        query = tool_input["query"]
        limit = tool_input.get("limit", 5)
        memories = synapse.recall(query, limit=limit)
        if not memories:
            return "No relevant memories found."
        lines = []
        for mem in memories:
            lines.append(f"- [{mem.memory_type}] {mem.content}")
        return f"Found {len(memories)} memories:\n" + "\n".join(lines)

    elif tool_name == "forget":
        query = tool_input["query"]
        memories = synapse.recall(query, limit=50)
        count = 0
        for mem in memories:
            synapse.forget(mem.id)
            count += 1
        return f"Forgot {count} memories matching '{query}'."

    else:
        return f"Unknown tool: {tool_name}"


def run_with_tools(
    synapse: Synapse,
    messages: List[Dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    system: str = "You are a helpful assistant with persistent memory. Use the remember/recall tools to maintain context across conversations.",
    api_key: Optional[str] = None,
    max_tool_rounds: int = 5,
) -> str:
    """Run a Claude conversation with automatic Synapse tool handling.

    Handles the full tool_use loop — Claude calls tools, we execute them,
    send results back, until Claude produces a final text response.

    Example::

        from synapse import Synapse
        from synapse.integrations.claude.tool import run_with_tools

        syn = Synapse("./memory")
        response = run_with_tools(
            syn,
            [{"role": "user", "content": "What do you remember about me?"}],
        )
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is required. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    tools = synapse_tools()

    for _ in range(max_tool_rounds):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Check if there are tool calls
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            # Final text response
            text_blocks = [b for b in response.content if b.type == "text"]
            return text_blocks[0].text if text_blocks else ""

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            result = handle_synapse_tool(synapse, tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return "Max tool rounds reached."
