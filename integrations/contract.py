"""Connector contract definitions and validation utilities.

This module defines a lightweight contract schema for "connectors" exposed by this
repository (MCP clients + SDK integrations). The contract is intentionally minimal
and pragmatic: enough to drive UX and onboarding surfaces without forcing deep
rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import re


REQUIRED_COMMANDS = ("install", "verify", "test", "repair", "uninstall", "doctor_checks")
ALLOWED_TYPES = {"client", "library", "agent", "service"}
REQUIRED_CONNECTOR_FIELDS = {"id", "type", "commands", "capabilities", "tier", "label"}
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,40}$")


@dataclass(frozen=True)
class ConnectorContract:
    """Normalized metadata for a single integration connector."""

    id: str
    type: str
    label: str
    tier: str
    commands: Dict[str, bool]
    capabilities: List[str]
    example_prompt: str
    doctor_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "tier": self.tier,
            "commands": dict(self.commands),
            "capabilities": list(self.capabilities),
            "doctor_checks": list(self.doctor_checks),
            "example_prompt": self.example_prompt,
        }


# Tier-0 (first-class): Claude, Cursor, OpenAI, LangChain, LangGraph
# Tier-1 (launch window): Windsurf, Continue, CrewAI, OpenClaw
# NOTE: Client integrations (Claude/Cursor/Windsurf/Continue/OpenClaw) are currently
# partially implemented; framework integrations are SDK/API-facing entry points.
CONNECTOR_REGISTRY: Sequence[ConnectorContract] = [
    ConnectorContract(
        id="claude",
        type="client",
        label="Claude Desktop",
        tier="tier-0",
        commands={
            "install": True,
            "verify": True,
            "test": True,
            "repair": True,
            "uninstall": True,
            "doctor_checks": True,
        },
        capabilities=["mcp", "install", "verify", "repair", "uninstall"],
        example_prompt="Add Synapse MCP to Claude Desktop and verify configuration",
        doctor_checks=["config_path_exists", "synapse_entry_present"],
    ),
    ConnectorContract(
        id="cursor",
        type="client",
        label="Cursor",
        tier="tier-0",
        commands={
            "install": True,
            "verify": True,
            "test": True,
            "repair": True,
            "uninstall": True,
            "doctor_checks": True,
        },
        capabilities=["mcp", "install", "verify", "repair", "uninstall"],
        example_prompt="Install Synapse for Cursor and verify the MCP entry in its config",
        doctor_checks=["config_path_exists", "synapse_entry_present"],
    ),
    ConnectorContract(
        id="openai",
        type="library",
        label="OpenAI/ChatGPT",
        tier="tier-0",
        commands={
            "install": False,
            "verify": False,
            "test": False,
            "repair": False,
            "uninstall": False,
            "doctor_checks": False,
        },
        capabilities=["openai.function_tooling", "function_memory", "chat_tools"],
        example_prompt="Add remember/recall tool handlers in your OpenAI chat loop",
        doctor_checks=[],
    ),
    ConnectorContract(
        id="langchain",
        type="library",
        label="LangChain",
        tier="tier-0",
        commands={
            "install": False,
            "verify": False,
            "test": False,
            "repair": False,
            "uninstall": False,
            "doctor_checks": False,
        },
        capabilities=["memory", "chat_history", "chat_vector"],
        example_prompt="Drop in SynapseMemory/SynapseChatMessageHistory in a LangChain graph",
        doctor_checks=[],
    ),
    ConnectorContract(
        id="langgraph",
        type="library",
        label="LangGraph",
        tier="tier-0",
        commands={
            "install": False,
            "verify": False,
            "test": False,
            "repair": False,
            "uninstall": False,
            "doctor_checks": False,
        },
        capabilities=["checkpointer", "memory_store", "stateful_memory"],
        example_prompt="Use SynapseCheckpointer and SynapseMemoryStore in LangGraph state flow",
        doctor_checks=[],
    ),
    ConnectorContract(
        id="windsurf",
        type="client",
        label="Windsurf",
        tier="tier-1",
        commands={
            "install": True,
            "verify": True,
            "test": True,
            "repair": True,
            "uninstall": True,
            "doctor_checks": True,
        },
        capabilities=["mcp", "install", "verify", "repair", "uninstall"],
        example_prompt="Install Synapse for Windsurf and validate `mcpServers.synapse` config",
        doctor_checks=["config_path_exists", "synapse_entry_present"],
    ),
    ConnectorContract(
        id="continue",
        type="client",
        label="Continue",
        tier="tier-1",
        commands={
            "install": True,
            "verify": True,
            "test": True,
            "repair": True,
            "uninstall": True,
            "doctor_checks": True,
        },
        capabilities=["mcp", "install", "verify", "repair", "uninstall"],
        example_prompt="Install Synapse for Continue and check modelContextProtocolServers entries",
        doctor_checks=["config_path_exists", "synapse_entry_present"],
    ),
    ConnectorContract(
        id="crewai",
        type="library",
        label="CrewAI",
        tier="tier-1",
        commands={
            "install": False,
            "verify": False,
            "test": False,
            "repair": False,
            "uninstall": False,
            "doctor_checks": False,
        },
        capabilities=["agent_memory", "crew_tools"],
        example_prompt="Import SynapseCrewMemory in your CrewAI agent initialization",
        doctor_checks=[],
    ),
    ConnectorContract(
        id="openclaw",
        type="client",
        label="OpenClaw",
        tier="tier-1",
        commands={
            "install": True,
            "verify": True,
            "test": True,
            "repair": True,
            "uninstall": True,
            "doctor_checks": True,
        },
        capabilities=["mcp", "skill_install", "repair", "uninstall"],
        example_prompt="Install Synapse skill into OpenClaw workspace and verify manifest",
        doctor_checks=["workspace_exists", "synapse_entry_present"],
    ),
]


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, ConnectorContract):
        return obj.to_dict()
    if isinstance(obj, Mapping):
        return obj
    raise TypeError("Connector entry must be a ConnectorContract or mapping")


def validate_connector_contract(spec: Mapping[str, Any]) -> ConnectorContract:
    if not isinstance(spec, Mapping):
        raise TypeError("Connector contract must be a mapping")

    missing = REQUIRED_CONNECTOR_FIELDS - set(spec.keys())
    if missing:
        raise ValueError(f"Connector contract missing required fields: {sorted(missing)}")

    conn_id = spec["id"]
    if not isinstance(conn_id, str) or not _ID_PATTERN.fullmatch(conn_id):
        raise ValueError(f"Connector id must be lowercase slug: {conn_id!r}")

    conn_type = spec["type"]
    if conn_type not in ALLOWED_TYPES:
        raise ValueError(f"Invalid connector type {conn_type!r}")

    if not isinstance(spec["label"], str) or not spec["label"]:
        raise ValueError("Connector label must be a non-empty string")

    if not isinstance(spec["tier"], str) or not spec["tier"]:
        raise ValueError("Connector tier must be a non-empty string")

    commands = spec["commands"]
    if not isinstance(commands, Mapping):
        raise ValueError("Connector commands must be a mapping")
    missing_cmds = set(REQUIRED_COMMANDS) - set(commands.keys())
    if missing_cmds:
        raise ValueError(f"Connector commands missing: {sorted(missing_cmds)}")
    for name in REQUIRED_COMMANDS:
        if not isinstance(commands.get(name), bool):
            raise ValueError(f"Connector command {name!r} must be boolean")

    capabilities = spec["capabilities"]
    if not isinstance(capabilities, (list, tuple)):
        raise ValueError("Connector capabilities must be a list")
    for capability in capabilities:
        if not isinstance(capability, str) or not capability:
            raise ValueError("Capability must be a non-empty string")

    doctor_checks = spec["doctor_checks"]
    if not isinstance(doctor_checks, (list, tuple)):
        raise ValueError("doctor_checks must be a list")
    for check in doctor_checks:
        if not isinstance(check, str) or not check:
            raise ValueError("doctor_checks entries must be non-empty strings")

    example_prompt = spec["example_prompt"]
    if not isinstance(example_prompt, str) or not example_prompt.strip():
        raise ValueError("example_prompt must be a non-empty string")

    return ConnectorContract(
        id=conn_id,
        type=conn_type,
        label=spec["label"],
        tier=spec["tier"],
        commands=dict(commands),
        capabilities=list(capabilities),
        example_prompt=example_prompt,
        doctor_checks=list(doctor_checks),
    )


def validate_connector_contracts(specs: Iterable[Any]) -> list[ConnectorContract]:
    validated = [validate_connector_contract(_as_mapping(spec)) for spec in specs]

    ids = [entry.id for entry in validated]
    duplicates = [conn_id for conn_id in set(ids) if ids.count(conn_id) > 1]
    if duplicates:
        raise ValueError(f"Duplicate connector ids: {sorted(set(duplicates))}")

    return validated


def contracts() -> list[ConnectorContract]:
    return validate_connector_contracts(CONNECTOR_REGISTRY)


def by_id(connector_id: str) -> ConnectorContract | None:
    for contract in contracts():
        if contract.id == connector_id:
            return contract
    return None
