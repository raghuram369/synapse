import argparse
import json
import unittest

import cli

from integrations import contract as connector_contract


class TestConnectorContractSchema(unittest.TestCase):
    def test_connectors_validate(self):
        contracts = connector_contract.validate_connector_contracts(connector_contract.CONNECTOR_REGISTRY)
        self.assertGreaterEqual(len(contracts), 1)

        required_keys = set(connector_contract.REQUIRED_COMMANDS)
        for spec in contracts:
            self.assertIsNotNone(spec.id)
            self.assertIsInstance(spec.id, str)
            self.assertIsInstance(spec.commands, dict)
            self.assertTrue(required_keys.issubset(spec.commands.keys()))
            for key in required_keys:
                self.assertIn(key, spec.commands)
                self.assertIsInstance(spec.commands[key], bool)
            self.assertIsInstance(spec.capabilities, list)
            self.assertIsInstance(spec.doctor_checks, list)
            self.assertIsInstance(spec.example_prompt, str)
            self.assertTrue(spec.example_prompt.strip())
            self.assertIn(spec.type, connector_contract.ALLOWED_TYPES)

    def test_required_built_in_connectors_present(self):
        contract_map = {spec.id: spec for spec in connector_contract.contracts()}
        for connector_id in [
            "claude",
            "cursor",
            "openai",
            "langchain",
            "langgraph",
            "windsurf",
            "continue",
            "crewai",
            "openclaw",
        ]:
            self.assertIn(connector_id, contract_map, f"Missing connector contract: {connector_id}")
            self.assertEqual(contract_map[connector_id].id, connector_id)
            self.assertIsInstance(contract_map[connector_id].capabilities, list)

    def test_integrations_cli_list_includes_contract_payload(self):
        from contextlib import redirect_stdout
        import io

        with unittest.mock.patch.object(
            cli,
            "_detect_client_installs",
            return_value=[
                ("claude", "Claude Desktop", True, True),
                ("cursor", "Cursor", True, False),
                ("windsurf", "Windsurf", False, False),
                ("continue", "Continue", False, False),
                ("openclaw", "OpenClaw", False, False),
            ],
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.cmd_integrations(
                    argparse.Namespace(integrations_action="list", db="/tmp/db", json=True),
                )
            payload = json.loads(buf.getvalue())

        ids = {row["id"] for row in payload["integrations"]}
        self.assertIn("openai", ids)
        self.assertIn("langchain", ids)
        self.assertIn("langgraph", ids)
        self.assertIn("crewai", ids)

        row = next(item for item in payload["integrations"] if item["id"] == "openai")
        self.assertIn("commands", row)
        self.assertIn("capabilities", row)
        self.assertIn("example_prompt", row)
        for command in connector_contract.REQUIRED_COMMANDS:
            self.assertIn(command, row["commands"])
