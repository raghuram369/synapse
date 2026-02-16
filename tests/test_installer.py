import argparse
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, mock_open, patch

import cli
import installer


class TestInstaller(unittest.TestCase):
    def _capture_output(self, fn, *args, **kwargs):
        out = io.StringIO()
        with redirect_stdout(out):
            fn(*args, **kwargs)
        return out.getvalue()

    def _parse_written_json(self, mock_file):
        written = "".join(call.args[0] for call in mock_file.write.call_args_list)
        return json.loads(written)

    def test_client_installer_targets(self):
        self.assertEqual(
            set(installer.ClientInstaller.TARGETS.keys()),
            {"claude", "openclaw", "nanoclaw"},
        )

    def test_install_claude_creates_config_when_missing(self):
        db_path = "/tmp/synapse_db"
        config_path = "/tmp/claude_desktop_config.json"
        write_handle = mock_open()()
        with patch("installer._claude_config_path", return_value=config_path), patch(
            "installer.os.path.exists",
            side_effect=lambda _: False,
        ), patch("installer._backup_file") as backup, patch(
            "installer.os.makedirs"
        ), patch("builtins.open", return_value=write_handle):
            out = self._capture_output(installer.install_claude, db_path)

        payload = self._parse_written_json(write_handle)
        self.assertEqual(payload["mcpServers"]["synapse"]["command"], "python3")
        self.assertEqual(
            payload["mcpServers"]["synapse"]["args"][0],
            os.path.abspath(os.path.join(os.path.dirname(installer.__file__), "mcp_server.py")),
        )
        self.assertEqual(payload["mcpServers"]["synapse"]["args"][2], db_path)
        self.assertIn("Synapse installed for Claude Desktop. Restart Claude to activate.", out)
        backup.assert_not_called()

    def test_install_claude_backs_up_existing_config(self):
        db_path = "/tmp/synapse_db"
        config_path = "/tmp/claude_desktop_config.json"
        read_handle = mock_open(
            read_data=json.dumps(
                {
                    "mcpServers": {
                        "synapse": {
                            "command": "python",
                            "args": ["/old/path/mcp_server.py", "--db", "/old/db"],
                        },
                    }
                }
            )
        )()
        write_handle = mock_open()()

        def open_side_effect(path, mode="r", *_, **__):
            return read_handle if "r" in mode else write_handle

        with patch("installer._claude_config_path", return_value=config_path), patch(
            "installer.os.path.exists", return_value=True
        ), patch(
            "installer.shutil.copy2"
        ) as copy2, patch("builtins.open", side_effect=open_side_effect):
            out = self._capture_output(installer.install_claude, db_path)

        copy2.assert_called_once_with(config_path, f"{config_path}.backup")
        payload = self._parse_written_json(write_handle)
        self.assertEqual(payload["mcpServers"]["synapse"]["args"][2], db_path)
        self.assertIn("Synapse installed for Claude Desktop. Restart Claude to activate.", out)

    def test_install_claude_replaces_non_dict_mcp_servers(self):
        config_path = "/tmp/claude_desktop_config.json"
        read_handle = mock_open(
            read_data=json.dumps({"mcpServers": "invalid"})
        )()
        write_handle = mock_open()()

        def open_side_effect(path, mode="r", *_, **__):
            return read_handle if "r" in mode else write_handle

        with patch("installer._claude_config_path", return_value=config_path), patch(
            "installer.os.path.exists", return_value=True
        ), patch("builtins.open", side_effect=open_side_effect), patch(
            "installer.shutil.copy2"
        ) as copy2:
            installer.install_claude("/tmp/new_db")

        copy2.assert_called_once_with(config_path, f"{config_path}.backup")
        payload = self._parse_written_json(write_handle)
        self.assertIsInstance(payload["mcpServers"], dict)
        self.assertEqual(
            payload["mcpServers"]["synapse"]["args"][0],
            os.path.abspath(os.path.join(os.path.dirname(installer.__file__), "mcp_server.py")),
        )

    def test_install_openclaw_writes_skill_files(self):
        root = "/tmp/openclaw_workspace/skills"
        with patch("installer._openclaw_workspace_root", return_value=root), patch(
            "installer._write_text"
        ) as write_text:
            out = self._capture_output(installer.install_openclaw, "/tmp/db")

        self.assertIn("Synapse skill installed for OpenClaw.", out)
        written = [call.args[0] for call in write_text.call_args_list]
        self.assertIn(f"{root}/synapse/SKILL.md", written)
        self.assertIn(f"{root}/synapse/manifest.json", written)
        self.assertIn(f"{root}/synapse/setup.sh", written)

    def test_install_openclaw_manifest_has_permissions(self):
        with patch("installer._openclaw_workspace_root", return_value="/tmp/openclaw_workspace/skills"), patch(
            "installer._write_text"
        ) as write_text:
            installer.install_openclaw("/tmp/db")

        manifest_call = [
            call for call in write_text.call_args_list
            if str(call.args[0]).endswith("manifest.json")
        ][0]
        manifest = json.loads(manifest_call.args[1])
        self.assertEqual(manifest["name"], "synapse-ai-memory")
        self.assertIn("permissions", manifest)
        self.assertIn("filesystem", manifest["permissions"])

    def test_install_nanoclaw_writes_skill_files(self):
        root = "/tmp/nanoclaw_workspace/skills"
        with patch("installer._nanoclaw_workspace_root", return_value=root), patch(
            "installer._write_text"
        ) as write_text:
            out = self._capture_output(installer.install_nanoclaw, "/tmp/db")

        self.assertIn("Synapse skill installed for NanoClaw.", out)
        written = [call.args[0] for call in write_text.call_args_list]
        self.assertIn(f"{root}/synapse/SKILL.md", written)
        self.assertIn(f"{root}/synapse/manifest.json", written)
        self.assertIn(f"{root}/synapse/setup.sh", written)

    def test_main_install_list(self):
        with patch.object(sys, "argv", ["synapse", "install", "--list"]), redirect_stdout(
            io.StringIO()
        ) as out:
            cli.main()

        output = out.getvalue()
        self.assertIn("claude", output)
        self.assertIn("openclaw", output)
        self.assertIn("nanoclaw", output)

    def test_main_install_dispatches_selected_target(self):
        mocked = MagicMock()
        with patch.object(sys, "argv", ["synapse", "install", "claude", "--db", "/tmp/db"]), patch(
            "installer.ClientInstaller.TARGETS",
            {"claude": mocked},
        ):
            cli.main()
        mocked.assert_called_once_with("/tmp/db")

    def test_main_install_unknown_target(self):
        with patch.object(sys, "argv", ["synapse", "install", "bad-target"]):
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()) as out:
                    cli.main()
            output = out.getvalue()
        self.assertIn("Unknown install target: bad-target", output)
