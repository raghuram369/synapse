import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from importers import ImportReport, MemoryImporter


class _RecordingSynapse:
    def __init__(self):
        self.remember_calls = []
        self.flushed = False
        self.closed = False

    def remember(self, content, **kwargs):
        record = {"content": content, "metadata": kwargs.get("metadata", {})}
        self.remember_calls.append(record)
        return SimpleNamespace(id=len(self.remember_calls), metadata=record["metadata"], content=content)

    def should_sleep(self):
        return False

    def close(self):
        self.closed = True

    def flush(self):
        self.flushed = True


class TestMemoryImporter(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="synapse-importers-")
        self.synapse = _RecordingSynapse()
        self.importer = MemoryImporter(self.synapse)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write(self, rel_path, content, *, mode="w"):
        path = os.path.join(self.tempdir.name, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _jsonl_line(self, payload):
        return json.dumps(payload) + "\n"

    def test_from_jsonl_imports_lines_and_metadata(self):
        path = self._write(
            "notes.jsonl",
            self._jsonl_line({"text": "Project kickoff", "topic": "plan", "source": "cli"})
            + self._jsonl_line({"text": "Daily standup", "topic": "standup", "source": "cli"}),
        )
        report = self.importer.from_jsonl(path)

        self.assertEqual(report.imported, 2)
        self.assertEqual(len(self.synapse.remember_calls), 2)
        self.assertIn("source:jsonl", report.tags_created)
        self.assertEqual(self.synapse.remember_calls[0]["metadata"]["topic"], "plan")
        self.assertIn("jsonl", self.synapse.remember_calls[1]["metadata"]["tags"])

    def test_from_jsonl_counts_bad_rows(self):
        path = self._write(
            "bad.jsonl",
            self._jsonl_line({"text": "Valid message"})
            + "not-json\n"
            + self._jsonl_line({"text": ""})
            + self._jsonl_line({"body": "Missing field"}),
        )
        report = self.importer.from_jsonl(path)

        self.assertEqual(report.imported, 1)
        self.assertEqual(report.errors, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(len(self.synapse.remember_calls), 1)

    def test_from_jsonl_respects_metadata_fields(self):
        path = self._write(
            "filtered.jsonl",
            self._jsonl_line({"body": "Use importer", "source": "cli", "topic": "ops", "junk": "ignore"}),
        )
        report = self.importer.from_jsonl(path, text_field="body", metadata_fields=["source", "topic"])

        self.assertEqual(report.imported, 1)
        metadata = self.synapse.remember_calls[0]["metadata"]
        self.assertNotIn("junk", metadata)
        self.assertEqual(metadata["source"], "cli")
        self.assertEqual(metadata["topic"], "ops")

    def test_from_csv_imports_rows(self):
        path = self._write(
            "rows.csv",
            "text,topic,owner\n"
            '"Buy groceries","home","alice"\n'
            '"","home","bob"\n',
        )
        report = self.importer.from_csv(path)

        self.assertEqual(report.imported, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(self.synapse.remember_calls[0]["content"], "Buy groceries")
        self.assertIn("source:csv", self.synapse.remember_calls[0]["metadata"]["tags"])
        self.assertEqual(self.synapse.remember_calls[0]["metadata"]["owner"], "alice")

    def test_from_markdown_folder_splits_by_headings(self):
        path = self._write(
            "notes/meeting.md",
            "# Agenda\n"
            "Plan release.\n\n"
            "# Risks\n"
            "API latency spike.\n",
        )
        report = self.importer.from_markdown_folder(self.tempdir.name, recursive=True)

        self.assertEqual(report.imported, 2)
        contents = [record["content"] for record in self.synapse.remember_calls]
        self.assertTrue(any("Agenda" in content for content in contents))
        self.assertTrue(any("Risks" in content for content in contents))
        self.assertIn("note:meeting", report.tags_created)

    def test_from_markdown_frontmatter_becomes_metadata(self):
        path = self._write(
            "notes/with_meta.md",
            "---\n"
            "title: Weekly Plan\n"
            "tags:\n"
            "  - planning\n"
            "  - sync\n"
            "author: Alice\n"
            "---\n"
            "Kick off sprint.\n"
            "\n"
            "Ship documentation.\n",
        )
        report = self.importer.from_markdown_folder(os.path.dirname(path), recursive=False)

        self.assertEqual(report.imported, 2)
        metadata = self.synapse.remember_calls[0]["metadata"]
        self.assertEqual(metadata["title"], "Weekly Plan")
        self.assertIn("planning", metadata["tags"])
        self.assertIn("note:with_meta", report.tags_created)

    def test_from_markdown_folder_respects_recursive_flag(self):
        self._write("root.md", "# One\nFirst.\n")
        self._write("deep/nested.md", "# Two\nSecond.\n")

        first = self.importer.from_markdown_folder(self.tempdir.name, recursive=False)
        self.assertEqual(first.imported, 1)

        second = self.importer.from_markdown_folder(self.tempdir.name, recursive=True)
        self.assertEqual(second.imported, 2)

    def test_from_chat_transcript_chatgpt_format(self):
        path = self._write(
            "chatgpt.json",
            json.dumps([
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "name": "Alice", "content": "Can you help me with my plan?"},
                {"role": "assistant", "content": "Absolutely, happy to help."},
                {"role": "assistant", "content": "ok"},
            ]),
        )
        report = self.importer.from_chat_transcript(path, format="chatgpt")

        self.assertEqual(report.imported, 2)
        first = self.synapse.remember_calls[0]
        self.assertEqual(first["metadata"]["source_format"], "chatgpt_export")
        self.assertEqual(first["metadata"]["participants"], ["Alice"])
        self.assertIn("participant:alice", first["metadata"]["tags"])
        self.assertIn("source:chatgpt_export", first["metadata"]["tags"])

    def test_from_chat_transcript_auto_detect_whatsapp_export(self):
        path = self._write(
            "chat.txt",
            "11/02/25, 10:11 PM - Zoe: Project kickoff at 10.\n"
            "11/02/25, 10:12 PM - Max: Great, I'll sync up.\n",
        )
        report = self.importer.from_chat_transcript(path, format="auto")

        self.assertEqual(report.imported, 2)
        self.assertIn("participant:zoe", self.synapse.remember_calls[0]["metadata"]["tags"])
        self.assertIn("source:whatsapp_export", self.synapse.remember_calls[0]["metadata"]["tags"])

    def test_from_clipboard_tags_and_classifies_url(self):
        with patch.object(MemoryImporter, "_read_clipboard_text", return_value="https://example.com/notes"):
            report = self.importer.from_clipboard()

        self.assertEqual(report.imported, 1)
        metadata = self.synapse.remember_calls[0]["metadata"]
        self.assertEqual(metadata["clipboard_type"], "url")
        self.assertIn("clipboard", metadata["tags"])
        self.assertIn("clipboard:url", metadata["tags"])

    def test_from_clipboard_tags_and_classifies_code(self):
        with patch.object(MemoryImporter, "_read_clipboard_text", return_value="def hello():\n    return True"):
            report = self.importer.from_clipboard()

        self.assertEqual(report.imported, 1)
        metadata = self.synapse.remember_calls[1]["metadata"]
        self.assertEqual(metadata["clipboard_type"], "code")
        self.assertIn("clipboard:code", metadata["tags"])


class TestImportCli(unittest.TestCase):
    def test_cli_import_chat_dispatches_to_memory_importer(self):
        from importers import ImportReport
        from cli import cmd_import

        fake_synapse = _RecordingSynapse()

        class FakeMemoryImporter:
            def __init__(self, _synapse):
                self.calls = []

            def from_chat_transcript(self, path, format="auto"):
                self.calls.append((path, format))
                return ImportReport(
                    imported=4,
                    skipped=1,
                    errors=0,
                    tags_created=["chat", "source:chatgpt_export"],
                    duration_ms=13.5,
                )

        output = io.StringIO()
        with patch("synapse.Synapse", return_value=fake_synapse), patch(
            "importers.MemoryImporter", FakeMemoryImporter
        ):
            with redirect_stdout(output):
                cmd_import(
                    argparse.Namespace(
                        db=":memory:",
                        input="chat",
                        path="/tmp/chat.json",
                        format="auto",
                        recursive=False,
                        text_field="text",
                        text_column="text",
                        no_dedup=False,
                        threshold=0.85,
                    )
                )

        text = output.getvalue()
        self.assertIn("✓ Imported from chat transcript source", text)
        self.assertIn("Imported: 4", text)
        self.assertIn("Tags created: chat, source:chatgpt_export", text)


if __name__ == "__main__":
    unittest.main()
