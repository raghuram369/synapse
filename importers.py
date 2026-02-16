"""High-level data import helpers for Synapse."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class ImportReport:
    imported: int = 0
    skipped: int = 0
    errors: int = 0
    tags_created: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


class MemoryImporter:
    """Import various source formats into a Synapse instance."""

    _WHATSAPP_LINE_RE = re.compile(
        r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4},? .*? - (?P<sender>[^:]+): (?P<message>.*)$"
    )
    _URL_RE = re.compile(r"(?i)^https?://")
    _SHORT_FILLERS = {
        "ok",
        "okay",
        "yeah",
        "yep",
        "no",
        "nah",
        "hi",
        "hey",
        "hmm",
        "um",
        "thanks",
        "thank you",
        "thx",
    }
    _CODE_MARKERS = (
        "def ",
        "class ",
        "import ",
        "SELECT ",
        "function ",
        "const ",
        "let ",
        "var ",
        "#include",
    )

    def __init__(self, synapse):
        self.synapse = synapse

    def from_chat_transcript(self, path: str, format: str = "auto") -> ImportReport:
        start = time.perf_counter()
        path_obj = Path(path)
        data = path_obj.read_text(encoding="utf-8")
        report = ImportReport()
        report_tags: Set[str] = set()

        messages = []
        detected_format = "plain_text"
        if format == "auto":
            messages, detected_format = self._detect_chat_messages(data)
        else:
            source: Any = data
            fmt = (format or "").lower()
            if fmt in {"chatgpt", "chatgpt_export", "claude", "claude_export"}:
                try:
                    source = json.loads(data)
                except json.JSONDecodeError:
                    source = data
            messages, detected_format = self._parse_chat_messages(source, format)

        for participant, text in messages:
            participant = self._coerce_text(participant)
            text = self._coerce_text(text)
            if not text or self._is_short_filler(text):
                report.skipped += 1
                continue

            tags = {
                f"source:{detected_format}",
                "source:chat_import",
                f"participant:{participant.lower()}" if participant else "participant:unknown",
            }
            metadata = {
                "source_format": detected_format,
                "participants": sorted({p for p in [participant] if p}),
            }
            if self._remember(text, metadata, tags, report):
                report_tags.update(tags)

        report.tags_created = sorted(report_tags)
        report.duration_ms = (time.perf_counter() - start) * 1000.0
        return report

    def from_markdown_folder(self, folder: str, recursive: bool = True) -> ImportReport:
        start = time.perf_counter()
        report = ImportReport()
        report_tags: Set[str] = set()

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(folder)

        if recursive:
            files = sorted(folder_path.rglob("*"))
        else:
            files = sorted(folder_path.glob("*"))

        for path in files:
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
                continue

            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                report.errors += 1
                continue

            frontmatter, body = self._split_frontmatter(raw)
            chunks = self._split_markdown_chunks(body)
            base_tag = f"note:{path.stem}"

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk or self._is_short_filler(chunk):
                    report.skipped += 1
                    continue
                tags = {
                    f"source:markdown_folder",
                    base_tag,
                }
                tags.update(self._coerce_tags(frontmatter.get("tags")))
                metadata = dict(frontmatter)
                metadata["source_format"] = "markdown_folder"
                metadata["source_file"] = str(path)
                if self._remember(chunk, metadata, tags, report):
                    report_tags.update(tags)

        report.tags_created = sorted(report_tags)
        report.duration_ms = (time.perf_counter() - start) * 1000.0
        return report

    def from_clipboard(self) -> ImportReport:
        start = time.perf_counter()
        report = ImportReport()
        report_tags: Set[str] = set()

        content = self._read_clipboard_text()
        if not content:
            report.duration_ms = (time.perf_counter() - start) * 1000.0
            return report

        content = content.strip()
        content_kind = self._classify_clipboard_content(content)
        tags = {
            "source:clipboard",
            "clipboard",
            f"clipboard:{content_kind}",
        }
        metadata = {
            "source_format": "clipboard",
            "clipboard_type": content_kind,
        }
        if self._remember(content, metadata, tags, report):
            report_tags.update(tags)
            # Keep a normalized single-line copy of code snippets for easier retrieval.
            if content_kind == "code":
                normalized = re.sub(r"\s+", " ", content).strip()
                if normalized and normalized != content:
                    aux_metadata = dict(metadata)
                    aux_metadata["clipboard_aux"] = "normalized"
                    aux_metadata["tags"] = sorted(tags)
                    try:
                        self.synapse.remember(normalized, metadata=aux_metadata)
                    except Exception:
                        pass

        report.tags_created = sorted(report_tags)
        report.duration_ms = (time.perf_counter() - start) * 1000.0
        return report

    def from_jsonl(
        self,
        path: str,
        text_field: str = "text",
        metadata_fields: Optional[Sequence[str]] = None,
    ) -> ImportReport:
        start = time.perf_counter()
        report = ImportReport()
        report_tags: Set[str] = set()
        metadata_fields = list(metadata_fields) if metadata_fields is not None else None

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    report.errors += 1
                    continue
                if not isinstance(payload, dict):
                    report.errors += 1
                    continue

                text = payload.get(text_field)
                if text_field not in payload:
                    continue
                if not isinstance(text, str) or not text.strip():
                    report.skipped += 1
                    continue

                metadata = {}
                if metadata_fields is not None:
                    for field in metadata_fields:
                        if field in payload:
                            metadata[field] = payload[field]
                else:
                    metadata = {
                        key: value
                        for key, value in payload.items()
                        if key != text_field and isinstance(key, str)
                    }

                tags = {"source:jsonl", "jsonl"}
                metadata["source_format"] = "jsonl"
                if self._remember(text, metadata, tags, report):
                    report_tags.update(tags)

        report.tags_created = sorted(report_tags)
        report.duration_ms = (time.perf_counter() - start) * 1000.0
        return report

    def from_csv(self, path: str, text_column: str = "text") -> ImportReport:
        start = time.perf_counter()
        report = ImportReport()
        report_tags: Set[str] = set()

        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or text_column not in reader.fieldnames:
                raise ValueError(f"Missing required text column: {text_column}")

            for row in reader:
                text = row.get(text_column)
                if not isinstance(text, str) or not text.strip():
                    report.skipped += 1
                    continue

                metadata = {
                    key: value
                    for key, value in row.items()
                    if key and key != text_column and value is not None
                }
                tags = {"source:csv", "csv"}
                metadata["source_format"] = "csv"
                if self._remember(text, metadata, tags, report):
                    report_tags.update(tags)

        report.tags_created = sorted(report_tags)
        report.duration_ms = (time.perf_counter() - start) * 1000.0
        return report

    def _remember(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]],
        tags: Set[str],
        report: ImportReport,
    ) -> bool:
        metadata_dict = dict(metadata or {})
        existing_tags = self._coerce_tags(metadata_dict.get("tags"))
        tags_to_store = set(existing_tags)
        tags_to_store.update(tags)

        if not tags_to_store:
            tags_to_store = set()
        metadata_dict["tags"] = sorted(tags_to_store)

        if self._is_short_filler(content):
            report.skipped += 1
            return False

        try:
            self.synapse.remember(self._coerce_text(content), metadata=metadata_dict)
            report.imported += 1
            return True
        except Exception:
            report.errors += 1
            return False

    def _split_frontmatter(self, text: str) -> (Dict[str, Any], str):
        if not text.startswith("---"):
            return {}, text

        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text

        end_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = index
                break

        if end_index is None:
            return {}, text

        frontmatter_blob = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :])
        return self._parse_frontmatter(frontmatter_blob), body

    def _parse_frontmatter(self, raw: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        current_list: Optional[str] = None

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue

            if stripped.startswith("- ") and current_list:
                parsed.setdefault(current_list, []).append(self._coerce_scalar(stripped[2:]))
                continue

            if ":" not in stripped:
                current_list = None
                continue

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_list = None
            if not key:
                continue

            if value == "":
                parsed[key] = []
                current_list = key
            else:
                parsed[key] = self._coerce_scalar(value)

        return parsed

    def _split_markdown_chunks(self, text: str) -> List[str]:
        lines = text.splitlines()
        has_headings = any(re.match(r"^#{1,6}\s+", line) for line in lines)

        if not has_headings:
            return [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]

        chunks: List[str] = []
        heading: Optional[str] = None
        bucket: List[str] = []

        def flush() -> None:
            if not bucket:
                return
            body = "\n".join(bucket).strip()
            if not body:
                return
            if heading:
                chunks.append(f"{heading}\n{body}")
            else:
                chunks.append(body)

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                flush()
                heading = heading_match.group(2).strip()
                bucket = []
                continue
            bucket.append(line)

        flush()
        if chunks:
            return chunks

        paragraphs = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
        return paragraphs

    def _detect_chat_messages(self, text: str) -> (List[tuple[str, str]], str):
        try:
            payload = json.loads(text)
            messages, detected = self._parse_chat_messages(payload, "chatgpt")
            if messages:
                return messages, detected
            messages, detected = self._parse_chat_messages(payload, "claude")
            if messages:
                return messages, detected
        except json.JSONDecodeError:
            messages = self._parse_whatsapp_messages(text)
            if messages:
                return messages, "whatsapp_export"

        messages = self._parse_plain_chat_text(text)
        return messages, "plain_text"

    def _parse_chat_messages(self, source: Any, source_name: str) -> (List[tuple[str, str]], str):
        if isinstance(source_name, str):
            source_name = source_name.lower()

        if source_name in {"chatgpt", "chatgpt_export"}:
            messages = self._extract_chat_messages(source, allow_system_skip=True)
            if messages:
                return messages, "chatgpt_export"

        if source_name in {"claude", "claude_export"}:
            messages = self._extract_chat_messages(source, allow_system_skip=True)
            if messages:
                return messages, "claude_export"

        if source_name == "whatsapp":
            messages = self._parse_whatsapp_messages(source if isinstance(source, str) else "")
            if messages:
                return messages, "whatsapp_export"

        if source_name in {"plain", "plain_text"}:
            messages = self._parse_plain_chat_text(source if isinstance(source, str) else "")
            if messages:
                return messages, "plain_text"

        if source_name == "auto":
            return self._parse_plain_chat_text(source if isinstance(source, str) else ""), "plain_text"

        return [], source_name

    def _extract_chat_messages(self, obj: Any, allow_system_skip: bool = True) -> List[tuple[str, str]]:
        messages = self._flatten_messages(obj)
        extracted: List[tuple[str, str]] = []

        for item in messages:
            if not isinstance(item, dict):
                continue

            role = self._coerce_text(self._extract_field(item, ["role", "type", "author", "speaker"]))
            if allow_system_skip and role.lower() in {"system", "tool"}:
                continue

            content = self._extract_message_content(item)
            if content is None:
                continue

            speaker = self._extract_field(item, ["author", "sender", "name", "user"])
            if not speaker and role and role not in {"assistant", "ai"}:
                speaker = role
            extracted.append((self._coerce_text(speaker), self._coerce_text(content)))

        return extracted

    def _flatten_messages(self, obj: Any) -> List[Any]:
        if isinstance(obj, list):
            if all(isinstance(item, dict) for item in obj):
                return obj
            flat: List[Any] = []
            for item in obj:
                flat.extend(self._flatten_messages(item))
            return flat

        if not isinstance(obj, dict):
            return []

        for key in ("messages", "chat_messages", "conversations", "history", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    return value
                messages: List[Any] = []
                for item in value:
                    messages.extend(self._flatten_messages(item))
                return messages

        mapping = obj.get("mapping")
        if isinstance(mapping, dict):
            message_items = []
            for item in mapping.values():
                if isinstance(item, dict):
                    nested = item.get("message")
                    if nested is not None:
                        message_items.append(nested)
            if message_items:
                return message_items

        return []

    def _parse_whatsapp_messages(self, text: str) -> List[tuple[str, str]]:
        messages: List[tuple[str, str]] = []
        for line in text.splitlines():
            match = self._WHATSAPP_LINE_RE.match(line.strip())
            if not match:
                continue
            speaker = self._coerce_text(match.group("sender"))
            body = self._coerce_text(match.group("message"))
            if speaker and body:
                messages.append((speaker, body))
        return messages

    def _parse_plain_chat_text(self, text: str) -> List[tuple[str, str]]:
        chunks = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
        messages: List[tuple[str, str]] = []
        for chunk in chunks:
            # Allow "Alice: hi" style chat exports.
            line_match = re.match(r"^\s*([^:]+):\s*(.+)$", chunk, re.DOTALL)
            if line_match:
                speaker = self._coerce_text(line_match.group(1))
                body = self._coerce_text(line_match.group(2))
                messages.append((speaker, body))
            else:
                messages.append(("unknown", chunk))
        return messages

    def _extract_message_content(self, item: Dict[str, Any]) -> Optional[str]:
        candidates = (
            ("content",),
            ("text",),
            ("message",),
            ("parts",),
            ("value",),
        )

        for path in candidates:
            value = self._extract_field(item, path)
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
            elif isinstance(value, list):
                parts = []
                for chunk in value:
                    if isinstance(chunk, dict):
                        for nested in ("text", "value", "content"):
                            nested_value = chunk.get(nested)
                            if isinstance(nested_value, str) and nested_value.strip():
                                parts.append(nested_value.strip())
                    elif isinstance(chunk, str) and chunk.strip():
                        parts.append(chunk.strip())
                if parts:
                    return " ".join(parts)
            elif isinstance(value, dict):
                for nested in ("text", "value", "parts", "content"):
                    nested_value = value.get(nested)
                    if isinstance(nested_value, str):
                        if nested_value.strip():
                            return nested_value.strip()
                    elif isinstance(nested_value, list):
                        parts = []
                        for chunk in nested_value:
                            if isinstance(chunk, str):
                                chunk = chunk.strip()
                                if chunk:
                                    parts.append(chunk)
                            elif isinstance(chunk, dict):
                                for nested_chunk_key in ("text", "value", "content"):
                                    chunk_value = chunk.get(nested_chunk_key)
                                    if isinstance(chunk_value, str):
                                        chunk_value = chunk_value.strip()
                                        if chunk_value:
                                            parts.append(chunk_value)
                        if parts:
                            return " ".join(parts)

        return None

    def _extract_field(self, payload: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in payload:
                value = payload[key]
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, dict):
                    role_value = value.get("role") or value.get("name") or value.get("id")
                    if isinstance(role_value, str):
                        return role_value.strip()
        return None

    def _is_short_filler(self, text: str) -> bool:
        text = self._coerce_text(text)
        if not text:
            return True
        if text.lower() in self._SHORT_FILLERS:
            return True
        words = text.split()
        if len(words) <= 2 and re.fullmatch(r"[a-zA-Z]{1,4}", text.replace("!", "").replace("?", "")):
            return True
        return False

    def _classify_clipboard_content(self, text: str) -> str:
        text = self._coerce_text(text)
        if not text:
            return "text"
        if self._URL_RE.search(text):
            return "url"
        if self._looks_like_code(text):
            return "code"
        return "text"

    def _looks_like_code(self, text: str) -> bool:
        if "```" in text:
            return True
        if any(marker in text for marker in self._CODE_MARKERS):
            return True
        if re.search(r"^\s{4,}\w", text, re.MULTILINE):
            return True
        return False

    def _read_clipboard_text(self) -> str:
        if os.name == "posix":
            # macOS has pbpaste; Linux/Unix often has xclip or xsel.
            for command in (["pbpaste"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
                try:
                    process = subprocess.run(command, capture_output=True, text=True, check=False)
                except FileNotFoundError:
                    continue
                if process.returncode == 0:
                    return process.stdout or ""
            return ""

        if os.name == "nt":
            process = subprocess.run([
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Clipboard",
            ], capture_output=True, text=True, check=False)
            if process.returncode == 0:
                return process.stdout or ""
            return ""

        return ""

    @staticmethod
    def _coerce_text(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _coerce_scalar(value: str) -> Any:
        if not value:
            return ""
        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
            return value[1:-1]
        if re.fullmatch(r"-?\d+", value):
            try:
                return int(value)
            except ValueError:
                return value
        if re.fullmatch(r"-?\d+\.\d+", value):
            try:
                return float(value)
            except ValueError:
                return value
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value

    @staticmethod
    def _coerce_tags(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(tag).strip() for tag in value if str(tag).strip()]
