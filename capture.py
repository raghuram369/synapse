"""Frictionless memory capture — clipboard watching and piped input."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse


def _get_clipboard() -> str:
    """Read clipboard text, platform-aware."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=3)
            return result.stdout
        elif system == "Linux":
            # Try xclip first, then xsel
            for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            result = subprocess.run(["powershell", "-command", "Get-Clipboard"], capture_output=True, text=True, timeout=3)
            return result.stdout
    except Exception:
        pass
    return ""


def clip_text(synapse: "Synapse", text: str, tags: Optional[List[str]] = None, source: str = "clip") -> Any:
    """Remember arbitrary text with optional tags."""
    metadata: Dict[str, Any] = {"source": source}
    if tags:
        metadata["tags"] = tags
    memory = synapse.remember(text.strip(), metadata=metadata)
    return memory


def clip_stdin(synapse: "Synapse", tags: Optional[List[str]] = None) -> Any:
    """Read stdin and remember it."""
    text = sys.stdin.read().strip()
    if not text:
        print("⚠️  No input received from stdin")
        return None
    memory = clip_text(synapse, text, tags=tags, source="stdin")
    print(f"✅ Remembered from stdin: {text[:80]}...")
    return memory


def clipboard_watch(synapse: "Synapse", interval: float = 2.0, tags: Optional[List[str]] = None) -> None:
    """Poll clipboard every N seconds, remember new content. Runs until Ctrl+C."""
    seen = set()
    # Seed with current clipboard to avoid capturing pre-existing content
    current = _get_clipboard().strip()
    if current:
        seen.add(current)

    print(f"👀 Watching clipboard (every {interval}s). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(interval)
            content = _get_clipboard().strip()
            if not content or content in seen:
                continue
            seen.add(content)
            metadata: Dict[str, Any] = {"source": "clipboard"}
            if tags:
                metadata["tags"] = tags
            memory = synapse.remember(content, metadata=metadata)
            preview = content[:80].replace("\n", " ")
            print(f"📋 Captured #{memory.id}: {preview}")
    except KeyboardInterrupt:
        print(f"\n🛑 Clipboard watch stopped. Captured {len(seen) - 1} items.")
