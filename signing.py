"""Signed share artifacts — SHA-256 sidecar signatures."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict


VERSION = "0.8.1"


def _compute_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sig_path(file_path: str) -> str:
    return f"{file_path}.sig"


def sign_artifact(file_path: str) -> str:
    """Sign a file and write a .sig sidecar. Returns sig path."""
    file_hash = _compute_hash(file_path)
    sig = {
        "hash": file_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "signer": "synapse-ai-memory",
        "version": VERSION,
    }
    sig_file = _sig_path(file_path)
    with open(sig_file, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
        f.write("\n")
    short_hash = file_hash.split(":")[1][:12]
    print(f"✅ Signed: {os.path.basename(sig_file)} ({file_hash[:19]}...{short_hash})")
    return sig_file


def verify_artifact(file_path: str) -> Dict[str, Any]:
    """Verify a file's .sig sidecar. Returns result dict."""
    sig_file = _sig_path(file_path)
    if not os.path.exists(sig_file):
        print(f"❌ No signature file found: {sig_file}")
        return {"valid": False, "error": "no signature file"}

    with open(sig_file, "r", encoding="utf-8") as f:
        sig = json.load(f)

    stored_hash = sig.get("hash", "")
    actual_hash = _compute_hash(file_path)
    valid = stored_hash == actual_hash

    if valid:
        print(f"✅ Verified: signature valid")
    else:
        print(f"❌ Verification failed")

    return {
        "valid": valid,
        "hash": actual_hash,
        "timestamp": sig.get("timestamp", ""),
        "stored_hash": stored_hash,
    }


def sign_pack(pack_path: str) -> str:
    """Sign a .brain file."""
    return sign_artifact(pack_path)


def verify_pack(pack_path: str) -> Dict[str, Any]:
    """Verify a .brain file signature."""
    return verify_artifact(pack_path)
