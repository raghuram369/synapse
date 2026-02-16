"""Platform-aware autostart service management for Synapse."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, Optional


PLIST_PATH = os.path.expanduser("~/Library/LaunchAgents/com.synapse.memory.plist")
SYSTEMD_PATH = os.path.expanduser("~/.config/systemd/user/synapse-memory.service")


def _detect_platform() -> str:
    s = platform.system()
    if s == "Darwin":
        return "macos"
    if s == "Linux":
        return "linux"
    return "other"


def _synapse_bin() -> str:
    return shutil.which("synapse") or f"{sys.executable} -m cli"


def install_service(db_path: str, sleep_schedule: str = "daily") -> str:
    """Install an autostart service. Returns path of created file."""
    plat = _detect_platform()
    db_path = os.path.abspath(os.path.expanduser(db_path))

    if plat == "macos":
        return _install_launchd(db_path, sleep_schedule)
    elif plat == "linux":
        return _install_systemd(db_path, sleep_schedule)
    else:
        print("⚠️  Unsupported platform for automatic service install.")
        print("  Manually add `synapse serve --db <path>` to your startup.")
        return ""


def _install_launchd(db_path: str, sleep_schedule: str) -> str:
    synapse_bin = shutil.which("synapse")
    if synapse_bin:
        program_args = f"""    <array>
        <string>{synapse_bin}</string>
        <string>serve</string>
        <string>--db</string>
        <string>{db_path}</string>
    </array>"""
    else:
        program_args = f"""    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>cli</string>
        <string>serve</string>
        <string>--db</string>
        <string>{db_path}</string>
    </array>"""

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.synapse.memory</string>
    <key>ProgramArguments</key>
{program_args}
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{os.path.expanduser("~/.synapse/synapse-service.log")}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.expanduser("~/.synapse/synapse-service.err")}</string>
</dict>
</plist>
"""
    os.makedirs(os.path.dirname(PLIST_PATH), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.synapse"), exist_ok=True)
    with open(PLIST_PATH, "w", encoding="utf-8") as f:
        f.write(plist)

    try:
        subprocess.run(["launchctl", "load", PLIST_PATH], capture_output=True)
    except Exception:
        pass

    print(f"✅ Synapse will start automatically on login")
    print(f"   Plist: {PLIST_PATH}")
    return PLIST_PATH


def _install_systemd(db_path: str, sleep_schedule: str) -> str:
    synapse_bin = shutil.which("synapse")
    if synapse_bin:
        exec_start = f"{synapse_bin} serve --db {db_path}"
    else:
        exec_start = f"{sys.executable} -m cli serve --db {db_path}"

    unit = f"""[Unit]
Description=Synapse AI Memory Service
After=default.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    unit_dir = os.path.dirname(SYSTEMD_PATH)
    os.makedirs(unit_dir, exist_ok=True)
    with open(SYSTEMD_PATH, "w", encoding="utf-8") as f:
        f.write(unit)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        subprocess.run(["systemctl", "--user", "enable", "synapse-memory.service"], capture_output=True)
        subprocess.run(["systemctl", "--user", "start", "synapse-memory.service"], capture_output=True)
    except Exception:
        pass

    print(f"✅ Synapse will start automatically on login")
    print(f"   Unit: {SYSTEMD_PATH}")
    return SYSTEMD_PATH


def uninstall_service() -> bool:
    """Remove the autostart service."""
    plat = _detect_platform()

    if plat == "macos":
        if os.path.exists(PLIST_PATH):
            try:
                subprocess.run(["launchctl", "unload", PLIST_PATH], capture_output=True)
            except Exception:
                pass
            os.remove(PLIST_PATH)
            print("✅ Synapse service removed from launchd")
            return True
        print("⬜ Synapse service not installed")
        return False

    elif plat == "linux":
        if os.path.exists(SYSTEMD_PATH):
            try:
                subprocess.run(["systemctl", "--user", "stop", "synapse-memory.service"], capture_output=True)
                subprocess.run(["systemctl", "--user", "disable", "synapse-memory.service"], capture_output=True)
            except Exception:
                pass
            os.remove(SYSTEMD_PATH)
            try:
                subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            except Exception:
                pass
            print("✅ Synapse service removed from systemd")
            return True
        print("⬜ Synapse service not installed")
        return False

    else:
        print("⬜ Synapse service not installed (unsupported platform)")
        return False


def service_status() -> Dict[str, object]:
    """Check if the synapse service is running."""
    plat = _detect_platform()
    result: Dict[str, object] = {"platform": plat, "installed": False, "running": False}

    if plat == "macos":
        result["installed"] = os.path.exists(PLIST_PATH)
        if result["installed"]:
            try:
                proc = subprocess.run(
                    ["launchctl", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                result["running"] = "com.synapse.memory" in proc.stdout
            except Exception:
                pass
    elif plat == "linux":
        result["installed"] = os.path.exists(SYSTEMD_PATH)
        if result["installed"]:
            try:
                proc = subprocess.run(
                    ["systemctl", "--user", "is-active", "synapse-memory.service"],
                    capture_output=True, text=True, timeout=5,
                )
                result["running"] = proc.stdout.strip() == "active"
            except Exception:
                pass

    if result["installed"]:
        if result["running"]:
            print("✅ Synapse service is running")
        else:
            print("⚠️  Synapse service is installed but not running")
    else:
        print("⬜ Synapse service not installed")

    return result
