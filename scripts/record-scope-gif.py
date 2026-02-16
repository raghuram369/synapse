#!/usr/bin/env python3
"""Generate a fake asciinema .cast file for the scope demo, then convert to GIF via agg."""
import json, subprocess, os

CAST_PATH = "assets/scope-demo.cast"
GIF_PATH = "assets/scope-demo.gif"

# Each entry: (delay_seconds, text)
# We simulate typing + output in a terminal
lines = []
t = 0.0

def out(text, delay=0.05):
    global t
    t += delay
    lines.append([t, "o", text])

def type_line(text, delay_per_char=0.04, pause_after=0.3):
    for ch in text:
        out(ch, delay_per_char)
    out("\r\n", 0.05)
    out("", pause_after)

def prompt():
    out("\r\n\x1b[32m>>> \x1b[0m", 0.3)

def output_line(text, delay=0.1):
    out(text + "\r\n", delay)

# Header
header = {
    "version": 2,
    "width": 82,
    "height": 28,
    "timestamp": 1739682000,
    "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
    "theme": {"fg": "#d4d4d4", "bg": "#1e1e1e"}
}

# Scene 1: imports
out("\x1b[1;36m# Memory Scoping Demo\x1b[0m\r\n", 0.5)
prompt()
type_line("from synapse import Synapse")
prompt()
type_line("s = Synapse()")
output_line("\x1b[2m✓ Store loaded (local)\x1b[0m")

# Scene 2: remember with scopes
prompt()
type_line('s.remember("I prefer dark mode", scope="public")')
output_line("\x1b[32m✓ saved [public]\x1b[0m")

prompt()
type_line('s.remember("Project deadline is March 1", scope="shared")')
output_line("\x1b[33m✓ saved [shared]\x1b[0m")

prompt()
type_line('s.remember("My SSN is 123-45-6789", scope="private")')
output_line("\x1b[31m✓ saved [private] 🔒\x1b[0m")

# Scene 3: recall shared only
prompt()
type_line('s.recall("what do you know?", scope="shared")')
output_line("")
output_line("\x1b[1mResults (scope=shared):\x1b[0m")
output_line('  1. "I prefer dark mode" \x1b[2m[public]\x1b[0m')
output_line('  2. "Project deadline is March 1" \x1b[2m[shared]\x1b[0m')
output_line("\x1b[2m  (1 private memory hidden)\x1b[0m")

# Scene 4: ScopePolicy
prompt()
type_line("from scope_policy import ScopePolicy")
prompt()
type_line('s = Synapse("./store", scope_policy=ScopePolicy.external())')
output_line("\x1b[33m⚡ Policy locked: max_scope=shared\x1b[0m")

prompt()
type_line('s.recall("what\'s my SSN?", scope="private")')
output_line("\x1b[31m⚠ scope clamped: private → shared\x1b[0m")
output_line("\x1b[2mNo results.\x1b[0m")

# Scene 5: sensitive flag
prompt()
type_line('s.remember("My kid goes to Lincoln Elementary", sensitive=True)')
output_line("\x1b[31m✓ saved [private] 🔒 SENSITIVE\x1b[0m")
output_line("\x1b[2m  auto-detected: school name → sensitive=True\x1b[0m")

out("\r\n\x1b[1;32m✓ Three layers: scope → sensitive → PII scrub\x1b[0m\r\n", 0.8)
out("", 2.0)

# Write cast
os.makedirs("assets", exist_ok=True)
with open(CAST_PATH, "w") as f:
    f.write(json.dumps(header) + "\n")
    for entry in lines:
        f.write(json.dumps(entry) + "\n")

print(f"Wrote {CAST_PATH} ({len(lines)} frames)")

# Convert to GIF
result = subprocess.run(
    ["agg", "--theme", "monokai", "--font-size", "16", CAST_PATH, GIF_PATH],
    capture_output=True, text=True
)
if result.returncode == 0:
    size = os.path.getsize(GIF_PATH)
    print(f"Wrote {GIF_PATH} ({size // 1024}KB)")
else:
    print(f"agg failed: {result.stderr}")
    print("Cast file is still available for manual conversion.")
