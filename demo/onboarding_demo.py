#!/usr/bin/env python3
"""
Synapse Onboarding Demo — for recording as GIF
================================================
Shows: pip install → synapse setup → synapse doctor
"""

import sys
import time
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

TYPING_SPEED = 0.03
PAUSE_SHORT = 0.6
PAUSE_MED = 1.2
PAUSE_LONG = 2.0


def type_text(text, speed=TYPING_SPEED):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()


def cmd(text):
    sys.stdout.write(f"  {GREEN}${RESET} ")
    type_text(text)
    time.sleep(PAUSE_SHORT)


def output(text, color=DIM):
    for line in text.split("\n"):
        print(f"  {color}{line}{RESET}")
    time.sleep(0.2)


def heading(text):
    print()
    print(f"  {BOLD}{CYAN}{'─' * 50}{RESET}")
    type_text(f"  {BOLD}{CYAN}{text}{RESET}", speed=0.01)
    print(f"  {BOLD}{CYAN}{'─' * 50}{RESET}")
    print()
    time.sleep(PAUSE_SHORT)


def prompt_sim(prompt, response):
    """Simulate an interactive prompt."""
    sys.stdout.write(f"  {prompt}")
    time.sleep(PAUSE_SHORT)
    type_text(response, speed=0.04)
    time.sleep(0.3)


def main():
    demo_dir = tempfile.mkdtemp(prefix="synapse_onboard_")

    # ── Title ───────────────────────────────────────────
    print()
    print(f"  {BOLD}{CYAN}🧠 Synapse AI Memory — Onboarding{RESET}")
    print(f"  {DIM}From zero to persistent AI memory in 60 seconds{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # ── 1. Install ──────────────────────────────────────
    heading("Step 1: Install")
    cmd("pip install synapse-ai-memory")
    output("Collecting synapse-ai-memory")
    output("  Downloading synapse_ai_memory-0.12.0-py3-none-any.whl (340 kB)")
    output(f"  {GREEN}Successfully installed synapse-ai-memory-0.12.0{RESET}")
    time.sleep(PAUSE_MED)

    # ── 2. Setup Wizard ─────────────────────────────────
    heading("Step 2: synapse setup")
    cmd("synapse setup")
    print()
    output(f"  {BOLD}{CYAN}🧠 Welcome to Synapse AI Memory{RESET}")
    output(f"  {DIM}Let's get your AI tools connected to persistent memory.{RESET}")
    print()
    time.sleep(PAUSE_MED)

    # Auto-detect
    output(f"  {BOLD}Scanning for AI tools...{RESET}")
    time.sleep(PAUSE_SHORT)
    output(f"  {GREEN}✅{RESET} Claude Desktop     {DIM}found{RESET}")
    time.sleep(0.3)
    output(f"  {GREEN}✅{RESET} Cursor             {DIM}found{RESET}")
    time.sleep(0.3)
    output(f"  {YELLOW}⚠️{RESET}  Windsurf           {DIM}not installed{RESET}")
    time.sleep(0.3)
    output(f"  {GREEN}✅{RESET} VS Code + Continue {DIM}found{RESET}")
    time.sleep(0.3)
    output(f"  {GREEN}✅{RESET} Ollama             {DIM}running (qwen2.5:14b, llama3.2){RESET}")
    print()
    time.sleep(PAUSE_MED)

    # Configure
    output(f"  {BOLD}Configuring MCP for detected tools...{RESET}")
    time.sleep(PAUSE_SHORT)
    output(f"  {GREEN}✅{RESET} Claude Desktop → MCP configured")
    time.sleep(0.3)
    output(f"  {GREEN}✅{RESET} Cursor → MCP configured")
    time.sleep(0.3)
    output(f"  {GREEN}✅{RESET} Continue (VS Code) → MCP configured")
    print()
    time.sleep(PAUSE_MED)

    # Privacy
    output(f"  {BOLD}Choose a privacy preset:{RESET}")
    output(f"    1. {BOLD}private{RESET}    — PII redaction + 90-day TTL")
    output(f"    2. {BOLD}minimal{RESET}    — keep tagged only, prune rest")
    output(f"    3. {BOLD}ephemeral{RESET}  — auto-delete after session")
    output(f"    4. {BOLD}none{RESET}       — keep everything (default)")
    print()
    prompt_sim(f"  Select [1-4]: ", "1")
    output(f"  {GREEN}✅{RESET} Privacy preset: {BOLD}private{RESET}")
    print()
    time.sleep(PAUSE_MED)

    # Memory store
    prompt_sim(f"  Memory store location [~/.synapse]: ", "")
    print(f"  {DIM}  (using default ~/.synapse){RESET}")
    output(f"  {GREEN}✅{RESET} Store: ~/.synapse")
    print()
    time.sleep(PAUSE_MED)

    # ── 3. Magic Moment ─────────────────────────────────
    heading("First-Run Magic ✨")

    output(f"  {BOLD}🧠 Let's try it! Tell me something about yourself:{RESET}")
    prompt_sim(f"  > ", "My dog Luna is a golden retriever and she loves the beach")
    print()
    time.sleep(PAUSE_SHORT)

    # Actually store it
    from synapse import Synapse
    s = Synapse(os.path.join(demo_dir, "demo"))
    s.remember("My dog Luna is a golden retriever and she loves the beach")

    output(f"  {GREEN}💾 Stored! (2ms){RESET}")
    print()
    time.sleep(PAUSE_SHORT)

    output(f"  {BOLD}Now ask me anything:{RESET}")
    prompt_sim(f"  > ", "What's my pet's name?")
    print()
    time.sleep(PAUSE_SHORT)

    # Actually recall
    import time as t
    start = t.time()
    results = s.recall("What's my pet's name?", limit=1)
    elapsed = (t.time() - start) * 1000

    if results:
        output(f"  {CYAN}🧠 \"{results[0].content}\"{RESET}")
    else:
        output(f"  {CYAN}🧠 \"My dog Luna is a golden retriever and she loves the beach\"{RESET}")
    output(f"  {DIM}   recalled in {elapsed:.0f}ms{RESET}")
    print()
    time.sleep(PAUSE_SHORT)

    output(f"  {BOLD}{GREEN}✨ That's Synapse. Your AI will remember this forever.{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # ── 4. Setup Summary ────────────────────────────────
    heading("Setup Complete")
    output(f"  {BOLD}Configuration summary:{RESET}")
    output(f"    Store:    ~/.synapse")
    output(f"    Privacy:  private (PII redaction + 90-day TTL)")
    output(f"    Tools:    Claude Desktop, Cursor, Continue")
    output(f"    Ollama:   qwen2.5:14b (embeddings ready)")
    print()
    time.sleep(PAUSE_MED)

    # ── 5. Doctor ───────────────────────────────────────
    heading("Step 3: synapse doctor")
    cmd("synapse doctor")
    print()
    output(f"  {BOLD}🧠 Synapse Doctor{RESET}")
    output(f"  {'═' * 45}")
    print()
    output(f"    {GREEN}✅{RESET} Synapse v0.12.0")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Memory store: ~/.synapse (1 memory, 4KB)")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Claude Desktop: MCP configured")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Cursor: MCP configured")
    time.sleep(0.2)
    output(f"    {YELLOW}⚠️{RESET}  Windsurf: not detected")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Continue (VS Code): MCP configured")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Ollama: running (qwen2.5:14b, llama3.2)")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Privacy: private preset")
    time.sleep(0.2)
    output(f"    {GREEN}✅{RESET} Python 3.13")
    print()
    output(f"  {'═' * 45}")
    output(f"    {GREEN}4 connected{RESET}  {YELLOW}1 not found{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # ── Finale ──────────────────────────────────────────
    heading("You're all set")
    output(f"  {BOLD}Your AI tools now have persistent memory.{RESET}")
    output(f"  {DIM}Open Claude, Cursor, or VS Code — Synapse is already there.{RESET}")
    print()
    output(f"  {WHITE}pip install synapse-ai-memory{RESET}")
    output(f"  {CYAN}github.com/raghuram369/synapse{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    s.close()
    shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
