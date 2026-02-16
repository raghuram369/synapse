#!/usr/bin/env python3
"""
Synapse Cross-Platform Demo
============================
Shows Synapse as a universal memory layer across Claude, GPT, and Gemini.

Act 1: Without memory — all three AIs are clueless
Act 2: Add memory once — all three remember
Act 3: Features (inbox, forget, vaults) work across all platforms
"""

import subprocess
import sys
import os
import time
import json
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse, create_synapse_with_vaults

# ── Colors ──────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

AUTO_MODE = "--auto" in sys.argv

CLAUDE_COLOR = MAGENTA
CODEX_COLOR = GREEN
GEMINI_COLOR = BLUE
SYNAPSE_COLOR = CYAN
USER_COLOR = YELLOW

# ── AI Backends ─────────────────────────────────────────

def _run_cli(cmd, timeout=60):
    """Run a CLI command and return stdout."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        out = result.stdout.strip()
        if not out:
            out = result.stderr.strip()
        # Strip common Gemini prefix
        for prefix in ["Loaded cached credentials.\n", "Loaded cached credentials."]:
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
        return out
    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except Exception as e:
        return f"Error: {e}"

def ask_claude(question: str, context: str = "") -> str:
    """Ask Claude via claude CLI."""
    if context:
        prompt = f"You have the following memory context about the user:\n{context}\n\nAnswer this question concisely in 1-2 sentences: {question}"
    else:
        prompt = f"Answer this question concisely. If you don't know, say 'I don't know.': {question}"
    return _run_cli(["claude", "-p", "--model", "claude-sonnet-4-20250514", prompt])

def ask_gpt(question: str, context: str = "") -> str:
    """Ask GPT via codex exec (non-interactive)."""
    if context:
        prompt = f"ONLY print a 1-2 sentence answer, nothing else. Memory context:\n{context}\n\nQuestion: {question}"
    else:
        prompt = f"ONLY print a 1-2 sentence answer, nothing else. If you don't know, say 'I don't know.'\n\nQuestion: {question}"
    return _run_cli(["codex", "exec", prompt], timeout=60)

def ask_gemini(question: str, context: str = "") -> str:
    """Ask Gemini via gemini CLI."""
    if context:
        prompt = f"You have the following memory context about the user:\n{context}\n\nAnswer this question concisely in 1-2 sentences: {question}"
    else:
        prompt = f"Answer this question concisely. If you don't know, say 'I don't know.': {question}"
    return _run_cli(["gemini", "-p", prompt])

# ── Display Helpers ─────────────────────────────────────

def slow_print(text, delay=0.02):
    if AUTO_MODE:
        print(text)
        return
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def section(title):
    print(f"\n{'═' * 60}")
    slow_print(f"  {BOLD}{title}{RESET}")
    print(f"{'═' * 60}\n")

def user_says(text):
    slow_print(f"  {USER_COLOR}👤 User:{RESET} {text}")

def synapse_cmd(text):
    slow_print(f"  {SYNAPSE_COLOR}🧠 synapse{RESET} {text}")

def ai_response(name, color, emoji, response):
    print(f"  {color}{emoji} {name}:{RESET}")
    for line in textwrap.wrap(response, width=55):
        print(f"     {DIM}{line}{RESET}")
    print()

def ask_all(question, synapse_instance=None):
    """Ask all three AIs, optionally with Synapse context."""
    context = ""
    if synapse_instance:
        results = synapse_instance.recall(question, limit=5)
        if results:
            context = "\n".join([f"- {m.content if hasattr(m, 'content') else m.get('content', str(m))}" for m in results])

    print()
    user_says(question)
    print()

    # Ask all three
    print(f"  {DIM}asking all three AIs...{RESET}\n")

    claude_resp = ask_claude(question, context)
    ai_response("Claude", CLAUDE_COLOR, "🟣", claude_resp)

    gpt_resp = ask_gpt(question, context)
    ai_response("Codex", CODEX_COLOR, "🟢", gpt_resp)

    gemini_resp = ask_gemini(question, context)
    ai_response("Gemini", GEMINI_COLOR, "🔵", gemini_resp)

def pause(msg="Press Enter to continue..."):
    if AUTO_MODE:
        time.sleep(0.5)
        return
    input(f"\n  {DIM}{msg}{RESET}")

# ── Main Demo ───────────────────────────────────────────

def main():
    import tempfile
    demo_dir = tempfile.mkdtemp(prefix="synapse_demo_")

    print(f"\n{BOLD}{SYNAPSE_COLOR}")
    print(r"   ____                                 ")
    print(r"  / ___| _   _ _ __   __ _ _ __  ___  ___")
    print(r"  \___ \| | | | '_ \ / _` | '_ \/ __|/ _ \ ")
    print(r"   ___) | |_| | | | | (_| | |_) \__ \  __/")
    print(r"  |____/ \__, |_| |_|\__,_| .__/|___/\___|")
    print(r"         |___/            |_|              ")
    print(f"{RESET}")
    slow_print(f"  {BOLD}One memory. Every AI. Universal recall.{RESET}\n")
    pause()

    # ── ACT 1: Without Synapse ──────────────────────────
    section("ACT 1: Without Synapse — Everyone's Clueless")

    slow_print(f"  {DIM}No memory configured. Let's ask a personal question.{RESET}")
    ask_all("What is my dog's name?")
    
    slow_print(f"  {RED}❌ None of them know. They have no memory.{RESET}")
    pause()

    # ── ACT 2: Add Memory → Universal Recall ────────────
    section("ACT 2: One Memory, Every AI Remembers")

    s = Synapse(os.path.join(demo_dir, "shared"))
    
    synapse_cmd('remember "My dog\'s name is Luna. She\'s a golden retriever."')
    s.remember("My dog's name is Luna. She's a golden retriever.")
    time.sleep(0.3)
    
    synapse_cmd('remember "I live in Austin, Texas."')
    s.remember("I live in Austin, Texas.")
    time.sleep(0.3)
    
    synapse_cmd('remember "My favorite coffee is a cortado from Fleet Coffee."')
    s.remember("My favorite coffee is a cortado from Fleet Coffee.")
    
    slow_print(f"\n  {GREEN}✅ 3 memories stored. Now let's ask again...{RESET}")
    pause()

    ask_all("What is my dog's name?", s)
    slow_print(f"  {GREEN}✅ All three know! Same memory, every platform.{RESET}")
    pause()

    ask_all("Where do I live and what coffee do I like?", s)
    slow_print(f"  {GREEN}✅ Rich context across all AIs from a single source.{RESET}")
    pause()

    # ── ACT 3: Memory Inbox ─────────────────────────────
    section("ACT 3: Memory Inbox — Trust Before Storage")

    s_inbox = Synapse(os.path.join(demo_dir, "inbox_demo"), inbox_mode=True)

    synapse_cmd('remember "My SSN is 123-45-6789"  # with inbox mode')
    s_inbox.remember("My SSN is 123-45-6789")
    
    pending = s_inbox.list_pending()
    slow_print(f"\n  {YELLOW}📥 Memory landed in inbox (pending). Not stored yet.{RESET}")
    slow_print(f"  {DIM}Pending items: {len(pending)}{RESET}")
    
    if pending:
        item_id = pending[0].get("id") or pending[0].get("item_id")
        slow_print(f"\n  {YELLOW}⚠️  This looks sensitive. Let's redact it.{RESET}")
        synapse_cmd(f'inbox redact {item_id} "My SSN is [REDACTED]"')
        s_inbox.redact_memory(item_id, "My SSN is [REDACTED]")
        slow_print(f"  {GREEN}✅ Redacted and approved. Original PII never persisted.{RESET}")

    pause()

    # ── ACT 4: Natural Language Forget ──────────────────
    section("ACT 4: Forget in Plain English")

    slow_print(f"  {DIM}Current memories in shared store: {len(s.store.memories)}{RESET}")
    
    synapse_cmd('nlforget "forget my dog\'s name" --dry-run')
    result = s.natural_forget("forget my dog's name", dry_run=True)
    slow_print(f"  {YELLOW}🔍 Dry run — would forget: {result.get('matched', result)}{RESET}")
    
    synapse_cmd('nlforget "forget my dog\'s name"')
    result = s.natural_forget("forget my dog's name")
    slow_print(f"  {RED}🗑️  Forgotten.{RESET}")
    
    slow_print(f"\n  {DIM}Let's verify across all AIs...{RESET}")
    ask_all("What is my dog's name?", s)
    slow_print(f"  {GREEN}✅ Forgotten everywhere. One command, universal effect.{RESET}")
    pause()

    # ── ACT 5: Per-User Vaults ──────────────────────────
    section("ACT 5: Per-User Vaults — Airtight Isolation")

    s_vaults = create_synapse_with_vaults(os.path.join(demo_dir, "vaults"))

    synapse_cmd('remember "I love sushi" --user alice')
    s_vaults.remember("I love sushi", user_id="alice")
    
    synapse_cmd('remember "I love pizza" --user bob')
    s_vaults.remember("I love pizza", user_id="bob")

    print()
    slow_print(f"  {CYAN}Alice's view:{RESET}")
    alice_results = s_vaults.recall("food preference", user_id="alice")
    for m in alice_results:
        slow_print(f"    → {m.content if hasattr(m, 'content') else str(m)}")

    slow_print(f"\n  {CYAN}Bob's view:{RESET}")
    bob_results = s_vaults.recall("food preference", user_id="bob")
    for m in bob_results:
        slow_print(f"    → {m.content if hasattr(m, 'content') else str(m)}")

    slow_print(f"\n  {GREEN}✅ Complete isolation. Alice can't see Bob's memories.{RESET}")
    pause()

    # ── Finale ──────────────────────────────────────────
    section("That's Synapse")
    
    slow_print(f"  {BOLD}One memory layer. Every AI platform.{RESET}")
    slow_print(f"  {DIM}Private. Portable. Zero dependencies.{RESET}")
    print()
    slow_print(f"  pip install synapse-ai-memory")
    slow_print(f"  {DIM}https://github.com/raghuram369/synapse{RESET}")
    print()

    # Cleanup
    s.close()
    s_inbox.close()
    s_vaults.close()
    import shutil
    shutil.rmtree(demo_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
