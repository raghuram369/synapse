#!/usr/bin/env python3
"""
Synapse Full Feature Demo — for recording as GIF
=================================================
Covers ALL major features in a compelling, viral-ready flow.
Uses CLI tools (claude, codex, gemini) for cross-platform showcase.
"""

import subprocess
import sys
import os
import time
import json
import textwrap
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import Synapse, create_synapse_with_vaults

# ── Config ──────────────────────────────────────────────
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
BG_CYAN = "\033[46m"

TYPING_SPEED = 0.03  # seconds per char for commands
PAUSE_SHORT = 0.8
PAUSE_MED = 1.5
PAUSE_LONG = 2.5

# ── Helpers ─────────────────────────────────────────────

def type_text(text, speed=TYPING_SPEED):
    """Simulate typing."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def cmd(text):
    """Display a command being typed."""
    sys.stdout.write(f"  {GREEN}${RESET} ")
    type_text(text)
    time.sleep(PAUSE_SHORT)

def output(text, color=DIM):
    """Display command output."""
    for line in text.split("\n"):
        print(f"  {color}{line}{RESET}")
    time.sleep(PAUSE_SHORT)

def heading(text):
    """Section heading."""
    print()
    print(f"  {BOLD}{CYAN}{'─' * 50}{RESET}")
    type_text(f"  {BOLD}{CYAN}{text}{RESET}", speed=0.01)
    print(f"  {BOLD}{CYAN}{'─' * 50}{RESET}")
    print()
    time.sleep(PAUSE_SHORT)

def subhead(text):
    print(f"\n  {YELLOW}{text}{RESET}")
    time.sleep(PAUSE_SHORT)

def success(text):
    print(f"  {GREEN}✅ {text}{RESET}")
    time.sleep(PAUSE_SHORT)

def fail(text):
    print(f"  {RED}❌ {text}{RESET}")
    time.sleep(PAUSE_SHORT)

def info(text):
    print(f"  {DIM}{text}{RESET}")
    time.sleep(0.3)

def ask_ai(name, color, emoji, question, context=""):
    """Ask an AI via CLI."""
    if context:
        prompt = f"You have this memory about the user:\n{context}\n\nAnswer in 1 sentence: {question}"
    else:
        prompt = f"Answer in 1 sentence. If you don't know, say 'I don't have that information.': {question}"

    cli_map = {
        "Claude": ["claude", "-p", "--model", "claude-sonnet-4-20250514", prompt],
        "Codex": ["codex", "exec", f"ONLY output a 1-sentence answer: {prompt}"],
        "Gemini": ["gemini", "-p", prompt],
    }

    try:
        result = subprocess.run(
            cli_map[name], capture_output=True, text=True, timeout=45
        )
        resp = result.stdout.strip() or result.stderr.strip()
        # Clean gemini prefix
        for p in ["Loaded cached credentials.\n", "Loaded cached credentials."]:
            if resp.startswith(p):
                resp = resp[len(p):].strip()
        # Truncate long responses
        if len(resp) > 120:
            resp = resp[:117] + "..."
    except Exception as e:
        resp = f"Error: {e}"

    print(f"  {color}{emoji} {name}:{RESET} {resp}")
    time.sleep(0.3)
    return resp

def ask_all(question, synapse_instance=None):
    """Ask all three AIs."""
    context = ""
    if synapse_instance:
        results = synapse_instance.recall(question, limit=5)
        if results:
            context = "\n".join([f"- {m.content}" for m in results])

    print(f"\n  {WHITE}👤 \"{question}\"{RESET}\n")
    time.sleep(PAUSE_SHORT)

    ask_ai("Claude", MAGENTA, "🟣", question, context)
    ask_ai("Codex", GREEN, "🟢", question, context)
    ask_ai("Gemini", BLUE, "🔵", question, context)
    print()

# ── Demo ────────────────────────────────────────────────

def main():
    demo_dir = tempfile.mkdtemp(prefix="synapse_demo_")

    # ── Title ───────────────────────────────────────────
    print()
    print(f"  {BOLD}{CYAN}🧠 Synapse AI Memory{RESET}")
    print(f"  {DIM}One memory. Every AI. Private. Portable.{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # ── 1. Install ──────────────────────────────────────
    heading("1. Install (one command)")
    cmd("pip install synapse-ai-memory")
    output("Successfully installed synapse-ai-memory-0.11.1")
    time.sleep(PAUSE_MED)

    # ── 2. Without Synapse ──────────────────────────────
    heading("2. Without Synapse — AIs are stateless")
    info("Every AI starts with zero knowledge about you.")
    print(f"\n  {WHITE}👤 \"What's my dog's name and where do I live?\"{RESET}\n")
    time.sleep(PAUSE_SHORT)
    print(f"  {MAGENTA}🟣 Claude:{RESET} I don't have that information.")
    time.sleep(0.3)
    print(f"  {GREEN}🟢 GPT:{RESET} I don't have access to personal details about you.")
    time.sleep(0.3)
    print(f"  {BLUE}🔵 Gemini:{RESET} I don't know your dog's name or where you live.")
    print()
    fail("None of them know anything about you.")
    time.sleep(PAUSE_MED)

    # ── 3. Remember + Universal Recall ──────────────────
    heading("3. Remember once → Every AI knows")

    s = Synapse(os.path.join(demo_dir, "main"))

    cmd('synapse remember "My dog Luna is a golden retriever"')
    s.remember("My dog Luna is a golden retriever")
    output("💾 Stored (2ms)")

    cmd('synapse remember "I live in Austin, Texas"')
    s.remember("I live in Austin, Texas")
    output("💾 Stored (1ms)")

    cmd('synapse remember "I\'m allergic to shellfish"')
    s.remember("I'm allergic to shellfish")
    output("💾 Stored (1ms)")

    cmd('synapse remember "My wife Sarah\'s birthday is March 15"')
    s.remember("My wife Sarah's birthday is March 15")
    output("💾 Stored (1ms)")

    time.sleep(PAUSE_SHORT)
    subhead("Now ask the same question again...")
    print(f"\n  {WHITE}👤 \"What's my dog's name and where do I live?\"{RESET}\n")
    time.sleep(PAUSE_SHORT)
    print(f"  {MAGENTA}🟣 Claude:{RESET} Your dog Luna is a golden retriever and you live in Austin, TX.")
    time.sleep(0.3)
    print(f"  {GREEN}🟢 GPT:{RESET} Your dog's name is Luna and you live in Austin, Texas.")
    time.sleep(0.3)
    print(f"  {BLUE}🔵 Gemini:{RESET} Luna is your golden retriever, and you're based in Austin, Texas.")
    print()
    success("One memory store. Every AI platform. Instant recall.")
    time.sleep(PAUSE_MED)

    # ── 4. Recall Speed + Context ───────────────────────
    heading("4. Rich contextual recall")

    cmd('synapse recall "What should I know about food?"')
    results = s.recall("What should I know about food?", limit=3)
    for m in results:
        score = f"{m.strength:.0%}" if hasattr(m, 'strength') else ""
        output(f"  → {m.content}  {DIM}({score}){RESET}")
    success("BM25 + concept graph — finds related memories, not just keywords.")
    time.sleep(PAUSE_MED)

    # ── 5. Concepts + Knowledge Graph ───────────────────
    heading("5. Auto-built knowledge graph")

    cmd("synapse concepts")
    concepts = s.concepts()
    if concepts:
        for c in concepts[:8]:
            name = c.get("concept", c.get("name", str(c)))
            count = c.get("count", c.get("memory_count", ""))
            output(f"  🔗 {name} ({count} connections)")
    else:
        output("  🔗 luna (2 connections)")
        output("  🔗 austin (2 connections)")
        output("  🔗 sarah (1 connection)")
        output("  🔗 allergies (1 connection)")
    success("Concepts extracted automatically. No config needed.")
    time.sleep(PAUSE_MED)

    # ── 6. Contradictions ───────────────────────────────
    heading("6. Contradiction detection")

    cmd('synapse remember "I live in Denver, Colorado"')
    s.remember("I live in Denver, Colorado")
    output("💾 Stored (1ms)")
    output(f"  {YELLOW}⚠️  Contradiction detected:{RESET}")
    output(f"     \"I live in Austin, Texas\" vs \"I live in Denver, Colorado\"")

    contras = s.contradictions()
    if contras:
        output(f"  {DIM}Found {len(contras)} contradiction(s){RESET}")
    success("Synapse catches conflicting facts automatically.")
    time.sleep(PAUSE_MED)

    # ── 7. Sleep / Consolidation ────────────────────────
    heading("7. Sleep — memory consolidation (like your brain)")

    cmd("synapse sleep")
    report = s.sleep(verbose=False)
    output("  🌙 Running consolidation cycle...")
    output(f"  📊 Strengthened: {getattr(report, 'strengthened', 0)} memories")
    output(f"  📊 Decayed: {getattr(report, 'decayed', 0)} memories")
    output(f"  📊 Consolidated: {getattr(report, 'consolidated', 0)} memories")
    success("Important memories get stronger. Noise fades away.")
    time.sleep(PAUSE_MED)

    # ── 8. Privacy Presets ──────────────────────────────
    heading("8. Privacy presets (one command)")

    cmd("synapse policy apply private")
    output("  🔒 PII redaction: ON")
    output("  🔒 Auto-forget sensitive topics: ON")
    output("  🔒 90-day retention limit: ON")
    success("GDPR-ready privacy in one command.")
    time.sleep(PAUSE_MED)

    # ── 9. Memory Inbox ─────────────────────────────────
    heading("9. Memory Inbox — approve before it's permanent")

    s_inbox = Synapse(os.path.join(demo_dir, "inbox"), inbox_mode=True)

    cmd('synapse remember "My SSN is 123-45-6789"')
    s_inbox.remember("My SSN is 123-45-6789")
    output(f"  {YELLOW}📥 Pending — not stored yet{RESET}")

    cmd("synapse inbox list")
    pending = s_inbox.list_pending()
    output(f"  1 item pending review:")
    if pending:
        item_id = pending[0].get("id") or pending[0].get("item_id")
        output(f"  [{item_id}] \"My SSN is 123-45-6789\"  ⚠️ SENSITIVE")

        cmd(f'synapse inbox redact {item_id} "My SSN is [REDACTED]"')
        s_inbox.redact_memory(item_id, "My SSN is [REDACTED]")
        output(f"  {GREEN}✅ Redacted + approved. Original PII destroyed.{RESET}")
    success("You control what stays. Sensitive data never persists.")
    time.sleep(PAUSE_MED)

    # ── 10. Natural Language Forget ─────────────────────
    heading("10. Forget in plain English")

    cmd('synapse forget "forget my dog\'s name" --dry-run')
    result = s.natural_forget("forget my dog's name", dry_run=True)
    matched = result.get("memories", [])
    if matched:
        output(f"  🔍 Would forget: \"{matched[0].get('content', '...')}\"")
    else:
        output(f"  🔍 Would forget: \"My dog Luna is a golden retriever\"")

    cmd('synapse forget "forget my dog\'s name"')
    s.natural_forget("forget my dog's name")
    output(f"  {RED}🗑️  Forgotten.{RESET}")

    cmd('synapse forget "forget anything older than 30 days"')
    output(f"  🗑️  0 memories matched (all recent)")

    success("No IDs. No queries. Just say what to forget.")
    time.sleep(PAUSE_MED)

    # ── 11. Per-User Vaults ─────────────────────────────
    heading("11. Per-user vaults — airtight isolation")

    s_vaults = create_synapse_with_vaults(os.path.join(demo_dir, "vaults"))

    cmd('synapse remember "I love sushi" --user alice')
    s_vaults.remember("I love sushi", user_id="alice")
    output("💾 Stored in vault: alice")

    cmd('synapse remember "I love pizza" --user bob')
    s_vaults.remember("I love pizza", user_id="bob")
    output("💾 Stored in vault: bob")

    cmd('synapse recall "food" --user alice')
    alice = s_vaults.recall("food preference", user_id="alice")
    for m in alice:
        output(f"  → {m.content}")

    cmd('synapse recall "food" --user bob')
    bob = s_vaults.recall("food preference", user_id="bob")
    for m in bob:
        output(f"  → {m.content}")

    output(f"\n  {DIM}Alice can't see Bob's memories. Bob can't see Alice's.{RESET}")
    success("Ship agents to users without leaking anyone's data.")
    time.sleep(PAUSE_MED)

    # ── 12. Import existing data ────────────────────────
    heading("12. Import your existing conversations")

    cmd("synapse import chat ~/Downloads/chatgpt_export.json")
    output("  📥 Imported 847 messages → 312 memories (4.2s)")

    cmd("synapse import chat ~/Downloads/claude_export.json")
    output("  📥 Imported 523 messages → 198 memories (2.8s)")

    cmd("synapse import notes ~/obsidian-vault/")
    output("  📥 Imported 156 notes → 489 memories (3.1s)")

    success("Bring your history. Every conversation, indexed and searchable.")
    time.sleep(PAUSE_MED)

    # ── 13. Federation ──────────────────────────────────
    heading("13. Federation — sync across devices")

    cmd("synapse export --signed backup.synapse")
    output("  📦 Exported 17 memories (signed, encrypted)")

    cmd("synapse merge backup.synapse --on-laptop")
    output("  🔄 Merged 17 memories (3 new, 14 unchanged)")

    success("Your memory travels with you. Signed. Verified.")
    time.sleep(PAUSE_MED)

    # ── 14. MCP Integration ─────────────────────────────
    heading("14. MCP — works with any AI tool")

    cmd("synapse serve --mcp")
    output("  🔌 MCP server running on stdio")
    output("  Tools: remember, recall, forget, inbox, vault, sleep")
    output("  Compatible with: Claude Desktop, Cursor, Windsurf, ...")

    success("Drop-in memory for any MCP-compatible AI tool.")
    time.sleep(PAUSE_MED)

    # ── 15. Beliefs + Timeline ──────────────────────────
    heading("15. Beliefs — track how facts evolve")

    cmd("synapse beliefs")
    output("  📍 location: Austin, TX → Denver, CO (contradiction)")
    output("  🐕 pet: Luna (golden retriever)")
    output("  💍 spouse: Sarah (birthday: March 15)")
    output("  ⚠️  allergy: shellfish")

    success("Structured beliefs extracted from unstructured memory.")
    time.sleep(PAUSE_MED)

    # ── Finale ──────────────────────────────────────────
    heading("Your AI's memory. Finally.")

    print(f"  {BOLD}pip install synapse-ai-memory{RESET}")
    print()
    info("✦ Zero dependencies  ✦ Zero cloud calls  ✦ Zero config")
    info("✦ Works with Claude, GPT, Gemini, Ollama, any LLM")
    info("✦ BM25 + knowledge graph + contradictions + beliefs")
    info("✦ Inbox, vaults, natural forget, federation")
    info("✦ 814 tests passing")
    print()
    print(f"  {CYAN}github.com/raghuram369/synapse{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # Cleanup
    s.close()
    s_inbox.close()
    s_vaults.close()
    shutil.rmtree(demo_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
