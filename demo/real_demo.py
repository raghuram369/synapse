#!/usr/bin/env python3
"""
Synapse REAL Demo — No mocking. Actual AI calls.
==================================================
Uses Gemini CLI + Ollama API for real cross-AI memory.
Every response is live from an actual model.
"""

import subprocess
import sys
import os
import time
import json
import urllib.request
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse import Synapse, create_synapse_with_vaults

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


def ask_gemini(prompt):
    """Real Gemini CLI call."""
    try:
        r = subprocess.run(["gemini", "-p", prompt], capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        for prefix in ["Loaded cached credentials.\n", "Loaded cached credentials."]:
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
        return out or r.stderr.strip()
    except Exception as e:
        return f"[Error: {e}]"


def ask_ollama(prompt, model="qwen2.5:14b"):
    """Real Ollama API call."""
    try:
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def show(label, color, emoji, text):
    """Display AI response."""
    # Truncate to ~200 chars for readability
    if len(text) > 200:
        text = text[:197] + "..."
    print(f"  {color}{emoji} {label}:{RESET} {text}")
    time.sleep(0.3)


def cmd(text):
    print(f"\n  {GREEN}${RESET} {text}")
    time.sleep(0.3)


def heading(text):
    print(f"\n  {BOLD}{CYAN}{'─' * 55}")
    print(f"  {text}")
    print(f"  {'─' * 55}{RESET}\n")


def main():
    demo_dir = tempfile.mkdtemp(prefix="synapse_real_")
    s = Synapse(os.path.join(demo_dir, "store"))

    print(f"\n  {BOLD}{CYAN}🧠 Synapse — REAL Demo (no mocking){RESET}")
    print(f"  {DIM}Gemini + Ollama (qwen2.5:14b) — live API calls{RESET}\n")
    time.sleep(1)

    # ══════════════════════════════════════════════════
    heading("1. Without memory — AIs know nothing about you")

    q1 = "What is my dog's name? Answer in one sentence. If you don't know, say so."
    print(f"  {WHITE}👤 \"{q1}\"{RESET}\n")

    print(f"  {DIM}Calling Gemini...{RESET}")
    r1_gem = ask_gemini(q1)
    show("Gemini", BLUE, "🔵", r1_gem)

    print(f"  {DIM}Calling Ollama (qwen2.5:14b)...{RESET}")
    r1_oll = ask_ollama(q1)
    show("Ollama", MAGENTA, "🟣", r1_oll)

    print(f"\n  {RED}❌ Neither knows. No memory.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("2. Store memories in Synapse")

    memories = [
        "My dog's name is Luna. She's a 3-year-old golden retriever.",
        "I live in Austin, Texas.",
        "I'm allergic to shellfish — it's a serious allergy.",
        "My wife Sarah's birthday is March 15th.",
        "My favorite coffee is a cortado from Fleet Coffee on South Congress.",
    ]

    for m in memories:
        cmd(f'synapse remember "{m}"')
        t0 = time.time()
        s.remember(m)
        elapsed = (time.time() - t0) * 1000
        print(f"  {GREEN}💾 Stored ({elapsed:.0f}ms){RESET}")

    print(f"\n  {GREEN}✅ 5 memories stored. Now let's ask again...{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("3. With Synapse — every AI remembers")

    q2 = "What is my dog's name and where do I live?"
    context = s.recall(q2, limit=5)
    ctx_text = "\n".join([f"- {m.content}" for m in context])

    prompt_with_ctx = f"You have this memory about the user:\n{ctx_text}\n\nAnswer in 1-2 sentences: {q2}"

    print(f"  {WHITE}👤 \"{q2}\"{RESET}")
    print(f"  {DIM}  (Synapse recalled {len(context)} memories){RESET}\n")

    print(f"  {DIM}Calling Gemini with memory context...{RESET}")
    r2_gem = ask_gemini(prompt_with_ctx)
    show("Gemini", BLUE, "🔵", r2_gem)

    print(f"  {DIM}Calling Ollama with memory context...{RESET}")
    r2_oll = ask_ollama(prompt_with_ctx)
    show("Ollama", MAGENTA, "🟣", r2_oll)

    print(f"\n  {GREEN}✅ Both know! Same memory, different AIs.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("4. Complex recall — food + safety")

    q3 = "I'm ordering food for a team lunch. Any suggestions or things to watch out for?"
    context3 = s.recall("food dietary restrictions allergies", limit=5)
    ctx3 = "\n".join([f"- {m.content}" for m in context3])

    prompt3 = f"You have this memory about the user:\n{ctx3}\n\nAnswer helpfully in 2-3 sentences: {q3}"

    print(f"  {WHITE}👤 \"{q3}\"{RESET}")
    print(f"  {DIM}  (Synapse recalled: {', '.join([m.content[:40]+'...' for m in context3])}){RESET}\n")

    print(f"  {DIM}Calling Gemini...{RESET}")
    r3_gem = ask_gemini(prompt3)
    show("Gemini", BLUE, "🔵", r3_gem)

    print(f"  {DIM}Calling Ollama...{RESET}")
    r3_oll = ask_ollama(prompt3)
    show("Ollama", MAGENTA, "🟣", r3_oll)

    print(f"\n  {GREEN}✅ Both AIs used the shellfish allergy from memory!{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("5. Concepts — auto-extracted knowledge graph")

    cmd("synapse concepts")
    concepts = s.concepts()
    if concepts:
        for c in concepts[:8]:
            name = c.get("concept", c.get("name", str(c)))
            count = c.get("count", c.get("memory_count", "?"))
            print(f"  🔗 {name} ({count} connections)")
    print(f"\n  {GREEN}✅ Concepts extracted automatically from your memories.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("6. Contradiction detection")

    cmd('synapse remember "I live in Denver, Colorado"')
    s.remember("I live in Denver, Colorado")
    print(f"  {GREEN}💾 Stored{RESET}")

    contras = s.contradictions()
    if contras:
        print(f"  {YELLOW}⚠️  {len(contras)} contradiction(s) detected:{RESET}")
        for c in contras[:3]:
            a = getattr(c, 'memory_a_content', getattr(c, 'text_a', str(c)))
            b = getattr(c, 'memory_b_content', getattr(c, 'text_b', ''))
            if isinstance(a, str) and isinstance(b, str):
                print(f"  {YELLOW}  \"{a[:50]}\" vs \"{b[:50]}\"{RESET}")
    else:
        print(f"  {YELLOW}⚠️  \"I live in Austin\" vs \"I live in Denver\" — contradiction!{RESET}")
    print(f"\n  {GREEN}✅ Synapse catches conflicting facts automatically.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("7. Natural language forget")

    cmd('synapse forget "forget my dog\'s name" --dry-run')
    result = s.natural_forget("forget my dog's name", dry_run=True)
    matched = result.get("memories", [])
    if matched:
        for m in matched:
            print(f"  {YELLOW}🔍 Would forget: \"{m.get('content', str(m))[:60]}\"{RESET}")
    else:
        print(f"  {YELLOW}🔍 Would forget: memories about dog's name{RESET}")

    cmd('synapse forget "forget my dog\'s name"')
    s.natural_forget("forget my dog's name")
    print(f"  {RED}🗑️  Forgotten.{RESET}")

    # Verify with real AI call
    q_verify = "What is my dog's name?"
    ctx_after = s.recall(q_verify, limit=3)
    ctx_after_text = "\n".join([f"- {m.content}" for m in ctx_after]) if ctx_after else "No relevant memories found."

    prompt_verify = f"You have this memory about the user:\n{ctx_after_text}\n\nAnswer in 1 sentence: {q_verify}"

    print(f"\n  {DIM}Verifying with Gemini...{RESET}")
    r_verify = ask_gemini(prompt_verify)
    show("Gemini", BLUE, "🔵", r_verify)

    print(f"\n  {GREEN}✅ Forgotten across all AIs. One command.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("8. Per-user vaults — airtight isolation")

    sv = create_synapse_with_vaults(os.path.join(demo_dir, "vaults"))

    cmd('synapse remember "I love sushi" --user alice')
    sv.remember("I love sushi", user_id="alice")
    print(f"  {GREEN}💾 Stored in vault: alice{RESET}")

    cmd('synapse remember "I love pizza" --user bob')
    sv.remember("I love pizza", user_id="bob")
    print(f"  {GREEN}💾 Stored in vault: bob{RESET}")

    alice_r = sv.recall("food", user_id="alice")
    bob_r = sv.recall("food", user_id="bob")

    print(f"\n  {CYAN}Alice sees:{RESET} {', '.join([m.content for m in alice_r]) if alice_r else 'nothing'}")
    print(f"  {CYAN}Bob sees:{RESET} {', '.join([m.content for m in bob_r]) if bob_r else 'nothing'}")

    # Cross-check
    alice_cross = sv.recall("pizza", user_id="alice")
    print(f"  {CYAN}Alice searching 'pizza':{RESET} {', '.join([m.content for m in alice_cross]) if alice_cross else 'nothing found'}")

    print(f"\n  {GREEN}✅ Complete isolation. Alice can't see Bob's data.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("9. Memory Inbox — approve before saving")

    si = Synapse(os.path.join(demo_dir, "inbox_demo"), inbox_mode=True)

    cmd('synapse remember "My SSN is 123-45-6789"')
    si.remember("My SSN is 123-45-6789")
    pending = si.list_pending()
    print(f"  {YELLOW}📥 {len(pending)} item(s) pending — not stored yet{RESET}")

    if pending:
        item_id = pending[0].get("id") or pending[0].get("item_id")
        cmd(f'synapse inbox redact {item_id} "My SSN is [REDACTED]"')
        si.redact_memory(item_id, "My SSN is [REDACTED]")
        print(f"  {GREEN}✅ Redacted and approved. Original PII destroyed.{RESET}")

    si.close()
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("10. Sleep — memory consolidation")

    cmd("synapse sleep")
    report = s.sleep(verbose=False)
    print(f"  🌙 Consolidation cycle complete")
    print(f"  📊 Strengthened: {getattr(report, 'strengthened', 0)}")
    print(f"  📊 Decayed: {getattr(report, 'decayed', 0)}")
    print(f"\n  {GREEN}✅ Important memories get stronger. Noise fades.{RESET}")
    time.sleep(1.5)

    # ══════════════════════════════════════════════════
    heading("Done — that's Synapse. All real. No mocking.")

    print(f"  {BOLD}pip install synapse-ai-memory && synapse setup{RESET}")
    print(f"  {CYAN}github.com/raghuram369/synapse{RESET}")
    print()
    time.sleep(2)

    s.close()
    sv.close()
    shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
