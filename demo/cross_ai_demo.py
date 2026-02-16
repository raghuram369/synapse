#!/usr/bin/env python3
"""
Synapse Cross-AI Demo — The Real Value Prop
=============================================
Tell Claude a fact → Synapse stores it → ChatGPT/Gemini recall it.
No CLI feeding. Shows actual MCP tool calls.
"""

import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse import Synapse

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
BG_DARK = "\033[48;5;236m"
BG_CLAUDE = "\033[48;5;53m"
BG_GPT = "\033[48;5;22m"
BG_GEMINI = "\033[48;5;17m"

TYPING_SPEED = 0.025
PAUSE_SHORT = 0.5
PAUSE_MED = 1.0
PAUSE_LONG = 2.0


def type_text(text, speed=TYPING_SPEED):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()


def heading(text):
    print()
    print(f"  {BOLD}{CYAN}{'─' * 55}{RESET}")
    type_text(f"  {BOLD}{CYAN}{text}{RESET}", speed=0.01)
    print(f"  {BOLD}{CYAN}{'─' * 55}{RESET}")
    print()
    time.sleep(PAUSE_SHORT)


def app_header(name, color, bg):
    print(f"  {bg}{color}{BOLD} {name} {RESET}")
    print(f"  {DIM}{'─' * 55}{RESET}")


def user_msg(text):
    time.sleep(PAUSE_SHORT)
    sys.stdout.write(f"  {WHITE}👤 You: {RESET}")
    type_text(text, speed=0.03)
    time.sleep(PAUSE_MED)


def ai_msg(name, color, text):
    sys.stdout.write(f"  {color}{name}: {RESET}")
    type_text(text, speed=0.015)
    time.sleep(PAUSE_SHORT)


def tool_call(name, args):
    print(f"  {DIM}  ┌ 🔧 Tool: {name}{RESET}")
    print(f"  {DIM}  │  {args}{RESET}")
    print(f"  {DIM}  └ ✅ done{RESET}")
    time.sleep(PAUSE_SHORT)


def main():
    demo_dir = tempfile.mkdtemp(prefix="synapse_cross_")
    s = Synapse(os.path.join(demo_dir, "shared_memory"))

    # ── Title ───────────────────────────────────────────
    print()
    print(f"  {BOLD}{CYAN}🧠 Synapse — One Memory, Every AI{RESET}")
    print(f"  {DIM}Tell Claude a fact. ChatGPT remembers it.{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    # ── Scene 1: Chatting with Claude Desktop ───────────
    heading("Scene 1: Talking to Claude Desktop")

    app_header("Claude Desktop", MAGENTA, BG_CLAUDE)
    print()

    user_msg("Hey Claude, just so you know — I'm deathly allergic to shellfish.")

    # Claude uses MCP to store
    print(f"\n  {DIM}  Claude calls Synapse MCP...{RESET}")
    time.sleep(PAUSE_SHORT)
    tool_call("synapse.remember", '"I\'m deathly allergic to shellfish"')
    s.remember("User is deathly allergic to shellfish. This is a serious health concern.")

    ai_msg("🟣 Claude", MAGENTA,
           "Got it, I've saved that. I'll always keep that in mind — especially")
    print(f"  {MAGENTA}         if food or restaurants come up. Stay safe! 🙏{RESET}")
    time.sleep(PAUSE_MED)

    user_msg("Also, my wife Sarah's birthday is March 15th. Don't let me forget!")

    print(f"\n  {DIM}  Claude calls Synapse MCP...{RESET}")
    time.sleep(PAUSE_SHORT)
    tool_call("synapse.remember", '"Wife Sarah\'s birthday is March 15th"')
    s.remember("User's wife Sarah has her birthday on March 15th. Important to remember.")

    ai_msg("🟣 Claude", MAGENTA,
           "Noted! I'll remind you when March rolls around. Sarah's a lucky one 😊")
    time.sleep(PAUSE_LONG)

    # ── Scene 2: Switch to Cursor (GPT) ─────────────────
    heading("Scene 2: Next day, open Cursor with GPT")

    app_header("Cursor (GPT-4o)", GREEN, BG_GPT)
    print()
    print(f"  {DIM}  (Different app. Different AI. Same Synapse memory.){RESET}")
    print()

    user_msg("I'm ordering lunch for the team. Can you suggest something for me?")

    # GPT recalls via MCP
    print(f"\n  {DIM}  GPT calls Synapse MCP...{RESET}")
    time.sleep(PAUSE_SHORT)
    tool_call("synapse.recall", '"food preferences and dietary restrictions"')

    results = s.recall("food preferences dietary restrictions allergies", limit=3)
    for r in results:
        print(f"  {DIM}  │ 🧠 \"{r.content[:60]}...\"{RESET}")
    print(f"  {DIM}  └ returned {len(results)} memories{RESET}")
    time.sleep(PAUSE_SHORT)

    ai_msg("🟢 GPT", GREEN,
           "Sure! Just a heads up — I know you're allergic to shellfish,")
    print(f"  {GREEN}        so I'll avoid any seafood-heavy options. How about:{RESET}")
    print(f"  {GREEN}        • Mediterranean platter (hummus, falafel, grilled chicken){RESET}")
    print(f"  {GREEN}        • Thai basil stir-fry (no shrimp){RESET}")
    print(f"  {GREEN}        • Build-your-own burrito bowl{RESET}")
    time.sleep(PAUSE_MED)

    user_msg("Wait — how did you know about my allergy? I told Claude, not you!")

    ai_msg("🟢 GPT", GREEN,
           "Synapse! Your memory is shared across all your AI tools. You told")
    print(f"  {GREEN}        Claude yesterday, and I can see it too. One memory, every AI.{RESET}")
    time.sleep(PAUSE_LONG)

    # ── Scene 3: Switch to Gemini in VS Code ────────────
    heading("Scene 3: Later that week, VS Code with Gemini")

    app_header("VS Code + Continue (Gemini)", BLUE, BG_GEMINI)
    print()
    print(f"  {DIM}  (Third app. Third AI. Same memory.){RESET}")
    print()

    user_msg("Hey, I need to plan something special. Any ideas for mid-March?")

    print(f"\n  {DIM}  Gemini calls Synapse MCP...{RESET}")
    time.sleep(PAUSE_SHORT)
    tool_call("synapse.recall", '"mid-March plans events"')

    results2 = s.recall("March plans birthday events", limit=3)
    for r in results2:
        print(f"  {DIM}  │ 🧠 \"{r.content[:60]}...\"{RESET}")
    print(f"  {DIM}  └ returned {len(results2)} memories{RESET}")
    time.sleep(PAUSE_SHORT)

    ai_msg("🔵 Gemini", BLUE,
           "March 15th is Sarah's birthday! 🎂 Here are some ideas:")
    print(f"  {BLUE}          • Book her favorite restaurant (just make sure no{RESET}")
    print(f"  {BLUE}            shellfish on the menu — I know about your allergy!){RESET}")
    print(f"  {BLUE}          • Weekend getaway — Austin has great spots this time of year{RESET}")
    print(f"  {BLUE}          • Surprise party with friends{RESET}")
    time.sleep(PAUSE_MED)

    user_msg("That's incredible. I never told you any of this.")

    ai_msg("🔵 Gemini", BLUE,
           "You didn't have to! Synapse connects all your AI tools to one memory.")
    print(f"  {BLUE}          Tell any AI once — they all remember. ✨{RESET}")
    time.sleep(PAUSE_LONG)

    # ── Scene 4: Forget ─────────────────────────────────
    heading("Scene 4: Privacy — Forget across everything")

    app_header("Any terminal", WHITE, BG_DARK)
    print()

    sys.stdout.write(f"  {GREEN}${RESET} ")
    type_text('synapse forget "forget my allergy info"')
    time.sleep(PAUSE_SHORT)

    s.natural_forget("forget my allergy")
    print(f"  {RED}🗑️  Forgotten across all AI tools.{RESET}")
    print(f"  {DIM}  Claude won't know. GPT won't know. Gemini won't know.{RESET}")
    time.sleep(PAUSE_LONG)

    # ── Finale ──────────────────────────────────────────
    heading("That's Synapse")

    print(f"  {BOLD}Tell one AI. They all remember.{RESET}")
    print(f"  {BOLD}Forget once. They all forget.{RESET}")
    print()
    print(f"  {DIM}✦ Works via MCP — Claude Desktop, Cursor, Windsurf, VS Code{RESET}")
    print(f"  {DIM}✦ Private — everything stays on your machine{RESET}")
    print(f"  {DIM}✦ Zero cloud. Zero API calls. Zero config.{RESET}")
    print()
    print(f"  {WHITE}pip install synapse-ai-memory && synapse setup{RESET}")
    print(f"  {CYAN}github.com/raghuram369/synapse{RESET}")
    print()
    time.sleep(PAUSE_LONG)

    s.close()
    shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
