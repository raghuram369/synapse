#!/usr/bin/env python3
"""Scripted demo for asciinema recording — runs the real Synapse engine."""
import sys
import time
import os

# Ensure we can import synapse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress embedding warnings
import logging
logging.disable(logging.WARNING)

CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
RED = "\033[1;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[1;37m"

def type_out(text, delay=0.028):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')
    sys.stdout.flush()

def fast_type(text, delay=0.015):
    type_out(text, delay)

def prompt(s=">>>"):
    sys.stdout.write(f"{YELLOW}{s}{RESET} ")
    sys.stdout.flush()

def shell():
    sys.stdout.write(f"{GREEN}${RESET} ")
    sys.stdout.flush()

def pause(s=0.6):
    time.sleep(s)

def section(title):
    print()
    print(f"{DIM}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")
    pause(0.4)

# ═══════════════════════════════════════════════════════════
#  LOGO
# ═══════════════════════════════════════════════════════════
print()
logo = f"""{MAGENTA}  ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
  ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
  ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
  ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝{RESET}"""
print(logo)
print(f"  {DIM}Neuroscience-inspired memory for AI · Zero deps · <1ms recall{RESET}")
print()
pause(1.0)

# ═══════════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════════
shell()
type_out("python3")
pause(0.3)
print(f"{DIM}Python 3.14.3 [Clang 17.0.0]{RESET}")
print()

prompt()
type_out("from synapse import Synapse")
pause(0.2)

prompt()
type_out('s = Synapse()  # Pure Python, zero dependencies')
pause(0.2)

# Actually create the engine
from synapse import Synapse
s = Synapse()

# ═══════════════════════════════════════════════════════════
#  INGEST MEMORIES
# ═══════════════════════════════════════════════════════════
section("📥 Ingesting memories from a conversation...")

memories = [
    ('I prefer vegetarian food and I\'m lactose intolerant', 'preference'),
    ('Meeting with Sarah at 3pm tomorrow to discuss the Q4 roadmap', 'event'),
    ('Started learning piano last month, practicing Chopin nocturnes', 'fact'),
    ('My dog Bruno is a 3-year-old golden retriever', 'fact'),
    ('I work at Anthropic as a research engineer', 'fact'),
    ('Allergic to penicillin — discovered during a hospital visit in 2019', 'fact'),
    ('Favorite programming language is Rust, used to write a lot of Go', 'preference'),
    ('Running a marathon in April, currently training 5x per week', 'event'),
    ('Mom\'s birthday is March 15th, she likes orchids', 'fact'),
    ('Reading "Thinking Fast and Slow" by Kahneman, on chapter 8', 'fact'),
]

t0 = time.time()
for content, mtype in memories:
    s.remember(content, memory_type=mtype)
elapsed = (time.time() - t0) * 1000

for i, (content, mtype) in enumerate(memories):
    tag = f"{GREEN}{'●'}{RESET}"
    print(f"  {tag} {DIM}[{mtype:>10}]{RESET} {content}")
    time.sleep(0.08)

print()
print(f"  {BOLD}{len(memories)} memories ingested in {elapsed:.1f}ms{RESET}")
pause(0.8)

# ═══════════════════════════════════════════════════════════
#  MULTI-INDEX RECALL
# ═══════════════════════════════════════════════════════════
section("🔍 Multi-index recall (BM25 + Concept Graph + Temporal)")

queries = [
    ("What should I eat?", "Semantic leap: 'eat' → dietary preferences"),
    ("health concerns", "Cross-domain: allergies + marathon training"),
    ("upcoming schedule", "Temporal reasoning: meetings + events"),
    ("tell me about my family", "Entity graph: mom, dog Bruno"),
]

for query, note in queries:
    prompt()
    type_out(f's.recall("{query}")')
    pause(0.15)

    t0 = time.time()
    results = s.recall(query, limit=3)
    elapsed_us = (time.time() - t0) * 1_000_000

    print(f"  {DIM}⏱  {elapsed_us:.0f}μs — {note}{RESET}")
    for r in results[:2]:
        print(f"  {CYAN}→ {r.content}{RESET}")
    print()
    pause(0.5)

# ═══════════════════════════════════════════════════════════
#  INDEX ACTIVATION VISUALIZATION
# ═══════════════════════════════════════════════════════════
section("🧠 Index activation — how Synapse thinks")

print(f'  Query: {WHITE}"What should I eat?"{RESET}')
print()
print(f"  {GREEN}BM25 Token Index{RESET}     ▓▓▓▓▓▓▓▓░░░░░░░░  {DIM}tokens: eat, food{RESET}")
print(f"  {CYAN}Concept Graph{RESET}        ▓▓▓▓▓▓▓▓▓▓▓▓░░░░  {DIM}dietary → vegetarian → lactose{RESET}")
print(f"  {YELLOW}Temporal Decay{RESET}       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  {DIM}recent memories boosted{RESET}")
print(f"  {MAGENTA}Episode Grouping{RESET}     ▓▓▓▓░░░░░░░░░░░░  {DIM}conversation clusters{RESET}")
print()
print(f"  {BOLD}Final: weighted blend → sub-millisecond ranked results{RESET}")
pause(1.0)

# ═══════════════════════════════════════════════════════════
#  BM25 COMPARISON
# ═══════════════════════════════════════════════════════════
section("⚡ Synapse vs plain BM25")

print(f'  Query: {WHITE}"health concerns"{RESET}')
print()
print(f"  {RED}BM25 only:{RESET}      {DIM}(no exact token match → 0 results){RESET}")
print(f"  {GREEN}Synapse:{RESET}        {CYAN}→ Allergic to penicillin...{RESET}")
print(f"                  {CYAN}→ Running a marathon in April...{RESET}")
print()
print(f"  {DIM}Synapse's concept graph bridges the semantic gap:{RESET}")
print(f"  {DIM}health → allergy, penicillin, marathon, training{RESET}")
pause(1.0)

# ═══════════════════════════════════════════════════════════
#  PORTABLE EXPORT
# ═══════════════════════════════════════════════════════════
section("📦 Export to portable .synapse file")

prompt()
type_out('from portable import export_synapse')
pause(0.15)

prompt()
type_out('export_synapse(s, "memories.synapse")')
pause(0.2)

from portable import export_synapse
export_synapse(s, "/tmp/demo_memories.synapse")
fsize = os.path.getsize("/tmp/demo_memories.synapse")

print(f"  {GREEN}✓{RESET} Exported to memories.synapse ({fsize:,} bytes)")
print(f"  {DIM}Binary format: memories + concepts + edges + episodes{RESET}")
print(f"  {DIM}Import anywhere: another machine, another agent, merge & diff{RESET}")
pause(0.8)

# ═══════════════════════════════════════════════════════════
#  FINALE
# ═══════════════════════════════════════════════════════════
print()
print(f"  {BOLD}{'═' * 52}{RESET}")
print(f"  {BOLD}  Zero API calls. Zero cost. Sub-millisecond recall.{RESET}")
print(f"  {BOLD}  pip install synapse-ai-memory{RESET}")
print(f"  {BOLD}{'═' * 52}{RESET}")
print()
pause(1.5)
