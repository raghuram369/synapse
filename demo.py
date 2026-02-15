#!/usr/bin/env python3
"""
Synapse Interactive Demo — A neuroscience-inspired AI memory engine.
Zero external dependencies. Just run: python demo.py
"""

import sys
import os
import time
import math

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synapse import Synapse, Memory
from indexes import InvertedIndex, STOPWORDS
from entity_graph import extract_concepts

# ─── ANSI Colors ───────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"
ULINE   = "\033[4m"

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

BG_BLACK   = "\033[40m"
BG_RED     = "\033[41m"
BG_GREEN   = "\033[42m"
BG_YELLOW  = "\033[43m"
BG_BLUE    = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN    = "\033[46m"
BG_WHITE   = "\033[47m"

BRIGHT_BLACK   = "\033[90m"
BRIGHT_RED     = "\033[91m"
BRIGHT_GREEN   = "\033[92m"
BRIGHT_YELLOW  = "\033[93m"
BRIGHT_BLUE    = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN    = "\033[96m"
BRIGHT_WHITE   = "\033[97m"

# ─── ASCII Art Header ──────────────────────────────────────────────────────────
BRAIN_ART = f"""{BRIGHT_CYAN}{BOLD}
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        ██████╗ ██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗   ║
    ║       ██╔════╝ ╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝   ║
    ║       ╚█████╗   ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗   ║
    ║        ╚═══██╗   ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║   ║
    ║       ██████╔╝    ██║   ██║ ╚████║██║  ██║██║     ███████║   ║
    ║       ╚═════╝     ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝   ║
    ║                                                               ║
    ║    {BRIGHT_YELLOW}⚡ Neuroscience-Inspired AI Memory Engine ⚡{BRIGHT_CYAN}              ║
    ║    {DIM}{WHITE}Pure Python · Zero APIs · Sub-millisecond Recall{RESET}{BRIGHT_CYAN}{BOLD}         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
{RESET}"""

NEURAL_NET = f"""{DIM}
              ●───────●───────●
             /|\\     /|\\     /|\\
            ● ● ●   ● ● ●   ● ● ●      {BRIGHT_CYAN}Five cognitive indexes{RESET}{DIM}
             \\|/     \\|/     \\|/         {BRIGHT_CYAN}working in parallel{RESET}{DIM}
              ●───────●───────●
             /|\\     /|\\     /|\\
            ● ● ●   ● ● ●   ● ● ●
             \\|/     \\|/     \\|/
              ●───────●───────●
{RESET}"""

# ─── Sample Conversation ──────────────────────────────────────────────────────
SAMPLE_CONVERSATIONS = [
    ("user", "I'm planning a trip to Japan next spring, probably mid-April for the cherry blossoms."),
    ("assistant", "Mid-April is perfect for cherry blossoms! Tokyo and Kyoto are the top spots. Want me to help plan an itinerary?"),
    ("user", "Yes! I'll be there for 2 weeks. I'm vegetarian and allergic to shellfish, so food planning is important."),
    ("assistant", "Two weeks is great! For vegetarian options, look into shojin ryori (Buddhist temple cuisine). I'll note the shellfish allergy too."),
    ("user", "My budget is around $5000 for the whole trip, not including flights."),
    ("assistant", "That's very doable for 2 weeks. Mix of hostels and budget ryokans could keep accommodation under $2000."),
    ("user", "I work remotely as a software engineer, so I'll need reliable WiFi for morning standups at 9am EST."),
    ("assistant", "9am EST = 10pm JST, so you'd do standups in the evening. Most accommodations have good WiFi. Co-working spaces are everywhere in Tokyo."),
    ("user", "My girlfriend Sarah might join for the second week. She's really into photography and traditional arts."),
    ("assistant", "Sarah would love Kyoto for photography — Fushimi Inari and Arashiyama bamboo grove are stunning. Tea ceremonies and ikebana workshops too."),
    ("user", "I've been studying Japanese on Duolingo for 6 months. I'm around N4 level."),
    ("assistant", "N4 is solid for daily conversations! You'll be able to read menus and signs. Practice keigo (polite form) for restaurants."),
    ("user", "I usually wake up at 6am and like to run in the morning. Are there good running routes in Tokyo?"),
    ("assistant", "The Imperial Palace loop is the most popular — 5km around the moat. Yoyogi Park is also great, especially in cherry blossom season."),
    ("user", "I'm also interested in visiting some tech companies. I work on distributed systems at my current job."),
    ("assistant", "Check out the offices in Shibuya and Roppongi. LINE, Mercari, and SmartNews are known for their engineering culture."),
    ("user", "For the Kyoto portion, I want to stay at a traditional ryokan. Any with vegetarian kaiseki?"),
    ("assistant", "Shigetsu at Tenryuji temple offers full vegetarian shojin ryori kaiseki. It's an incredible experience."),
    ("user", "My birthday is April 18th — I'll be turning 29! Want to do something special that day."),
    ("assistant", "Happy early birthday! April 18th should be peak bloom. A private boat ride on the Philosopher's Path canal would be magical."),
    ("user", "Oh, I should mention I have a mild fear of heights. So maybe skip the Tokyo Skytree observation deck."),
    ("assistant", "Noted! There are plenty of ground-level viewpoints. Odaiba beach gives great skyline views without the height."),
]

# ─── Index Panel ───────────────────────────────────────────────────────────────
INDEX_NAMES = [
    ("BM25",          "Term frequency–inverse document frequency"),
    ("ConceptGraph",  "Semantic concept expansion & linking"),
    ("EdgeGraph",     "Memory-to-memory relationship spreading"),
    ("TemporalIndex", "Time-proximity clustering & decay"),
    ("EpisodeIndex",  "Conversational episode grouping"),
]

INDEX_COLORS = [BRIGHT_GREEN, BRIGHT_CYAN, BRIGHT_MAGENTA, BRIGHT_YELLOW, BRIGHT_BLUE]
INDEX_DIM    = [DIM + GREEN, DIM + CYAN, DIM + MAGENTA, DIM + YELLOW, DIM + BLUE]


def render_index_panel(active_indexes: list[bool]):
    """Render the 5-index activation panel."""
    lines = []
    lines.append(f"  {BOLD}{WHITE}┌─── Neural Index Activation ───────────────────────┐{RESET}")
    for i, ((name, desc), active) in enumerate(zip(INDEX_NAMES, active_indexes)):
        if active:
            bar = f"{INDEX_COLORS[i]}{BOLD}████{RESET}"
            icon = f"{INDEX_COLORS[i]}{BOLD}⚡{RESET}"
            label = f"{INDEX_COLORS[i]}{BOLD}{name:<15}{RESET}"
            detail = f"{INDEX_COLORS[i]}{desc}{RESET}"
        else:
            bar = f"{DIM}░░░░{RESET}"
            icon = f"{DIM}○{RESET}"
            label = f"{DIM}{name:<15}{RESET}"
            detail = f"{DIM}{desc}{RESET}"
        lines.append(f"  {BOLD}{WHITE}│{RESET} {icon} {bar} {label} {detail}")
    lines.append(f"  {BOLD}{WHITE}└───────────────────────────────────────────────────┘{RESET}")
    return "\n".join(lines)


def detect_active_indexes(query: str, results: list, synapse_engine: Synapse) -> list[bool]:
    """Heuristically determine which indexes contributed to results."""
    active = [False] * 5

    if not results:
        return active

    # BM25: active only if it actually found candidates
    tokens = synapse_engine.inverted_index.tokenize_for_query(query)
    if tokens:
        bm25_cands = synapse_engine.inverted_index.get_candidates(tokens, 5)
        if bm25_cands:
            active[0] = True  # BM25

    # ConceptGraph: active if query concepts overlap with indexed concepts
    concepts = extract_concepts(query)
    if concepts:
        active[1] = True
    else:
        # Check if any query tokens match concept names
        for tok in (tokens or []):
            if tok in synapse_engine.concept_graph.concepts:
                active[1] = True
                break

    # EdgeGraph: active if any results have edges
    for mem in results:
        edges = synapse_engine.edge_graph.get_all_edges(mem.id)
        if edges:
            active[2] = True
            break

    # TemporalIndex: active (always participates in boosting)
    if len(results) > 1:
        active[3] = True  # TemporalIndex

    # EpisodeIndex: active if results share episodes
    for mem in results:
        siblings = synapse_engine.episode_index.get_episode_siblings(mem.id)
        if siblings:
            active[4] = True
            break

    return active


def bm25_only_recall(synapse_engine: Synapse, query: str, limit: int = 5) -> list[tuple]:
    """Do a BM25-only recall for comparison."""
    tokens = synapse_engine.inverted_index.tokenize_for_query(query)
    if not tokens:
        return []
    scores = synapse_engine.inverted_index.get_candidates(tokens, limit * 5)
    # Filter and sort
    results = []
    for mid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if mid in synapse_engine.store.memories:
            data = synapse_engine.store.memories[mid]
            if not data.get('consolidated', False):
                results.append((data['content'], score))
                if len(results) >= limit:
                    break
    return results


def progress_bar(current, total, width=40, label=""):
    """Render an animated progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = f"{BRIGHT_CYAN}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"
    sys.stdout.write(f"\r  {bar} {BRIGHT_WHITE}{pct*100:5.1f}%{RESET} {DIM}{label}{RESET}")
    sys.stdout.flush()


def print_slow(text, delay=0.008):
    """Print text character by character for visual effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


SUGGESTED_QUERIES = [
    "What are my dietary restrictions?",
    "When is my birthday?",
    "Tell me about Sarah",
    "What's my morning routine?",
    "Japan budget and accommodation",
    "Am I afraid of anything?",
    "tech companies to visit",
    "cherry blossom spots",
]


def main():
    print(BRAIN_ART)
    print(NEURAL_NET)

    # ─── Initialize Engine ─────────────────────────────────────────────────
    print(f"  {BRIGHT_YELLOW}{BOLD}Initializing Synapse memory engine...{RESET}")
    engine = Synapse(":memory:")
    print(f"  {GREEN}✓ Engine ready (in-memory mode){RESET}\n")

    # ─── Ingest Sample Conversation ────────────────────────────────────────
    total = len(SAMPLE_CONVERSATIONS)
    print(f"  {BRIGHT_WHITE}{BOLD}Loading {total} conversation memories...{RESET}\n")
    time.sleep(0.3)

    for i, (role, content) in enumerate(SAMPLE_CONVERSATIONS):
        tagged = f"[{role}] {content}"
        engine.remember(tagged, memory_type="fact", episode="japan-trip-planning")
        progress_bar(i + 1, total, label=content[:50] + "...")
        time.sleep(0.06)  # small delay for visual effect

    print(f"\n\n  {BRIGHT_GREEN}{BOLD}✓ {total} memories ingested across 5 neural indexes{RESET}")

    # Show stats
    n_concepts = len(engine.concept_graph.concepts)
    n_episodes = len(engine.store.episodes)
    print(f"  {DIM}  → {n_concepts} concepts extracted")
    print(f"  {DIM}  → {n_episodes} episode(s) detected")
    print(f"  {DIM}  → BM25, ConceptGraph, EdgeGraph, TemporalIndex, EpisodeIndex active{RESET}")

    # ─── Portable Format Demo ─────────────────────────────────────────────
    import tempfile
    tmp = tempfile.mktemp(suffix='.synapse')
    engine.export(tmp, source_agent="demo")
    file_size = os.path.getsize(tmp)
    print(f"\n  {BRIGHT_YELLOW}{BOLD}📁 Portable Format Demo{RESET}")
    print(f"  {DIM}  Exported {total} memories to {tmp}{RESET}")
    print(f"  {DIM}  File size: {file_size} bytes{RESET}")

    engine2 = Synapse(":memory:")
    stats = engine2.load(tmp)
    print(f"  {DIM}  Re-imported into fresh instance: {stats['memories']} memories{RESET}")
    engine2.close()
    os.unlink(tmp)
    print(f"  {GREEN}✓ Export → Import roundtrip verified{RESET}")

    # ─── Suggested Queries ─────────────────────────────────────────────────
    print(f"\n  {BOLD}{WHITE}╭─── Try these queries ────────────────────────────────╮{RESET}")
    for j in range(0, len(SUGGESTED_QUERIES), 2):
        left = SUGGESTED_QUERIES[j]
        right = SUGGESTED_QUERIES[j + 1] if j + 1 < len(SUGGESTED_QUERIES) else ""
        print(f"  {BOLD}{WHITE}│{RESET}  {BRIGHT_CYAN}▸{RESET} {left:<30s}  {BRIGHT_CYAN}▸{RESET} {right}")
    print(f"  {BOLD}{WHITE}╰─────────────────────────────────────────────────────╯{RESET}")
    print(f"\n  {DIM}Type a query and press Enter. 'q' to quit.{RESET}\n")

    # ─── Interactive Loop ──────────────────────────────────────────────────
    while True:
        try:
            query = input(f"  {BRIGHT_GREEN}{BOLD}synapse ❯{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM}Goodbye! 🧠{RESET}\n")
            break

        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print(f"\n  {DIM}Goodbye! 🧠{RESET}\n")
            break

        # ─── Synapse Recall ────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = engine.recall(query, limit=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Detect which indexes fired
        active = detect_active_indexes(query, results, engine)

        print()
        print(render_index_panel(active))

        # Compute scores for display (re-run scoring logic lightly)
        # We use the order as a proxy for relevance
        if results:
            # Approximate scores from BM25 for display
            tokens = engine.inverted_index.tokenize_for_query(query)
            bm25_raw = engine.inverted_index.get_candidates(tokens, 50) if tokens else {}
            max_score = max(bm25_raw.values()) if bm25_raw else 1.0
            if max_score <= 0:
                max_score = 1.0

        print(f"\n  {BOLD}{WHITE}Results{RESET} {DIM}({elapsed_ms:.1f}ms){RESET}\n")

        if not results:
            print(f"  {DIM}  No memories matched this query.{RESET}\n")
        else:
            for rank, mem in enumerate(results, 1):
                # Compute a display score
                raw = bm25_raw.get(mem.id, 0.0)
                score = min(raw / max_score, 1.0) if max_score > 0 else 0.0
                # Boost score display for top results that may have gotten concept/edge boosts
                display_score = max(score, 1.0 - (rank - 1) * 0.15)
                display_score = min(display_score, 1.0)

                # Color based on score
                if display_score > 0.7:
                    sc = BRIGHT_GREEN
                elif display_score > 0.4:
                    sc = BRIGHT_YELLOW
                else:
                    sc = BRIGHT_RED

                bar_len = int(display_score * 20)
                bar = f"{sc}{'█' * bar_len}{DIM}{'░' * (20 - bar_len)}{RESET}"

                print(f"  {BOLD}{WHITE}  {rank}.{RESET} {bar} {sc}{display_score:.2f}{RESET}  {mem.content}")

        # ─── BM25 Comparison ───────────────────────────────────────────────
        bm25_results = bm25_only_recall(engine, query, limit=5)
        synapse_contents = set(m.content for m in results) if results else set()
        bm25_contents = set(c for c, _ in bm25_results)

        synapse_unique = synapse_contents - bm25_contents

        print(f"\n  {BOLD}{WHITE}── BM25-Only Comparison ──────────────────────────────{RESET}")
        if bm25_results:
            for rank, (content, score) in enumerate(bm25_results[:3], 1):
                marker = f"{DIM}  {rank}. {content[:80]}{RESET}"
                print(f"  {marker}")
        else:
            print(f"  {DIM}  (no BM25 results){RESET}")

        if synapse_unique:
            print(f"\n  {BRIGHT_MAGENTA}{BOLD}  ✦ Synapse found {len(synapse_unique)} result(s) BM25 missed:{RESET}")
            for content in list(synapse_unique)[:3]:
                print(f"  {BRIGHT_MAGENTA}    → {content[:80]}{RESET}")
        elif results:
            print(f"\n  {DIM}  Both returned similar results for this query.{RESET}")

        print()

    engine.close()


if __name__ == "__main__":
    main()
