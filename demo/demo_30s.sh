#!/usr/bin/env bash
# Synapse AI Memory — 30-second demo script
# Produces: demo_output/report.md, demo_output/card.md, demo_output/card.html
# Usage: bash demo/demo_30s.sh [--gif]
set -euo pipefail

DEMO_DIR="${1:-demo_output}"
mkdir -p "$DEMO_DIR"

echo "🧠 Synapse AI Memory — 30-Second Demo"
echo "══════════════════════════════════════"
echo ""

# Step 1: Install (skip if already installed)
if ! python3 -c "import synapse" 2>/dev/null; then
    echo "📦 Installing..."
    pip install synapse-ai-memory -q
fi
echo "✅ synapse-ai-memory installed"
echo ""

# Step 2: Remember some facts
echo "💾 Remembering facts..."
python3 -c "
from synapse import Synapse
s = Synapse('$DEMO_DIR/demo.synapse')
s.remember('User prefers dark mode and compact layouts')
s.remember('User works on Project Atlas — a real-time trading dashboard')
s.remember('User is vegetarian, allergic to shellfish')
s.remember('User lives in Denver, Colorado', valid_from='2024-06')
s.remember('User lived in Austin, Texas', valid_from='2024-01', valid_to='2024-06')
s.remember('User likes Starbucks coffee')
s.remember('User switched to a local roaster and dislikes Starbucks now')
s.remember('Project Atlas uses PostgreSQL and Redis')
s.remember('Weekly standup is Monday 10am CT')
s.remember('User prefers Python over JavaScript for backend work')
print('  10 memories stored')
s.close()
"
echo ""

# Step 3: Run benchmark
echo "📊 Running recall challenge..."
python3 -c "
import sys, os
sys.path.insert(0, '.')
from bench.consumer_bench import run_all, save_artifacts, generate_report_md
results = run_all()
save_artifacts(results, '$DEMO_DIR')
# Quick stats
r = results.get('recall', {})
if r:
    w_tok = r['without']['tokens_injected']
    s_tok = r['with']['tokens_injected']
    reduction = round((1 - s_tok / w_tok) * 100) if w_tok else 0
    print(f'  Token reduction: {reduction}%')
    print(f'  Recall: {r[\"with\"][\"relevant_found\"]}/{r[\"total_relevant\"]} relevant facts')
tt = results.get('timetravel', {})
if tt:
    print(f'  Time travel: {tt[\"correct\"]}/{tt[\"total\"]} correct')
c = results.get('contradictions', {})
if c:
    print(f'  Contradictions: {c[\"detected\"]}/{c[\"total\"]} detected')
"
echo ""

# Step 4: Generate shareable card
echo "🃏 Generating share card..."
python3 -c "
from synapse import Synapse
from card_share import share_card, generate_caption
s = Synapse('$DEMO_DIR/demo.synapse')
card_md = share_card(s, 'Project Atlas overview', budget=1200, policy='balanced')
card_html = share_card(s, 'Project Atlas overview', budget=1200, policy='balanced', format='html')
with open('$DEMO_DIR/card.md', 'w') as f: f.write(card_md)
with open('$DEMO_DIR/card.html', 'w') as f: f.write(card_html)
pack = s.compile_context('Project Atlas overview', budget=1200)
caption = generate_caption('Project Atlas', pack)
with open('$DEMO_DIR/caption.txt', 'w') as f: f.write(caption)
print('  card.md + card.html + caption.txt generated')
s.close()
"
echo ""

# Step 5: Show what was generated
echo "══════════════════════════════════════"
echo "✅ Demo complete! Artifacts:"
for f in "$DEMO_DIR"/*.md "$DEMO_DIR"/*.html "$DEMO_DIR"/*.json "$DEMO_DIR"/*.txt; do
    [ -f "$f" ] && echo "   📄 $f ($(wc -c < "$f" | tr -d ' ') bytes)"
done
echo ""
echo "📋 Caption (copy & paste for X/HN/Reddit):"
echo "──────────────────────────────────────"
cat "$DEMO_DIR/caption.txt" 2>/dev/null || echo "(no caption generated)"
echo "──────────────────────────────────────"
