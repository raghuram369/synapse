"""Local read-only inspector web UI for Synapse memory stores."""

from __future__ import annotations

import json
import webbrowser
from collections import defaultdict
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

from context_pack import ContextPack
from entity_graph import extract_concepts
from synapse import Synapse


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Synapse AI Memory Inspector</title>
    <style>
      :root {
        --bg: #04070d;
        --panel: #0b1220;
        --panel-soft: #121a2b;
        --line: #24304b;
        --text: #d2def9;
        --muted: #8ea0c7;
        --accent: #7aa2ff;
        --ok: #63f3b7;
        --warn: #ffce64;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        padding: 0;
        font-family: "Fira Code", "JetBrains Mono", Menlo, Consolas, monospace;
        background: radial-gradient(circle at 10% 0%, #102040 0%, #04070d 45%, #020409 100%);
        color: var(--text);
      }
      .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
        display: grid;
        gap: 1rem;
      }
      h1, h2 {
        margin: 0 0 0.5rem 0;
        letter-spacing: 0.06em;
      }
      section {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
        border-radius: 10px;
        padding: 1rem;
        animation: pop 260ms ease-out;
      }
      .grid-2 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1rem;
      }
      .grid-3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.8rem;
      }
      .mono {
        font-size: 13px;
      }
      .chip {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background: #1c2740;
        border: 1px solid var(--line);
        border-radius: 999px;
        font-size: 12px;
        margin-right: 0.3rem;
      }
      .muted {
        color: var(--muted);
      }
      .controls {
        display: grid;
        gap: 0.5rem;
      }
      input, button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--line);
        background: #0c1730;
        color: var(--text);
        padding: 0.55rem 0.65rem;
        font: inherit;
      }
      button {
        cursor: pointer;
        background: #19284a;
        transition: transform 120ms ease;
      }
      button:hover {
        transform: translateY(-1px);
      }
      .slider-row {
        display: grid;
        gap: 0.3rem;
      }
      .timeline-label {
        color: var(--muted);
        font-size: 12px;
      }
      .cards {
        display: grid;
        gap: 0.6rem;
        max-height: 500px;
        overflow: auto;
      }
      .card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.6rem;
        background: rgba(10, 18, 34, 0.9);
      }
      pre {
        margin: 0.4rem 0 0 0;
        white-space: pre-wrap;
        background: #0a1222;
        padding: 0.55rem;
        border-radius: 8px;
        border: 1px solid var(--line);
        color: #d3e4ff;
      }
      .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
      }
      .small-note {
        color: var(--muted);
        font-size: 12px;
      }
      canvas {
        width: 100%;
        min-height: 280px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #090f1d;
      }
      @keyframes pop {
        from { transform: translateY(4px); opacity: 0.6; }
        to { transform: translateY(0); opacity: 1; }
      }
      @media (max-width: 900px) {
        .grid-3 {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <section>
        <h1>Synapse AI Memory Inspector</h1>
        <div id="stats" class="mono"></div>
      </section>

      <section>
        <h2>Timeline</h2>
        <div class="slider-row">
          <label for="timeline-slider" class="timeline-label">Show memories up to date</label>
          <input id="timeline-slider" type="range" min="0" max="1" value="1" step="1" />
          <div id="timeline-date" class="small-note">loading</div>
        </div>
        <div id="timeline-cards" class="cards mono"></div>
      </section>

      <section>
        <h2>Contradictions</h2>
        <div id="contradictions" class="cards mono"></div>
      </section>

      <section>
        <h2>Belief History</h2>
        <div class="grid-2" style="margin-bottom:0.6rem;">
          <div class="controls">
            <label for="belief-query" class="small-note">search tracked beliefs</label>
            <input id="belief-query" type="text" placeholder="e.g. likes, project, alice" />
          </div>
        </div>
        <div id="beliefs" class="cards mono"></div>
      </section>

      <section>
        <h2>Concept Graph</h2>
        <div class="two-col">
          <div>
            <canvas id="concept-canvas" height="360"></canvas>
            <div id="graph-help" class="small-note">Tip: click a node to show associated memories.</div>
          </div>
          <div>
            <div id="graph-meta" class="small-note">Tap a concept to see related memories.</div>
            <div id="graph-memories" class="cards mono" style="max-height: 360px;"></div>
          </div>
        </div>
      </section>

      <section>
        <h2>Context Preview</h2>
        <div class="grid-2">
          <div class="controls">
            <input id="compile-query" type="text" placeholder="compile_context query" />
            <div class="slider-row">
              <label for="compile-budget" class="small-note">budget</label>
              <input id="compile-budget" type="range" min="200" max="8000" step="100" value="1200" />
              <div id="compile-budget-label" class="small-note"></div>
            </div>
            <button id="compile-trigger">Run compile_context()</button>
          </div>
          <div class="controls">
            <label for="compile-output" class="small-note">Formatted ContextPack</label>
            <pre id="compile-output" class="mono"></pre>
          </div>
        </div>
      </section>

      <section>
        <h2>Search</h2>
        <div class="grid-2">
          <div class="controls">
            <input id="recall-query" type="text" placeholder="search memory recall" />
            <button id="recall-trigger">Search</button>
          </div>
          <div id="recall-results" class="cards mono"></div>
        </div>
      </section>
    </div>

    <script>
      const state = {
        memories: [],
        memoriesById: {},
        concepts: [],
        conceptEdges: [],
        conceptToMemories: {},
        contradictions: [],
        beliefs: [],
        selectedNode: null,
      };

      const $ = (selector) => document.querySelector(selector);
      const fmtDate = (epoch) => new Date(epoch * 1000).toLocaleString();

      async function getJSON(url) {
        const response = await fetch(url);
        return await response.json();
      }

      function setStats(stats) {
        const el = $("#stats");
        el.innerHTML =
          `<span class="chip">memories: ${stats.memory_count}</span>` +
          `<span class="chip">concepts: ${stats.concept_count}</span>` +
          `<span class="chip">contradictions: ${stats.contradictions}</span>` +
          `<span class="chip">beliefs: ${stats.belief_count}</span>`;
      }

      function renderTimeline() {
        const slider = $("#timeline-slider");
        const cardArea = $("#timeline-cards");
        const label = $("#timeline-date");
        const max = Number(slider.value) / 1000;
        const visible = state.memories.filter((item) => item.created_at <= max);
        label.textContent = `up to ${fmtDate(max)}`;
        visible.sort((a, b) => b.created_at - a.created_at);
        if (!visible.length) {
          cardArea.innerHTML = "<div class=\"muted\">No memories in range.</div>";
          return;
        }
        cardArea.innerHTML = visible.map((memory) => {
          const score = memory.effective_strength != null ? memory.effective_strength.toFixed(3) : "n/a";
          const concepts = (memory.concepts || []).join(", ");
          return (
            `<div class="card">` +
            `<div><strong>#${memory.id}</strong> · ${fmtDate(memory.created_at)} · strength ${score}</div>` +
            `<div class="muted">${memory.memory_type}</div>` +
            `<div>${memory.content}</div>` +
            `<div class="muted">concepts: ${concepts || "none"}</div>` +
            `</div>`
          );
        }).join("");
      }

      function renderContradictions() {
        const area = $("#contradictions");
        if (!state.contradictions.length) {
          area.innerHTML = "<div class=\"muted\">No unresolved contradictions.</div>";
          return;
        }
        area.innerHTML = state.contradictions.map((entry) => {
          return (
            `<div class="card">` +
            `<div><strong>${entry.kind}</strong> · confidence ${(entry.confidence || 0).toFixed(2)}</div>` +
            `<div class=\"muted\">#${entry.left_id}: ${entry.left_text}</div>` +
            `<div class=\"muted\">#${entry.right_id}: ${entry.right_text}</div>` +
            `</div>`
          );
        }).join("");
      }

      function renderBeliefs(items) {
        const area = $("#beliefs");
        if (!items.length) {
          area.innerHTML = "<div class=\"muted\">No matching beliefs.</div>";
          return;
        }
        area.innerHTML = items.map((item) => {
          const current = item.current || {};
          const versions = (item.versions || []);
          const history = versions.map((v) => `${v.fact_key} · ${v.value} · confidence ${v.confidence.toFixed(2)} · ${fmtDate(v.valid_from)}`).join("<br>");
          return (
            `<div class="card">` +
            `<div><strong>${item.fact_key}</strong></div>` +
            `<div class=\"muted\">current: ${current.value || "n/a"} (memory ${current.memory_id || "?"})</div>` +
            `<div class=\"small-note\">${history}</div>` +
            `</div>`
          );
        }).join("");
      }

      async function loadBeliefs() {
        const q = $("#belief-query").value.trim();
        const endpoint = q ? `/api/beliefs?q=${encodeURIComponent(q)}` : "/api/beliefs";
        const payload = await getJSON(endpoint);
        renderBeliefs(payload.beliefs || []);
      }

      function drawGraph() {
        const canvas = $("#concept-canvas");
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const styleHeight = canvas.clientHeight;
        const styleWidth = canvas.clientWidth;
        canvas.width = Math.max(1, Math.floor(styleWidth * dpr));
        canvas.height = Math.max(1, Math.floor(styleHeight * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const width = styleWidth;
        const height = styleHeight;

        const gravity = 0.08;
        const repulsion = 3800;
        const spring = 0.0022;
        const damping = 0.86;

        for (const edge of state.conceptEdges) {
          const source = state.concepts[edge.sourceIndex];
          const target = state.concepts[edge.targetIndex];
          if (!source || !target) continue;
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dist = Math.hypot(dx, dy) || 20;
          const force = spring * (dist - 80);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          source.vx += fx * 0.02;
          source.vy += fy * 0.02;
          target.vx -= fx * 0.02;
          target.vy -= fy * 0.02;
        }

        for (let i = 0; i < state.concepts.length; i += 1) {
          const a = state.concepts[i];
          for (let j = i + 1; j < state.concepts.length; j += 1) {
            const b = state.concepts[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist2 = dx * dx + dy * dy + 35;
            const force = -repulsion / dist2;
            const inv = Math.sqrt(dist2);
            a.vx += (dx / inv) * force;
            a.vy += (dy / inv) * force;
            b.vx -= (dx / inv) * force;
            b.vy -= (dy / inv) * force;
          }
        }

        for (const node of state.concepts) {
          node.vx += (width / 2 - node.x) * 0.0009;
          node.vy += (height / 2 - node.y) * 0.0009;
          node.vx += (Math.random() - 0.5) * gravity * 0.001;
          node.vy += (Math.random() - 0.5) * gravity * 0.001;
          node.x += node.vx;
          node.y += node.vy;
          node.vx *= damping;
          node.vy *= damping;
          node.x = Math.min(width - 16, Math.max(16, node.x));
          node.y = Math.min(height - 16, Math.max(16, node.y));
        }

        ctx.clearRect(0, 0, width, height);
        for (const edge of state.conceptEdges) {
          const source = state.concepts[edge.sourceIndex];
          const target = state.concepts[edge.targetIndex];
          if (!source || !target) continue;
          ctx.strokeStyle = "rgba(122, 162, 255, 0.4)";
          ctx.lineWidth = Math.max(0.8, Math.min(5, edge.weight * 2.2));
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }

        for (const node of state.concepts) {
          const r = 6 + (node.size || 2) * 1.8;
          ctx.fillStyle = state.selectedNode === node ? "#ffbf76" : "#7aa2ff";
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#cce0ff";
          ctx.font = "11px monospace";
          ctx.fillText(node.id, node.x + r + 4, node.y + 3);
        }
      }

      function startGraph() {
        if (!state.concepts.length) {
          $("#graph-meta").textContent = "No concepts to visualize yet.";
          return;
        }
        $("#graph-meta").textContent = `${state.concepts.length} nodes, ${state.conceptEdges.length} edges`;
        for (const node of state.concepts) {
          node.x = Math.random() * 360 + 40;
          node.y = Math.random() * 220 + 40;
          node.vx = 0;
          node.vy = 0;
        }
        function tick() {
          drawGraph();
          requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      }

      function selectNodeByHit(x, y) {
        for (const node of state.concepts) {
          const r = 6 + (node.size || 2) * 1.8;
          if (Math.hypot(x - node.x, y - node.y) <= r + 3) {
            state.selectedNode = node;
            const ids = state.conceptToMemories[node.id] || [];
            const memories = ids.map((id) => state.memoriesById[id]).filter(Boolean);
            $("#graph-memories").innerHTML = memories.length
              ? memories.map((memory) => `<div class="card">#${memory.id} · ${memory.content}</div>`).join("")
              : "<div class='muted'>No memories for this concept.</div>";
            $("#graph-meta").textContent = `node=${node.id}, memories=${ids.length}, activation=${node.activation.toFixed(3)}`;
            return;
          }
        }
      }

      function initGraphEvents() {
        const canvas = $("#concept-canvas");
        canvas.addEventListener("click", (event) => {
          const rect = canvas.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          selectNodeByHit(x, y);
        });
      }

      async function runRecall() {
        const q = $("#recall-query").value.trim();
        const payload = await getJSON(`/api/recall?q=${encodeURIComponent(q)}&limit=8`);
        const area = $("#recall-results");
        const results = payload.results || [];
        if (!results.length) {
          area.innerHTML = "<div class=\"muted\">No results.</div>";
          return;
        }
        area.innerHTML = results.map((item) => {
          const breakdown = item.score_breakdown || {};
          const lines = Object.entries(breakdown)
            .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(3) : v}`)
            .join("<br>");
          return (
            `<div class=\"card\">` +
            `<div><strong>#${item.id}</strong> (${item.memory_type}) score=${(item.score || 0).toFixed(4)}</div>` +
            `<div>${item.content}</div>` +
            `<div class=\"muted\">score_breakdown:<br>${lines || "n/a"}</div>` +
            `</div>`
          );
        }).join("");
      }

      async function runCompile() {
        const q = $("#compile-query").value.trim();
        const budget = Number($("#compile-budget").value);
        if (!q) {
          $("#compile-output").textContent = "enter a compile query";
          return;
        }
        const payload = await getJSON(`/api/compile?q=${encodeURIComponent(q)}&budget=${budget}`);
        if (payload.context_text) {
          $("#compile-output").textContent = payload.context_text;
        } else {
          $("#compile-output").textContent = JSON.stringify(payload.pack, null, 2);
        }
      }

      function hydrateTimelineSlider(memories) {
        const slider = $("#timeline-slider");
        if (!memories.length) {
          slider.disabled = true;
          return;
        }
        const times = memories.map((item) => item.created_at).filter((value) => Number.isFinite(value));
        const lo = Math.floor(Math.min(...times) * 1000);
        const hi = Math.ceil(Math.max(...times) * 1000);
        slider.min = String(lo);
        slider.max = String(hi);
        slider.value = String(hi);
        slider.oninput = () => renderTimeline();
        renderTimeline();
      }

      async function init() {
        const stats = await getJSON("/api/stats");
        setStats(stats);

        const memories = await getJSON("/api/memories");
        state.memories = memories;
        state.memoriesById = {};
        for (const item of memories) {
          state.memoriesById[item.id] = item;
        }
        hydrateTimelineSlider(memories);

        const concepts = await getJSON("/api/concepts");
        state.concepts = (concepts.nodes || []).map((item, index) => ({
          ...item,
          index,
          vx: 0,
          vy: 0,
          x: 0,
          y: 0,
        }));
        state.conceptEdges = (concepts.edges || []).map((item) => ({
          ...item,
          sourceIndex: state.concepts.findIndex((node) => node.id === item.source),
          targetIndex: state.concepts.findIndex((node) => node.id === item.target),
        })).filter((edge) => edge.sourceIndex >= 0 && edge.targetIndex >= 0);
        state.conceptToMemories = concepts.concept_to_memories || {};
        renderTimeline();
        renderContradictions();
        startGraph();
        initGraphEvents();

        $("#belief-query").addEventListener("input", loadBeliefs);
        await loadBeliefs();

        const contradictions = await getJSON("/api/contradictions");
        state.contradictions = contradictions.contradictions || [];
        renderContradictions();

        $("#compile-budget").oninput = () => {
          $("#compile-budget-label").textContent = `${$("#compile-budget").value} tokens`;
        };
        $("#compile-budget").dispatchEvent(new Event("input"));
        $("#compile-trigger").addEventListener("click", runCompile);

        $("#recall-trigger").addEventListener("click", runRecall);
      }

      init().catch((error) => {
        const msg = String(error && error.message ? error.message : error);
        $("#stats").textContent = `Failed to initialize dashboard: ${msg}`;
      });
    </script>
  </body>
</html>
"""


class SynapseInspector:
    """Serve a tiny local dashboard for inspecting Synapse data."""

    def __init__(self, synapse: Synapse, port: int = 9471):
        self.synapse = synapse
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        """Start a local HTTP server serving a single-page app."""
        handler_class = self._build_handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler_class)
        self.port = self._server.server_port

        target = f"http://127.0.0.1:{self.port}/"
        print(f"Starting Synapse AI Memory Inspector on {target}")
        webbrowser.open(target)

        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()

    def _coerce_int_param(self, raw: List[str] | None, default: int) -> int:
        if not raw:
            return default
        try:
            return int(raw[0])
        except (TypeError, ValueError):
            return default

    def _collect_memories(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get("consolidated", False):
                continue
            memory = self.synapse._memory_data_to_object(memory_data)
            concepts = [name for name, _ in extract_concepts(memory.content)]
            items.append(
                self._serialize_memory(memory, include_score=False, concepts=sorted(set(concepts)))
            )
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def _collect_concepts(self) -> Dict[str, Any]:
        concept_to_memory_ids: Dict[str, List[int]] = defaultdict(list)
        memory_concepts: Dict[int, Tuple[str, ...]] = {}

        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get("consolidated", False):
                continue
            concept_list = [name for name, _ in extract_concepts(memory_data.get("content", ""))]
            concepts = tuple(dict.fromkeys(concept_list))
            if not concepts:
                continue
            concept_set = tuple(concepts)
            memory_concepts[memory_id] = concept_set
            for concept in concept_set:
                concept_to_memory_ids[concept].append(memory_id)

        nodes: List[Dict[str, Any]] = []
        for concept, memory_ids in sorted(concept_to_memory_ids.items(), key=lambda item: (item[0])):
            node = self.synapse.concept_graph.concepts.get(concept)
            strength = self.synapse.concept_graph.concept_activation_strength(concept)
            nodes.append({
                "id": concept,
                "activation": float(strength),
                "size": float(min(7.0, 2.0 + (strength * 5))),
                "memory_count": len(memory_ids),
                "memory_ids": memory_ids,
                "category": getattr(node, "category", "general"),
            })

        co_occurrence: Dict[Tuple[str, str], float] = defaultdict(float)
        for concepts in memory_concepts.values():
            unique = sorted(set(concepts))
            if len(unique) < 2:
                continue
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    co_occurrence[(unique[i], unique[j])] += 1.0

        edges = []
        for (source, target), weight in sorted(co_occurrence.items()):
            edges.append({"source": source, "target": target, "weight": float(weight)})

        return {
            "nodes": nodes,
            "edges": edges,
            "concept_to_memories": {key: value for key, value in concept_to_memory_ids.items()},
        }

    def _collect_contradictions(self) -> Dict[str, Any]:
        entries = []
        for contradiction in self.synapse.contradictions():
            left = self.synapse.store.memories.get(contradiction.memory_id_a, {})
            right = self.synapse.store.memories.get(contradiction.memory_id_b, {})
            entries.append({
                "left_id": contradiction.memory_id_a,
                "right_id": contradiction.memory_id_b,
                "kind": contradiction.kind,
                "confidence": float(contradiction.confidence),
                "description": contradiction.description,
                "left_text": left.get("content", ""),
                "right_text": right.get("content", ""),
            })
        return {"contradictions": entries}

    def _collect_beliefs(self, query: str = "") -> Dict[str, Any]:
        all_beliefs = self.synapse.beliefs()
        query = (query or "").strip().lower()
        payload: List[Dict[str, Any]] = []

        for fact_key, current in sorted(all_beliefs.items()):
            if query and query not in fact_key.lower() and query not in current.value.lower():
                continue
            versions = [
                self._serialize_belief(v)
                for v in self.synapse.belief_history(fact_key)
            ]
            payload.append({
                "fact_key": fact_key,
                "current": self._serialize_belief(current),
                "versions": versions,
            })
        return {"beliefs": payload}

    def _collect_stats(self) -> Dict[str, Any]:
        top_hot = self.synapse.hot_concepts(k=10)
        return {
            "memory_count": self.synapse.count(),
            "concept_count": len(self.synapse.concept_graph.concepts),
            "edge_count": len(self.synapse.store.edges),
            "contradictions": len(self.synapse.contradictions()),
            "belief_count": len(self.synapse.beliefs()),
            "top_hot_concepts": [{"name": name, "activation": score} for name, score in top_hot],
        }

    def _serialize_belief(self, belief: Any) -> Dict[str, Any]:
        return {
            "fact_key": belief.fact_key,
            "memory_id": belief.memory_id,
            "value": belief.value,
            "valid_from": belief.valid_from,
            "valid_to": belief.valid_to,
            "reason": belief.reason,
            "confidence": float(belief.confidence),
        }

    def _serialize_memory(self, memory, include_score: bool = False, concepts: Iterable[str] | None = None) -> Dict[str, Any]:
        score_breakdown = asdict(memory.score_breakdown) if getattr(memory, "score_breakdown", None) else None
        payload: Dict[str, Any] = {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type,
            "memory_level": memory.memory_level,
            "strength": float(memory.strength),
            "effective_strength": float(memory.effective_strength),
            "created_at": float(memory.created_at),
            "last_accessed": float(memory.last_accessed),
            "metadata": memory.metadata,
            "consolidated": bool(memory.consolidated),
            "concepts": sorted(set(concepts or [])),
            "disputes": memory.disputes,
        }
        if score_breakdown is not None:
            payload["score_breakdown"] = score_breakdown
            if include_score:
                score = score_breakdown.get("bm25_score", 0.0)
                payload["score"] = float(score)
        return payload

    def _collect_recall(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        memories = self.synapse.recall(context=query, limit=limit, explain=True)
        results = [
            self._serialize_memory(memory, include_score=True, concepts=[name for name, _ in extract_concepts(memory.content)])
            for memory in memories
        ]
        return {"query": query, "count": len(results), "results": results}

    def _collect_compile(self, query: str = "", budget: int = 2000) -> Dict[str, Any]:
        pack: ContextPack = self.synapse.compile_context(query=query, budget=int(budget))
        return {
            "query": query,
            "budget": budget,
            "context_text": pack.to_system_prompt(),
            "pack": pack.to_dict(),
        }

    def _build_handler(self):
        inspector = self

        class _Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, text: str, status: int = 200) -> None:
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                if path in {"/", "/index.html"}:
                    return self._send_text(_DASHBOARD_HTML)

                if not path.startswith("/api/"):
                    return self._send_json({"error": "Not found"}, status=404)

                params = parse_qs(parsed.query)
                try:
                    if path == "/api/stats":
                        payload = inspector._collect_stats()
                    elif path == "/api/memories":
                        payload = inspector._collect_memories()
                    elif path == "/api/concepts":
                        payload = inspector._collect_concepts()
                    elif path == "/api/contradictions":
                        payload = inspector._collect_contradictions()
                    elif path == "/api/beliefs":
                        payload = inspector._collect_beliefs(params.get("q", [None])[0] or "")
                    elif path == "/api/recall":
                        payload = inspector._collect_recall(
                            query=(params.get("q", [""])[0] or ""),
                            limit=inspector._coerce_int_param(params.get("limit"), 10),
                        )
                    elif path == "/api/compile":
                        payload = inspector._collect_compile(
                            query=(params.get("q", [""])[0] or ""),
                            budget=inspector._coerce_int_param(params.get("budget"), 2000),
                        )
                    else:
                        return self._send_json({"error": "Unknown API endpoint"}, status=404)
                except Exception as exc:
                    return self._send_json({"error": str(exc)}, status=500)

                return self._send_json(payload)

            def log_message(self, format: str, *args):  # noqa: A003
                return

        return _Handler
