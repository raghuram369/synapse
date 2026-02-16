"""Local read-only inspector web UI for Synapse memory stores."""

from __future__ import annotations

import json
import os
import time
import webbrowser
from collections import defaultdict
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

from context_pack import ContextPack
from entity_graph import extract_concepts
from synapse import Synapse


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Synapse AI Memory Inspector</title>
<style>
:root{--bg:#04070d;--panel:#0b1220;--panel-soft:#121a2b;--line:#24304b;--text:#d2def9;--muted:#8ea0c7;--accent:#7aa2ff;--ok:#63f3b7;--warn:#ffce64;--err:#ff6e6e;--episodic:#7aa2ff;--semantic:#63f3b7;--preference:#ffce64}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Fira Code","JetBrains Mono",Menlo,Consolas,monospace;background:radial-gradient(circle at 10% 0%,#102040 0%,#04070d 45%,#020409 100%);color:var(--text);min-height:100vh}
.header{background:var(--panel);border-bottom:1px solid var(--line);padding:0.8rem 1rem;position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.header h1{font-size:1.1rem;letter-spacing:0.06em;white-space:nowrap}
.search-bar{flex:1;min-width:200px;max-width:400px}
.search-bar input{width:100%;border-radius:8px;border:1px solid var(--line);background:#0c1730;color:var(--text);padding:0.45rem 0.65rem;font:inherit;font-size:13px}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:52px;z-index:99;overflow-x:auto}
.tab{padding:0.6rem 1.2rem;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;white-space:nowrap;font-size:13px;transition:all 150ms}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.page{max-width:1400px;margin:0 auto;padding:1rem}
section{border:1px solid var(--line);background:linear-gradient(180deg,var(--panel) 0%,var(--panel-soft) 100%);border-radius:10px;padding:1rem;animation:pop 260ms ease-out;margin-bottom:1rem}
.tab-content{display:none}
.tab-content.active{display:block}
h2{margin:0 0 0.75rem;letter-spacing:0.06em;font-size:1rem}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.8rem}
.mono{font-size:13px}
.chip{display:inline-block;padding:0.2rem 0.5rem;background:#1c2740;border:1px solid var(--line);border-radius:999px;font-size:12px;margin:0.15rem}
.muted{color:var(--muted)}
.small-note{color:var(--muted);font-size:12px}
input,button,select,textarea{border-radius:8px;border:1px solid var(--line);background:#0c1730;color:var(--text);padding:0.5rem 0.65rem;font:inherit;font-size:13px}
button{cursor:pointer;background:#19284a;transition:transform 120ms ease}
button:hover{transform:translateY(-1px)}
.cards{display:grid;gap:0.6rem;max-height:600px;overflow:auto}
.card{border:1px solid var(--line);border-radius:8px;padding:0.6rem;background:rgba(10,18,34,0.9);animation:pop 200ms ease-out}
pre{margin:0.4rem 0 0;white-space:pre-wrap;background:#0a1222;padding:0.55rem;border-radius:8px;border:1px solid var(--line);color:#d3e4ff;font-size:12px;max-height:400px;overflow:auto}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.8rem}
.stat-box{text-align:center;padding:1rem;border:1px solid var(--line);border-radius:10px;background:rgba(10,18,34,0.7)}
.stat-box .value{font-size:1.6rem;color:var(--accent);font-weight:bold}
.stat-box .label{font-size:11px;color:var(--muted);margin-top:0.3rem}
.type-episodic{border-left:3px solid var(--episodic)}
.type-semantic{border-left:3px solid var(--semantic)}
.type-preference{border-left:3px solid var(--preference)}
.type-promotion{border-left:3px solid var(--warn);background:rgba(255,206,100,0.05)}
.resolve-btn{background:#2a1a3a;border-color:#5a3a7a;font-size:11px;padding:0.3rem 0.6rem;margin-top:0.4rem}
.resolve-btn:hover{background:#3a2a5a}
.topic-group{margin-bottom:1rem}
.topic-header{color:var(--accent);font-size:14px;margin-bottom:0.5rem;cursor:pointer}
.topic-filter{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem}
.topic-chip{padding:0.3rem 0.7rem;border-radius:999px;border:1px solid var(--line);background:#1c2740;cursor:pointer;font-size:12px;transition:all 150ms}
.topic-chip.active{background:var(--accent);color:#000;border-color:var(--accent)}
.digest-card{border:1px solid var(--line);border-radius:8px;padding:0.8rem;background:rgba(10,18,34,0.9);margin-bottom:0.6rem}
.digest-date{color:var(--accent);font-weight:bold}
.controls{display:grid;gap:0.5rem}
.slider-row{display:grid;gap:0.3rem}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:0.8rem}
canvas{width:100%;min-height:280px;border:1px solid var(--line);border-radius:8px;background:#090f1d}
@keyframes pop{from{transform:translateY(4px);opacity:0.6}to{transform:translateY(0);opacity:1}}
@media(max-width:768px){.grid-2,.two-col{grid-template-columns:1fr}.header{flex-direction:column}.search-bar{max-width:100%}.stat-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<div class="header">
  <h1>🧠 Synapse Inspector</h1>
  <div class="search-bar"><input id="global-search" type="text" placeholder="Search across all panels…"/></div>
</div>

<div class="tabs" id="tab-bar">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="timeline">Timeline</div>
  <div class="tab" data-tab="beliefs">Beliefs</div>
  <div class="tab" data-tab="contradictions">Contradictions</div>
  <div class="tab" data-tab="context">Context</div>
  <div class="tab" data-tab="sleep">Sleep</div>
</div>

<div class="page">

<!-- OVERVIEW TAB -->
<div class="tab-content active" id="tab-overview">
  <section>
    <h2>Health Dashboard</h2>
    <div class="stat-grid" id="health-stats"></div>
    <div class="small-note" style="margin-top:0.6rem" id="health-refresh-note">Auto-refreshes every 30s</div>
  </section>
  <section>
    <h2>Concept Graph</h2>
    <div class="two-col">
      <div>
        <canvas id="concept-canvas" height="360"></canvas>
        <div class="small-note">Click a node to see related memories</div>
      </div>
      <div>
        <div id="graph-meta" class="small-note">Tap a concept to see related memories.</div>
        <div id="graph-memories" class="cards mono" style="max-height:360px"></div>
      </div>
    </div>
  </section>
  <section>
    <h2>Search</h2>
    <div class="grid-2">
      <div class="controls">
        <input id="recall-query" type="text" placeholder="search memory recall"/>
        <button id="recall-trigger">Search</button>
      </div>
      <div id="recall-results" class="cards mono"></div>
    </div>
  </section>
</div>

<!-- TIMELINE TAB -->
<div class="tab-content" id="tab-timeline">
  <section>
    <h2>What Changed</h2>
    <div class="grid-2" style="margin-bottom:1rem">
      <div class="controls">
        <label class="small-note">From</label>
        <input id="timeline-from" type="date"/>
      </div>
      <div class="controls">
        <label class="small-note">To</label>
        <input id="timeline-to" type="date"/>
      </div>
    </div>
    <button id="timeline-fetch" style="margin-bottom:1rem;width:auto;padding:0.4rem 1.2rem">Load Timeline</button>
    <div class="small-note" style="margin-bottom:0.5rem">
      <span class="chip" style="border-left:3px solid var(--episodic)">episodic</span>
      <span class="chip" style="border-left:3px solid var(--semantic)">semantic</span>
      <span class="chip" style="border-left:3px solid var(--preference)">preference</span>
      <span class="chip" style="border-left:3px solid var(--warn)">promotion</span>
    </div>
    <div id="timeline-cards" class="cards mono"></div>
  </section>
</div>

<!-- BELIEFS TAB -->
<div class="tab-content" id="tab-beliefs">
  <section>
    <h2>What It Knows — Top Beliefs</h2>
    <div style="margin-bottom:1rem">
      <input id="belief-query" type="text" placeholder="Filter by topic…" style="width:100%;max-width:400px"/>
    </div>
    <div id="belief-topics" class="topic-filter"></div>
    <div id="beliefs" class="cards mono"></div>
  </section>
</div>

<!-- CONTRADICTIONS TAB -->
<div class="tab-content" id="tab-contradictions">
  <section>
    <h2>Contradictions (Disputes)</h2>
    <div id="contradictions" class="cards mono"></div>
  </section>
</div>

<!-- CONTEXT TAB -->
<div class="tab-content" id="tab-context">
  <section>
    <h2>Context Preview</h2>
    <div class="grid-2">
      <div class="controls">
        <input id="compile-query" type="text" placeholder="Enter query for compile_context"/>
        <div class="slider-row">
          <label class="small-note">Budget: <span id="compile-budget-label">2000</span> tokens</label>
          <input id="compile-budget" type="range" min="500" max="4000" step="100" value="2000"/>
        </div>
        <div>
          <label class="small-note">Policy</label>
          <select id="compile-policy">
            <option value="balanced">balanced</option>
            <option value="precise">precise</option>
            <option value="broad">broad</option>
            <option value="temporal">temporal</option>
          </select>
        </div>
        <button id="compile-trigger">Preview Context</button>
      </div>
      <div>
        <label class="small-note">Result</label>
        <pre id="compile-output" class="mono">Enter a query and click Preview</pre>
      </div>
    </div>
  </section>
</div>

<!-- SLEEP TAB -->
<div class="tab-content" id="tab-sleep">
  <section>
    <h2>Sleep Digest History</h2>
    <div id="sleep-digests"></div>
  </section>
</div>

</div><!-- .page -->

<script>
const $=s=>document.querySelector(s);
const $$=s=>document.querySelectorAll(s);
const fmtDate=e=>new Date(e*1000).toLocaleString();
const fmtDateShort=e=>new Date(e*1000).toLocaleDateString();
const state={memories:[],memoriesById:{},concepts:[],conceptEdges:[],conceptToMemories:{},contradictions:[],beliefs:[],selectedNode:null};

async function getJSON(u){const r=await fetch(u);return r.json()}
async function postJSON(u,d){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});return r.json()}

// Tab navigation
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
  });
});

// Global search
$('#global-search').addEventListener('input',e=>{
  const q=e.target.value.toLowerCase();
  document.querySelectorAll('.card,.digest-card').forEach(c=>{
    c.style.display=(!q||c.textContent.toLowerCase().includes(q))?'':'none';
  });
});

// Health Dashboard
async function loadHealth(){
  const h=await getJSON('/api/health');
  const el=$('#health-stats');
  el.innerHTML=[
    {v:h.memory_count,l:'Memories'},
    {v:h.concept_count,l:'Concepts'},
    {v:h.triple_count,l:'Triples'},
    {v:h.contradiction_count,l:'Contradictions'},
    {v:h.belief_count,l:'Beliefs'},
    {v:h.storage_display||'—',l:'Storage'},
    {v:h.last_sleep_display||'never',l:'Last Sleep'},
  ].map(x=>`<div class="stat-box"><div class="value">${x.v}</div><div class="label">${x.l}</div></div>`).join('');
}
setInterval(loadHealth,30000);

// Timeline
$('#timeline-fetch').addEventListener('click',async()=>{
  const from=$('#timeline-from').value;
  const to=$('#timeline-to').value;
  let url='/api/timeline';
  const params=[];
  if(from)params.push('from='+from);
  if(to)params.push('to='+to);
  if(params.length)url+='?'+params.join('&');
  const data=await getJSON(url);
  const area=$('#timeline-cards');
  const items=data.memories||[];
  if(!items.length){area.innerHTML='<div class="muted">No memories in range.</div>';return;}
  area.innerHTML=items.map(m=>{
    const cls=m.is_promotion?'type-promotion':'type-'+(m.memory_type||'episodic');
    const label=m.is_promotion?'⬆ PROMOTED: '+m.memory_type:m.memory_type;
    return `<div class="card ${cls}"><div><strong>#${m.id}</strong> · ${fmtDate(m.created_at)} · strength ${(m.effective_strength||0).toFixed(3)}</div><div class="muted">${label}</div><div>${m.content}</div></div>`;
  }).join('');
});

// Set default date range
(function(){
  const now=new Date();
  const week=new Date(now-7*86400000);
  $('#timeline-to').value=now.toISOString().slice(0,10);
  $('#timeline-from').value=week.toISOString().slice(0,10);
})();

// Beliefs
let allBeliefs=[];
let activeTopic=null;

async function loadBeliefs(){
  const q=$('#belief-query').value.trim();
  const endpoint=q?'/api/beliefs?q='+encodeURIComponent(q):'/api/beliefs';
  const data=await getJSON(endpoint);
  allBeliefs=data.beliefs||[];
  renderBeliefTopics();
  renderBeliefs();
}

function renderBeliefTopics(){
  const topics=new Set();
  allBeliefs.forEach(b=>{const parts=b.fact_key.split('.');if(parts.length>1)topics.add(parts[0]);});
  const el=$('#belief-topics');
  if(!topics.size){el.innerHTML='';return;}
  el.innerHTML='<span class="topic-chip'+(activeTopic===null?' active':'')+'" data-topic="">All</span>'+
    [...topics].sort().map(t=>'<span class="topic-chip'+(activeTopic===t?' active':'')+'" data-topic="'+t+'">'+t+'</span>').join('');
  el.querySelectorAll('.topic-chip').forEach(c=>c.addEventListener('click',()=>{
    activeTopic=c.dataset.topic||null;
    renderBeliefTopics();
    renderBeliefs();
  }));
}

function renderBeliefs(){
  const area=$('#beliefs');
  let items=allBeliefs;
  if(activeTopic)items=items.filter(b=>b.fact_key.startsWith(activeTopic+'.'));
  if(!items.length){area.innerHTML='<div class="muted">No matching beliefs.</div>';return;}
  area.innerHTML=items.map(b=>{
    const c=b.current||{};
    const sources=(b.versions||[]).length;
    return `<div class="card"><div><strong>${b.fact_key}</strong></div><div>${c.value||'n/a'}</div><div class="small-note">confidence: ${(c.confidence||0).toFixed(2)} · sources: ${sources} · updated: ${c.valid_from?fmtDate(c.valid_from):'—'}</div></div>`;
  }).join('');
}

$('#belief-query').addEventListener('input',loadBeliefs);

// Contradictions
async function loadContradictions(){
  const data=await getJSON('/api/contradictions');
  state.contradictions=data.contradictions||[];
  renderContradictions();
}

function renderContradictions(){
  const area=$('#contradictions');
  if(!state.contradictions.length){area.innerHTML='<div class="muted">No unresolved contradictions. 🎉</div>';return;}
  area.innerHTML=state.contradictions.map((e,i)=>{
    return `<div class="card"><div><strong>${e.kind}</strong> · confidence ${(e.confidence||0).toFixed(2)}</div><div class="muted">#${e.left_id}: ${e.left_text}</div><div class="muted">#${e.right_id}: ${e.right_text}</div><div><button class="resolve-btn" data-idx="${i}" data-winner="${e.left_id}">Keep #${e.left_id}</button> <button class="resolve-btn" data-idx="${i}" data-winner="${e.right_id}">Keep #${e.right_id}</button></div></div>`;
  }).join('');
  area.querySelectorAll('.resolve-btn').forEach(btn=>btn.addEventListener('click',async()=>{
    await postJSON('/api/resolve-contradiction',{contradiction_id:parseInt(btn.dataset.idx),winner_memory_id:parseInt(btn.dataset.winner)});
    loadContradictions();
  }));
}

// Context Preview
$('#compile-budget').oninput=()=>{$('#compile-budget-label').textContent=$('#compile-budget').value};
$('#compile-trigger').addEventListener('click',async()=>{
  const q=$('#compile-query').value.trim();
  if(!q){$('#compile-output').textContent='Enter a query first';return;}
  const budget=Number($('#compile-budget').value);
  const policy=$('#compile-policy').value;
  const data=await postJSON('/api/preview-context',{query:q,budget:budget,policy:policy});
  if(data.context_text){$('#compile-output').textContent=data.context_text;}
  else{$('#compile-output').textContent=JSON.stringify(data.pack,null,2);}
});

// Sleep Digests
async function loadDigests(){
  const data=await getJSON('/api/digests');
  const area=$('#sleep-digests');
  const digests=data.digests||[];
  if(!digests.length){area.innerHTML='<div class="muted">No sleep digests yet. Run: <code>synapse sleep --digest</code></div>';return;}
  area.innerHTML=digests.map(d=>{
    const topics=(d.hot_topics||[]).map(t=>'<span class="chip">'+t+'</span>').join('');
    return `<div class="digest-card"><div class="digest-date">${d.date||'unknown'}</div><div class="small-note">consolidated: ${d.consolidated||0} · promoted: ${d.promoted||0} · pruned: ${d.pruned||0}</div><div>${topics||'<span class="muted">no hot topics</span>'}</div></div>`;
  }).join('');
}

// Recall search
$('#recall-trigger').addEventListener('click',async()=>{
  const q=$('#recall-query').value.trim();
  const data=await getJSON('/api/recall?q='+encodeURIComponent(q)+'&limit=8');
  const area=$('#recall-results');
  const results=data.results||[];
  if(!results.length){area.innerHTML='<div class="muted">No results.</div>';return;}
  area.innerHTML=results.map(item=>{
    const bd=item.score_breakdown||{};
    const lines=Object.entries(bd).map(([k,v])=>k+': '+(typeof v==='number'?v.toFixed(3):v)).join('<br>');
    return `<div class="card"><div><strong>#${item.id}</strong> (${item.memory_type}) score=${(item.score||0).toFixed(4)}</div><div>${item.content}</div><div class="muted">breakdown:<br>${lines||'n/a'}</div></div>`;
  }).join('');
});

// Concept graph (kept from original)
function drawGraph(){
  const canvas=$('#concept-canvas');const ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1;const w=canvas.clientWidth;const h=canvas.clientHeight;
  canvas.width=Math.max(1,Math.floor(w*dpr));canvas.height=Math.max(1,Math.floor(h*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const spring=0.0022,repulsion=3800,damping=0.86;
  for(const e of state.conceptEdges){const s=state.concepts[e.si],t=state.concepts[e.ti];if(!s||!t)continue;const dx=t.x-s.x,dy=t.y-s.y,d=Math.hypot(dx,dy)||20,f=spring*(d-80),fx=dx/d*f,fy=dy/d*f;s.vx+=fx*0.02;s.vy+=fy*0.02;t.vx-=fx*0.02;t.vy-=fy*0.02;}
  for(let i=0;i<state.concepts.length;i++){const a=state.concepts[i];for(let j=i+1;j<state.concepts.length;j++){const b=state.concepts[j];const dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy+35,f=-repulsion/d2,inv=Math.sqrt(d2);a.vx+=dx/inv*f;a.vy+=dy/inv*f;b.vx-=dx/inv*f;b.vy-=dy/inv*f;}}
  for(const n of state.concepts){n.vx+=(w/2-n.x)*0.0009;n.vy+=(h/2-n.y)*0.0009;n.x+=n.vx;n.y+=n.vy;n.vx*=damping;n.vy*=damping;n.x=Math.min(w-16,Math.max(16,n.x));n.y=Math.min(h-16,Math.max(16,n.y));}
  ctx.clearRect(0,0,w,h);
  for(const e of state.conceptEdges){const s=state.concepts[e.si],t=state.concepts[e.ti];if(!s||!t)continue;ctx.strokeStyle='rgba(122,162,255,0.4)';ctx.lineWidth=Math.max(0.8,Math.min(5,e.weight*2.2));ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(t.x,t.y);ctx.stroke();}
  for(const n of state.concepts){const r=6+(n.size||2)*1.8;ctx.fillStyle=state.selectedNode===n?'#ffbf76':'#7aa2ff';ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);ctx.fill();ctx.fillStyle='#cce0ff';ctx.font='11px monospace';ctx.fillText(n.id,n.x+r+4,n.y+3);}
}
function startGraph(){
  if(!state.concepts.length){$('#graph-meta').textContent='No concepts yet.';return;}
  $('#graph-meta').textContent=state.concepts.length+' nodes, '+state.conceptEdges.length+' edges';
  for(const n of state.concepts){n.x=Math.random()*360+40;n.y=Math.random()*220+40;n.vx=0;n.vy=0;}
  (function tick(){drawGraph();requestAnimationFrame(tick)})();
}
$('#concept-canvas').addEventListener('click',e=>{
  const rect=e.target.getBoundingClientRect();const x=e.clientX-rect.left,y=e.clientY-rect.top;
  for(const n of state.concepts){if(Math.hypot(x-n.x,y-n.y)<=6+(n.size||2)*1.8+3){
    state.selectedNode=n;const ids=state.conceptToMemories[n.id]||[];
    const mems=ids.map(id=>state.memoriesById[id]).filter(Boolean);
    $('#graph-memories').innerHTML=mems.length?mems.map(m=>`<div class="card">#${m.id} · ${m.content}</div>`).join(''):'<div class="muted">No memories.</div>';
    $('#graph-meta').textContent=`node=${n.id}, memories=${ids.length}, activation=${n.activation.toFixed(3)}`;return;
  }}
});

// Init
async function init(){
  await loadHealth();
  const memories=await getJSON('/api/memories');
  state.memories=memories;state.memoriesById={};
  for(const m of memories)state.memoriesById[m.id]=m;

  const concepts=await getJSON('/api/concepts');
  state.concepts=(concepts.nodes||[]).map((n,i)=>({...n,i,vx:0,vy:0,x:0,y:0}));
  state.conceptEdges=(concepts.edges||[]).map(e=>({...e,si:state.concepts.findIndex(n=>n.id===e.source),ti:state.concepts.findIndex(n=>n.id===e.target)})).filter(e=>e.si>=0&&e.ti>=0);
  state.conceptToMemories=concepts.concept_to_memories||{};
  startGraph();

  await loadBeliefs();
  await loadContradictions();
  await loadDigests();
}
init().catch(e=>{$('#health-stats').textContent='Init error: '+(e.message||e)});
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

    # ── helpers ──────────────────────────────────────────────

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
            memory_concepts[memory_id] = concepts
            for concept in concepts:
                concept_to_memory_ids[concept].append(memory_id)

        nodes: List[Dict[str, Any]] = []
        for concept, memory_ids in sorted(concept_to_memory_ids.items()):
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
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    co_occurrence[(unique[i], unique[j])] += 1.0

        edges = [{"source": s, "target": t, "weight": float(w)} for (s, t), w in sorted(co_occurrence.items())]
        return {"nodes": nodes, "edges": edges, "concept_to_memories": dict(concept_to_memory_ids)}

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
            versions = [self._serialize_belief(v) for v in self.synapse.belief_history(fact_key)]
            payload.append({"fact_key": fact_key, "current": self._serialize_belief(current), "versions": versions})
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

    def _collect_health(self) -> Dict[str, Any]:
        """Extended health stats for the dashboard."""
        triple_count = len(self.synapse.triple_index._triples) if hasattr(self.synapse, 'triple_index') else 0
        last_sleep = getattr(self.synapse, '_last_sleep_at', None)
        last_sleep_display = "never"
        if last_sleep:
            from datetime import datetime, timezone
            last_sleep_display = datetime.fromtimestamp(last_sleep, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Storage size estimate
        mem_count = self.synapse.count()
        storage_bytes = 0
        for mid, mdata in self.synapse.store.memories.items():
            storage_bytes += len(json.dumps(mdata))
        if storage_bytes > 1_000_000:
            storage_display = f"{storage_bytes / 1_000_000:.1f} MB"
        elif storage_bytes > 1000:
            storage_display = f"{storage_bytes / 1000:.1f} KB"
        else:
            storage_display = f"{storage_bytes} B"

        return {
            "memory_count": mem_count,
            "concept_count": len(self.synapse.concept_graph.concepts),
            "triple_count": triple_count,
            "contradiction_count": len(self.synapse.contradictions()),
            "belief_count": len(self.synapse.beliefs()),
            "edge_count": len(self.synapse.store.edges),
            "last_sleep_at": last_sleep,
            "last_sleep_display": last_sleep_display,
            "storage_bytes": storage_bytes,
            "storage_display": storage_display,
        }

    def _collect_timeline(self, from_date: str = "", to_date: str = "") -> Dict[str, Any]:
        """Return memories in a date range, with promotion detection."""
        from datetime import datetime, timezone
        items = []
        from_ts = 0.0
        to_ts = float('inf')
        if from_date:
            try:
                from_ts = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        if to_date:
            try:
                to_ts = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() + 86400
            except ValueError:
                pass

        for memory_id, memory_data in self.synapse.store.memories.items():
            if memory_data.get("consolidated", False):
                continue
            created = memory_data.get("created_at", 0)
            if created < from_ts or created > to_ts:
                continue
            memory = self.synapse._memory_data_to_object(memory_data)
            meta = memory.metadata if isinstance(memory.metadata, dict) else {}
            is_promotion = meta.get("promoted_from") is not None or memory.memory_level == "general"
            item = {
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "memory_level": memory.memory_level,
                "effective_strength": float(memory.effective_strength),
                "created_at": float(memory.created_at),
                "is_promotion": is_promotion,
            }
            items.append(item)

        items.sort(key=lambda x: x["created_at"], reverse=True)
        return {"memories": items}

    def _collect_digests(self) -> Dict[str, Any]:
        """Read sleep digests from ~/.synapse/digests/."""
        digest_dir = os.path.expanduser("~/.synapse/digests")
        digests = []
        if os.path.isdir(digest_dir):
            files = sorted(os.listdir(digest_dir), reverse=True)[:7]
            for fname in files:
                fpath = os.path.join(digest_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    digests.append({
                        "date": data.get("date", fname.replace(".json", "")),
                        "consolidated": data.get("consolidated", 0),
                        "promoted": data.get("promoted", 0),
                        "pruned": data.get("pruned", 0),
                        "hot_topics": data.get("hot_topics", []),
                    })
                except (json.JSONDecodeError, OSError):
                    digests.append({"date": fname, "consolidated": 0, "promoted": 0, "pruned": 0, "hot_topics": []})
        return {"digests": digests}

    def _collect_recall(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        memories = self.synapse.recall(context=query, limit=limit, explain=True)
        results = [
            self._serialize_memory(memory, include_score=True, concepts=[name for name, _ in extract_concepts(memory.content)])
            for memory in memories
        ]
        return {"query": query, "count": len(results), "results": results}

    def _collect_compile(self, query: str = "", budget: int = 2000, policy: str = "balanced") -> Dict[str, Any]:
        pack: ContextPack = self.synapse.compile_context(query=query, budget=int(budget), policy=policy)
        return {"query": query, "budget": budget, "policy": policy, "context_text": pack.to_system_prompt(), "pack": pack.to_dict()}

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

    # ── handler ──────────────────────────────────────────────

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

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", 0))
                return self.rfile.read(length) if length else b""

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
                    elif path == "/api/health":
                        payload = inspector._collect_health()
                    elif path == "/api/memories":
                        payload = inspector._collect_memories()
                    elif path == "/api/concepts":
                        payload = inspector._collect_concepts()
                    elif path == "/api/contradictions":
                        payload = inspector._collect_contradictions()
                    elif path == "/api/beliefs":
                        payload = inspector._collect_beliefs(params.get("q", [None])[0] or "")
                    elif path == "/api/timeline":
                        payload = inspector._collect_timeline(
                            from_date=(params.get("from", [""])[0] or ""),
                            to_date=(params.get("to", [""])[0] or ""),
                        )
                    elif path == "/api/digests":
                        payload = inspector._collect_digests()
                    elif path == "/api/pending":
                        from review_queue import ReviewQueue
                        rq = ReviewQueue(inspector.synapse)
                        payload = {"count": rq.count(), "items": rq.list_pending()}
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

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    body_raw = self._read_body()
                    body = json.loads(body_raw) if body_raw else {}

                    if path == "/api/resolve-contradiction":
                        cid = body.get("contradiction_id")
                        winner = body.get("winner_memory_id")
                        if cid is None or winner is None:
                            return self._send_json({"error": "contradiction_id and winner_memory_id required"}, status=400)
                        inspector.synapse.resolve_contradiction(int(cid), int(winner))
                        return self._send_json({"status": "resolved"})

                    elif path == "/api/preview-context":
                        query = body.get("query", "")
                        budget = int(body.get("budget", 2000))
                        policy = body.get("policy", "balanced")
                        payload = inspector._collect_compile(query=query, budget=budget, policy=policy)
                        return self._send_json(payload)

                    else:
                        return self._send_json({"error": "Unknown POST endpoint"}, status=404)
                except Exception as exc:
                    return self._send_json({"error": str(exc)}, status=500)

            def log_message(self, format: str, *args):  # noqa: A003
                return

        return _Handler
