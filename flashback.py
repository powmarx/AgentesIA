"""Flashback: replay a saved experiment, test by test, in the browser.

The data is sent once as JSON to an embedded HTML page, and the animation runs
client-side (no Streamlit reruns, so no flicker). Each frame shows the test input,
the expected output and the model's answer, plus a running scoreboard per level.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
# Fixed height on purpose: height="content" would resize the iframe as frames change
# (grids of different sizes) and make the page jump during playback.
PLAYER_HEIGHT = 700

HTML = """
<style>
  body { font-family: "Source Sans", system-ui, sans-serif; margin: 0; color: #1f2937; }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  button, select { font-size: 15px; padding: 6px 12px; border: 1px solid #9ca3af; border-radius: 6px;
                   background: #fff; cursor: pointer; }
  button:hover { background: #eff6ff; }
  button.play { background: #2171B5; color: #fff; border-color: #2171B5; min-width: 92px; }
  input[type=range] { flex: 1; min-width: 180px; accent-color: #2171B5; }
  #status { font-size: 15px; margin: 6px 0 10px; min-height: 24px; }
  .chip { display: inline-block; padding: 2px 10px; border-radius: 12px; font-weight: 600;
          border: 1px solid #4b5563; }
  .verdict { font-weight: 700; margin-left: 8px; }
  .ok { color: #047857; } .bad { color: #b91c1c; } .warn { color: #92400e; }
  .grids { display: flex; gap: 20px; flex-wrap: wrap; }
  .card { border: 1px solid #d1d5db; border-radius: 8px; padding: 8px 10px; min-width: 230px; }
  .card h4 { margin: 0 0 6px; font-size: 14px; }
  .card .note { font-size: 12px; color: #6b7280; min-height: 16px; }
  .score { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
  .score div { border: 1px solid #d1d5db; border-radius: 8px; padding: 6px 12px; font-size: 14px; }
  .score b { font-size: 18px; }
  .tokens { font-size: 13px; color: #4b5563; margin-top: 6px; }
</style>
<div class="controls">
  <button id="first" title="Início">⏮</button>
  <button id="prev" title="Anterior">◀</button>
  <button id="play" class="play">▶ Tocar</button>
  <button id="next" title="Próximo">▶</button>
  <label>Velocidade
    <select id="speed">
      <option value="0.25">0,25 teste/s</option>
      <option value="0.5" selected>0,5 teste/s</option>
      <option value="1">1 teste/s</option>
      <option value="2">2 testes/s</option>
      <option value="4">4 testes/s</option>
    </select>
  </label>
  <input id="slider" type="range" min="0" max="0" value="0" aria-label="Posição no experimento">
</div>
<div id="status" role="status" aria-live="polite"></div>
<div class="grids">
  <div class="card"><h4>Entrada do teste</h4><canvas id="c_in"></canvas><div class="note" id="n_in"></div></div>
  <div class="card"><h4>Saída esperada</h4><canvas id="c_exp"></canvas><div class="note" id="n_exp"></div></div>
  <div class="card"><h4>Resposta do modelo</h4><canvas id="c_pred"></canvas><div class="note" id="n_pred"></div></div>
</div>
<div class="tokens" id="tokens"></div>
<div class="score" id="score"></div>
<script>
const DATA = __DATA__;
const PALETTE = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25"];
const MAXPX = 220;
const $ = id => document.getElementById(id);
const fmt = n => Math.round(n).toLocaleString("pt-BR");
let idx = 0, timer = null;

function draw(canvas, grid, ref) {
  const ctx = canvas.getContext("2d");
  if (!grid) { canvas.width = MAXPX; canvas.height = 60; ctx.fillStyle = "#6b7280";
    ctx.font = "14px sans-serif"; ctx.fillText("(sem resposta válida)", 12, 34); return; }
  const h = grid.length, w = Math.max(...grid.map(r => r.length));
  const cell = Math.max(4, Math.floor(MAXPX / Math.max(h, w)));
  canvas.width = w * cell; canvas.height = h * cell;
  for (let y = 0; y < h; y++) for (let x = 0; x < grid[y].length; x++) {
    ctx.fillStyle = PALETTE[grid[y][x]] || "#fff";
    ctx.fillRect(x * cell, y * cell, cell, cell);
    ctx.strokeStyle = "#444"; ctx.lineWidth = 0.5;
    ctx.strokeRect(x * cell, y * cell, cell, cell);
    if (ref && (ref[y] === undefined || ref[y][x] !== grid[y][x])) {
      ctx.strokeStyle = "#ff0000"; ctx.lineWidth = 2;
      ctx.strokeRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
    }
  }
}

function levelChip(level) {
  const i = DATA.levels.indexOf(level);
  const dark = i >= 2;
  return `<span class="chip" style="background:${DATA.colors[level]};color:${dark ? "#fff" : "#1f2937"}">${level}</span>`;
}

function render() {
  const e = DATA.events[idx];
  const t = DATA.tasks[e.task][e.test];
  const shape = g => g ? g.length + "×" + Math.max(...g.map(r => r.length)) : "—";
  draw($("c_in"), t.input, null);
  draw($("c_exp"), t.output, null);
  draw($("c_pred"), e.pred, t.output);
  $("n_in").textContent = shape(t.input);
  $("n_exp").textContent = shape(t.output);
  $("n_pred").textContent = e.pred ? shape(e.pred) + (shape(e.pred) !== shape(t.output) ? " (formato diferente)" : "")
                                   : "sem resposta" + (e.err ? ": " + e.err.slice(0, 80) : "");
  $("status").innerHTML = `Teste ${idx + 1} de ${DATA.events.length} · tarefa <b>${e.task}</b> #${e.test + 1} · `
    + levelChip(e.level)
    + (e.err ? `<span class="verdict warn">! erro de API (fora da conta)</span>`
             : `<span class="verdict ${e.ok ? "ok" : "bad"}">${e.ok ? "✓ acertou" : "✗ errou"}</span>`);
  $("tokens").textContent = `Tokens: entrada ${fmt(e.it)}, raciocínio ${fmt(e.rt)}, resposta ${fmt(e.at)} · ${e.sec.toFixed(0)} s`;
  const slider = $("slider");
  slider.value = idx;
  slider.setAttribute("aria-valuetext", `Teste ${idx + 1} de ${DATA.events.length}`);
  const sc = {};
  DATA.levels.forEach(l => sc[l] = {ok: 0, n: 0, rt: 0, err: 0});
  for (let i = 0; i <= idx; i++) {
    const x = DATA.events[i];
    if (x.err) { sc[x.level].err++; continue; }  // API failures say nothing about the model
    sc[x.level].n++; sc[x.level].ok += x.ok ? 1 : 0; sc[x.level].rt += x.rt + x.at;
  }
  $("score").innerHTML = DATA.levels.filter(l => DATA.present.includes(l)).map(l =>
    `<div>${levelChip(l)}<br><b>${sc[l].ok}/${sc[l].n}</b> testes certos<br>`
    + `<span style="font-size:12px;color:#4b5563">${fmt(sc[l].n ? sc[l].rt / sc[l].n : 0)} tokens de saída/teste`
    + (sc[l].err ? ` · ${sc[l].err} erro(s) de API fora da conta` : "") + `</span></div>`).join("");
}

function goTo(i) { idx = Math.max(0, Math.min(DATA.events.length - 1, i)); render(); }
function stop() { if (timer) clearInterval(timer); timer = null; $("play").textContent = "▶ Tocar"; }
function play() {
  if (idx >= DATA.events.length - 1) goTo(0);
  $("play").textContent = "⏸ Pausar";
  timer = setInterval(() => { if (idx >= DATA.events.length - 1) { stop(); return; } goTo(idx + 1); },
                      1000 / parseFloat($("speed").value));
}
function restart() { if (timer) { stop(); play(); } }

$("play").onclick = () => timer ? stop() : play();
$("next").onclick = () => { stop(); goTo(idx + 1); };
$("prev").onclick = () => { stop(); goTo(idx - 1); };
$("first").onclick = () => { stop(); goTo(0); };
$("speed").onchange = restart;
$("slider").max = DATA.events.length - 1;
$("slider").oninput = ev => { stop(); goTo(parseInt(ev.target.value)); };

// Small API for tests: goTo, getState, play, stop.
window.flashback = { goTo, play, stop, getState: () => ({ index: idx, playing: !!timer, total: DATA.events.length }) };
render();
</script>
"""


def build_payload(ev: pd.DataFrame, levels: list[str], colors: dict[str, str]) -> dict:
    """Pack events plus the task grids once; events refer to tasks by id."""
    ev = ev.sort_values("time", kind="stable")
    tasks = {}
    for tid in ev["task_id"].unique():
        data = json.loads((DATA_DIR / f"{tid}.json").read_text())
        tasks[tid] = [{"input": t["input"], "output": t["output"]} for t in data["test"]]
    events = []
    for r in ev.itertuples():
        pred = r.predicted if isinstance(getattr(r, "predicted", None), list) else None
        err = r.error if isinstance(r.error, str) else None
        events.append({
            "task": r.task_id, "test": int(r.test_index), "level": r.reasoning, "ok": bool(r.correct),
            "pred": pred, "err": err, "it": int(r.input_tokens), "rt": int(r.reasoning_tokens),
            "at": int(r.answer_tokens), "sec": float(r.seconds),
        })
    present = [l for l in levels if l in set(ev["reasoning"])]
    return {"tasks": tasks, "events": events, "levels": levels, "present": present, "colors": colors}


def render_player(ev: pd.DataFrame, levels: list[str], colors: dict[str, str]) -> None:
    # The HTML is built only from our own data (task grids, integer answer grids, numbers);
    # text from errors is inserted with textContent inside the player, never as HTML.
    payload = json.dumps(build_payload(ev, levels, colors)).replace("</", "<\\/")
    # Same data -> byte-identical HTML, so Streamlit reruns keep the iframe (and the playback).
    st.iframe(HTML.replace("__DATA__", payload), height=PLAYER_HEIGHT)
