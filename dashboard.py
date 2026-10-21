"""Streamlit dashboard for the ARC-AGI-1 solver.

Run with:  uv run streamlit run dashboard.py

Central question: does a higher reasoning level solve more ARC tasks, and at what
cost in tokens and time? The dashboard can launch main.py and, while it runs,
reads results/live_<label>_<level>.jsonl to update charts every few seconds.
"""

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from flashback import render_player
from llm import GOOGLE_MODELS, MODELS, NO_MINIMAL, REASONING_LEVELS, _google_keys, check_google_key

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
LEVELS = REASONING_LEVELS
# Allowed waits between API calls, in seconds (label shown in the sidebar combo).
WAIT_OPTIONS = {10: "10 s", 20: "20 s", 30: "30 s", 40: "40 s", 60: "1 min", 120: "2 min",
                180: "3 min", 300: "5 min"}

# One fixed color per reasoning level, used everywhere: a blue ramp, light to dark
# as reasoning grows (ordered, so it reads well and also works for colorblind users).
LEVEL_COLORS = {"minimal": "#C6DBEF", "low": "#6BAED6", "medium": "#2171B5", "high": "#08306B"}
# Result colors are separate from level colors (and always paired with a glyph).
STATE_COLORS = {"Acertou": "#009E73", "Errou": "#D9D9D9", "Erro de API": "#F4C27A",
                "Em andamento": "#FFFFFF"}
STATE_GLYPH = {"Acertou": "✓", "Errou": "✗", "Erro de API": "!", "Em andamento": "…"}

st.set_page_config(page_title="ARC-AGI-1: reasoning", layout="wide")

# ---------------------------------------------------------------- mode
# The published app is read-only. Experiments run only on your own machine, when BOTH:
#   1. ARC_LOCAL_MODE=1 is set in .env (the .env file is never committed or deployed), and
#   2. the browser is on the same machine (st.context.ip_address is None for localhost).
# Anything else (the cloud, another computer on the network) falls back to read-only.
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$")  # label becomes a file name


def run_mode_enabled() -> bool:
    return os.environ.get("ARC_LOCAL_MODE", "").strip() == "1" and st.context.ip_address is None


RUN_MODE = run_mode_enabled()


# ---------------------------------------------------------------- helpers
def fmt_int(n: float) -> str:
    """Brazilian thousands separator: 12345 -> 12.345."""
    return f"{round(n):,}".replace(",", ".")


def fmt_pct(x: float) -> str:
    return f"{x:.0f}%".replace(".", ",")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n (returned in percent)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * max(0, centre - half), 100 * min(1, centre + half)


@st.cache_data
def tests_in_task(task_id: str) -> int:
    return len(json.loads((DATA_DIR / f"{task_id}.json").read_text())["test"])


def read_ids(name: str, limit: int) -> list[str]:
    """Task ids of a tasks file in the project folder (works whatever the working directory)."""
    lines = (ROOT / Path(name).name).read_text().splitlines()
    ids = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    return ids[:limit] if limit else ids


def load_events() -> pd.DataFrame:
    """Read every live_*.jsonl file. Label and level come from the file name."""
    frames = []
    for f in RESULTS_DIR.glob("live_*.jsonl"):
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # line still being written
                continue
        if rows:
            df = pd.DataFrame(rows)
            df["label"] = f.stem.removeprefix("live_").rsplit("_", 1)[0]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    if "error" not in df:  # logs written before the error field existed
        df["error"] = None
    if "model" not in df:  # logs written before the model was recorded used the default model
        df["model"] = MODELS["google"]
    df["model"] = df["model"].fillna(MODELS["google"])
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def task_table(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, level): complete only when all its tests have a result."""
    ev = ev.assign(out_tokens=ev["reasoning_tokens"] + ev["answer_tokens"])
    g = ev.groupby(["task_id", "reasoning"], sort=False)
    t = g.agg(done=("correct", "size"), n_tests=("n_tests", "max"), right=("correct", "sum"),
              out_tokens=("out_tokens", "sum"), seconds=("seconds", "sum"), time=("time", "max"),
              api_error=("error", lambda s: bool(s.notna().any()))).reset_index()
    t["complete"] = t["done"] == t["n_tests"]
    # A task with an API failure says nothing about the model: it is left out of accuracy.
    t["valid"] = t["complete"] & ~t["api_error"]
    t["solved"] = t["valid"] & (t["right"] == t["n_tests"])

    def state(r) -> str:
        if not r["complete"]:
            return "Em andamento"
        if r["api_error"]:
            return "Erro de API"
        return "Acertou" if r["solved"] else "Errou"

    t["state"] = t.apply(state, axis=1)
    return t


# Task patterns across reasoning levels, in the order they are shown.
PATTERNS = {
    "ajudou": ("Mais reasoning ajudou",
               "Errou nos níveis baixos e acertou a partir de certo nível, sem voltar a errar nos mais altos."),
    "inconsistente": ("Resultado inconsistente",
                      "Acertou em algum nível e errou em outro mais alto. Pode ser acaso: o modelo "
                      "não é determinístico."),
    "todos": ("Acertou em todos os níveis", "Tarefas fáceis para o modelo, já no reasoning mais baixo."),
    "nenhum": ("Não acertou em nenhum nível", "Tarefas difíceis demais para este modelo."),
    "sem_dados": ("Sem comparação possível",
                  "Menos de 2 níveis com resultado válido (falha de API ou execução em andamento)."),
}


def classify_tasks(tasks: pd.DataFrame) -> dict[str, str]:
    """Pattern of each task across the reasoning levels that have a valid result."""
    result = {}
    for tid, g in tasks[tasks["valid"]].groupby("task_id"):
        g = g.assign(rank=g["reasoning"].map(LEVELS.index)).sort_values("rank")
        s = [bool(x) for x in g["solved"]]
        if len(s) < 2:
            result[tid] = "sem_dados"
        elif all(s):
            result[tid] = "todos"
        elif not any(s):
            result[tid] = "nenhum"
        elif s == sorted(s):  # False...False True...True: never gets worse with more reasoning
            result[tid] = "ajudou"
        else:
            result[tid] = "inconsistente"
    for tid in set(tasks["task_id"]) - set(result):
        result[tid] = "sem_dados"
    return result


def level_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (per completed task) and cost (per test) for each reasoning level."""
    tasks = task_table(ev)
    rows = []
    for level in LEVELS:
        e = ev[ev["reasoning"] == level]
        if e.empty:
            continue
        all_t = tasks[tasks["reasoning"] == level]
        t = all_t[all_t["valid"]]
        n, k = len(t), int(t["solved"].sum())
        lo, hi = wilson(k, n)
        ok = e[e["error"].isna()]  # failed calls have 0 tokens and would bias the cost averages
        rows.append({
            "reasoning": level,
            "tasks_done": n,
            "tasks_solved": k,
            "accuracy": 100 * k / n if n else 0.0,
            "ci_low": lo,
            "ci_high": hi,
            "tests": len(e),
            "reasoning_tokens": ok["reasoning_tokens"].mean() if len(ok) else 0.0,
            # At "minimal" the API reports 0 thinking tokens, but the model still reasons
            # out loud in the visible answer. Output tokens (thinking + answer) is the
            # comparable cost across levels.
            "output_tokens": (ok["reasoning_tokens"] + ok["answer_tokens"]).mean() if len(ok) else 0.0,
            "input_tokens": ok["input_tokens"].mean() if len(ok) else 0.0,
            "answer_tokens": ok["answer_tokens"].mean() if len(ok) else 0.0,
            "seconds": ok["seconds"].mean() if len(ok) else 0.0,
            "errors": int(e["error"].notna().sum()),
            "tasks_excluded": int(all_t["complete"].sum() - n),
        })
    return pd.DataFrame(rows)


def level_color_scale() -> alt.Scale:
    return alt.Scale(domain=LEVELS, range=[LEVEL_COLORS[l] for l in LEVELS])


# ---------------------------------------------------------------- charts
def accuracy_chart(s: pd.DataFrame) -> alt.Chart:
    s = s.assign(label=s.apply(lambda r: f"{fmt_pct(r.accuracy)} ({r.tasks_solved}/{r.tasks_done})", axis=1))
    base = alt.Chart(s).encode(x=alt.X("reasoning:N", sort=LEVELS, title="Nível de reasoning",
                                      scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2),
                                      axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(stroke="#555", strokeWidth=0.5).encode(
        # Domain goes past 100 so the label above the error bar is never clipped;
        # ticks stop at 100.
        y=alt.Y("accuracy:Q", title="Tarefas resolvidas (%)", scale=alt.Scale(domain=[0, 115]),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100])),
        color=alt.Color("reasoning:N", scale=level_color_scale(), legend=None),
    )
    # White halo under the dark line keeps the interval visible on the darkest bar.
    halo = base.mark_rule(color="white", strokeWidth=4.5).encode(y="ci_low:Q", y2="ci_high:Q")
    ci = base.mark_rule(color="#222", strokeWidth=1.5).encode(y="ci_low:Q", y2="ci_high:Q")
    text = base.mark_text(dy=-10, fontSize=15, fontWeight="bold", color="#222").encode(
        y="ci_high:Q", text="label:N")
    return (bars + halo + ci + text).properties(
        height=340, title="Acerto por nível (barra de erro: intervalo de 95%, Wilson)")


def cost_chart(s: pd.DataFrame, column: str, title: str, y_title: str, fmt) -> alt.Chart:
    s = s.assign(label=s[column].map(fmt))
    base = alt.Chart(s).encode(x=alt.X("reasoning:N", sort=LEVELS, title="Nível de reasoning",
                                      scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2),
                                      axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(stroke="#555", strokeWidth=0.5).encode(
        y=alt.Y(f"{column}:Q", title=y_title, axis=alt.Axis(
            labelExpr="replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')")),
        color=alt.Color("reasoning:N", scale=level_color_scale(), legend=None),
    )
    text = base.mark_text(dy=-10, fontSize=15, fontWeight="bold", color="#222").encode(
        y=f"{column}:Q", text="label:N")
    return (bars + text).properties(height=340, title=title)


CELL_STYLE = {  # background, text color
    "Acertou": ("#009E73", "#FFFFFF"),
    "Errou": ("#D9D9D9", "#333333"),
    "Erro de API": ("#F4C27A", "#333333"),
    "Em andamento": ("#FFFFFF", "#666666"),
}


def task_table_html(tasks: pd.DataFrame, kinds: dict[str, str], order: list[str],
                    levels: list[str]) -> str:
    """One HTML table: tasks grouped by pattern, one column per level, cost inside each cell.

    Plain HTML on purpose: it always renders at its natural size and is easy to read.
    """
    cell = {(r.task_id, r.reasoning): r for r in tasks.itertuples()}
    th = "padding:6px 10px;text-align:center;border-bottom:2px solid #888;"
    rows = [f'<tr><th style="{th}text-align:left">Tarefa</th>'
            + "".join(f'<th style="{th}"><span style="background:{LEVEL_COLORS[l]};'
                      f'color:{"#fff" if LEVELS.index(l) >= 2 else "#1f2937"};padding:2px 10px;'
                      f'border-radius:10px;font-weight:600">{l}</span></th>' for l in levels)
            + "</tr>"]
    for key, (title, explanation) in PATTERNS.items():
        ids = [t for t in order if kinds.get(t) == key]
        if not ids:
            continue
        rows.append(
            f'<tr><td colspan="{len(levels) + 1}" style="padding:14px 4px 4px;">'
            f'<b>{title}</b> ({len(ids)})<br><span style="font-size:13px;opacity:.75">'
            f'{explanation}</span></td></tr>')
        for tid in ids:
            tds = [f'<td style="padding:4px 10px;font-family:monospace">{tid}</td>']
            for l in levels:
                r = cell.get((tid, l))
                if r is None:
                    tds.append('<td style="border:1px solid #ddd;text-align:center;color:#aaa">–</td>')
                    continue
                bg, fg = CELL_STYLE[r.state]
                tokens = f'<br><span style="font-size:11px">{fmt_int(r.out_tokens)} tokens</span>' \
                    if r.state != "Em andamento" else ""
                tds.append(
                    f'<td title="{r.state}: {r.right}/{r.n_tests} testes certos" '
                    f'style="background:{bg};color:{fg};border:1px solid #fff;text-align:center;'
                    f'padding:4px 10px;min-width:90px"><b style="font-size:16px">'
                    f'{STATE_GLYPH[r.state]}</b>{tokens}</td>')
            rows.append("<tr>" + "".join(tds) + "</tr>")
    return ('<table style="border-collapse:collapse;font-size:14px">' + "".join(rows) + "</table>")


# ---------------------------------------------------------------- model comparison
# Model colors: no blues (those are the reasoning levels) and no green (that is "acertou").
# The first two are the most different from each other, because 2 models is the usual case.
MODEL_PALETTE = ["#D55E00", "#6A3D9A", "#CC79A7", "#8C564B", "#B8860B", "#4D4D4D", "#E7298A"]


def model_colors(models: list[str]) -> dict[str, str]:
    """Color by position in the list of models that have results, so it stays fixed in the tab."""
    return {m: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, m in enumerate(models)}


def short_model(model: str) -> str:
    return model.removeprefix("gemini-")


def all_task_results(ev_all: pd.DataFrame) -> pd.DataFrame:
    """Valid task results of every experiment: one row per (model, level, task).

    If the same model/level/task was run in more than one experiment, the most recent
    result is used, so a rerun replaces an older attempt.
    """
    frames = []
    for (label, model), g in ev_all.groupby(["label", "model"]):
        t = task_table(g)
        frames.append(t.assign(label=label, model=model))
    res = pd.concat(frames, ignore_index=True)
    res = res[res["valid"]].sort_values("time", kind="stable")
    return res.drop_duplicates(["model", "reasoning", "task_id"], keep="last").reset_index(drop=True)


def common_levels(res: pd.DataFrame, models: list[str]) -> list[str]:
    """Levels for which every chosen model has at least one valid task."""
    return [l for l in LEVELS if all(((res["model"] == m) & (res["reasoning"] == l)).any() for m in models)]


def common_results(res: pd.DataFrame, models: list[str], levels: list[str]) -> tuple[pd.DataFrame, int]:
    """Results restricted to the tasks that EVERY chosen model solved or failed at EVERY chosen level."""
    sub = res[res["model"].isin(models) & res["reasoning"].isin(levels)]
    sets = [set(g["task_id"]) for _, g in sub.groupby(["model", "reasoning"])]
    if not sets or len(sets) != len(models) * len(levels):
        return sub.iloc[0:0], 0
    common = set.intersection(*sets)
    return sub[sub["task_id"].isin(common)], len(common)


def compare_summary(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level), g in sub.groupby(["model", "reasoning"]):
        n, k = len(g), int(g["solved"].sum())
        lo, hi = wilson(k, n)
        rows.append({"model": model, "reasoning": level, "n": n, "k": k, "accuracy": 100 * k / n,
                     "ci_low": lo, "ci_high": hi, "out_tokens": g["out_tokens"].mean(),
                     "seconds": g["seconds"].mean()})
    return pd.DataFrame(rows)


def model_color_scale(models: list[str], colors: dict[str, str]) -> alt.Scale:
    return alt.Scale(domain=models, range=[colors[m] for m in models])


def compare_bar_chart(s: pd.DataFrame, models: list[str], colors: dict[str, str], column: str,
                      y_title: str, title: str, label_fmt, with_ci: bool = False) -> alt.Chart:
    """Grouped bars: one group per reasoning level, one bar per model (same layout in both charts)."""
    s = s.assign(label=s[column].map(label_fmt))
    base = alt.Chart(s).encode(
        x=alt.X("reasoning:N", sort=LEVELS, title="Nível de reasoning",
                axis=alt.Axis(labelAngle=0, labelFontSize=13),
                scale=alt.Scale(paddingInner=0.3, paddingOuter=0.15)),
        xOffset=alt.XOffset("model:N", sort=models, scale=alt.Scale(paddingInner=0.1)),
    )
    if with_ci:
        y = alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(domain=[0, 115]),
                  axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]))
    else:
        y = alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(nice=True), axis=alt.Axis(
            labelExpr="replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')"))
    tooltip = [alt.Tooltip("model:N", title="Modelo"), alt.Tooltip("reasoning:N", title="Nível"),
               alt.Tooltip("label:N", title=y_title), alt.Tooltip("n:Q", title="Tarefas")]
    if with_ci:
        tooltip += [alt.Tooltip("ci_low:Q", title="IC 95% de (%)", format=".0f"),
                    alt.Tooltip("ci_high:Q", title="IC 95% até (%)", format=".0f")]
    bars = base.mark_bar(stroke="#444", strokeWidth=0.5).encode(
        y=y,
        color=alt.Color("model:N", scale=model_color_scale(models, colors), legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelFontSize=13, symbolType="square",
            labelExpr="replace(datum.label, 'gemini-', '')")),
        tooltip=tooltip,
    )
    layers = [bars]
    top = column
    if with_ci:
        layers += [base.mark_rule(color="white", strokeWidth=4).encode(y="ci_low:Q", y2="ci_high:Q"),
                   base.mark_rule(color="#222", strokeWidth=1.5).encode(y="ci_low:Q", y2="ci_high:Q")]
        top = "ci_high"
    font = 13 if len(models) <= 2 else 11
    layers.append(base.mark_text(baseline="bottom", dy=-3, fontSize=font, fontWeight="bold",
                                 color="#222").encode(y=f"{top}:Q", text="label:N"))
    return alt.layer(*layers).properties(height=330, title=title)


def paired_chart(sub: pd.DataFrame, a: str, b: str, levels: list[str], colors: dict[str, str]) -> alt.Chart:
    """For each level, how many tasks: both solved / only A / only B / neither."""
    cats = ["Ambos acertaram", f"Só {short_model(a)}", f"Só {short_model(b)}", "Nenhum acertou"]
    fill = ["#009E73", colors[a], colors[b], "#D9D9D9"]
    rows = []
    for level in levels:
        p = sub[sub["reasoning"] == level].pivot(index="task_id", columns="model", values="solved").astype(bool)
        counts = [int((p[a] & p[b]).sum()), int((p[a] & ~p[b]).sum()),
                  int((~p[a] & p[b]).sum()), int((~p[a] & ~p[b]).sum())]
        start = 0
        for cat, n in zip(cats, counts):
            rows.append({"reasoning": level, "cat": cat, "n": n, "x0": start, "x1": start + n,
                         "mid": start + n / 2, "dark": cat != "Nenhum acertou"})
            start += n
    df = pd.DataFrame(rows)
    y = alt.Y("reasoning:N", sort=levels, title=None, axis=alt.Axis(labelFontSize=13))
    bars = alt.Chart(df[df["n"] > 0]).mark_bar(stroke="white", strokeWidth=1.5, height={"band": 0.7}).encode(
        y=y, x=alt.X("x0:Q", title="Tarefas", axis=alt.Axis(tickMinStep=1, format="d")), x2="x1:Q",
        color=alt.Color("cat:N", scale=alt.Scale(domain=cats, range=fill), legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelFontSize=13, symbolType="square")),
        tooltip=[alt.Tooltip("reasoning:N", title="Nível"), alt.Tooltip("cat:N", title="Resultado"),
                 alt.Tooltip("n:Q", title="Tarefas")],
    )
    text = alt.Chart(df[df["n"] > 0]).mark_text(fontSize=14, fontWeight="bold").encode(
        y=y, x="mid:Q", text="n:Q",
        color=alt.condition(alt.datum.dark, alt.value("white"), alt.value("#333")),
    )
    return (bars + text).properties(height=46 * len(levels) + 20)


def model_cards_html(sub: pd.DataFrame, models: list[str], colors: dict[str, str]) -> str:
    """One card per model with totals over the chosen levels (equal heights, generated numbers)."""
    cards = []
    for m in models:
        g = sub[sub["model"] == m]
        k, n = int(g["solved"].sum()), len(g)
        solved_tokens = g["out_tokens"].sum() / k if k else None
        cost = f"{fmt_int(solved_tokens)} tokens por acerto" if solved_tokens else "nenhum acerto"
        cards.append(
            f'<div style="flex:1;min-width:220px;border:1px solid #ddd;border-left:6px solid {colors[m]};'
            f'border-radius:8px;padding:10px 14px">'
            f'<div style="font-weight:600;font-size:15px">{m}</div>'
            f'<div style="font-size:30px;font-weight:700;line-height:1.3">{fmt_pct(100 * k / n)}</div>'
            f'<div style="font-size:13px;opacity:.8">{k} de {n} acertos (tarefa × nível)<br>'
            f'{fmt_int(g["out_tokens"].mean())} tokens de saída por tarefa · {cost}</div></div>')
    return '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(cards) + "</div>"


def compare_table_html(sub: pd.DataFrame, models: list[str], levels: list[str],
                       colors: dict[str, str]) -> str:
    """Tasks x (model, level): model header spanning its levels, same cell style as 'Por tarefa'."""
    cols = [(m, l) for m in models for l in levels]
    cell = {(r.task_id, r.model, r.reasoning): r for r in sub.itertuples()}
    tasks = sorted(sub["task_id"].unique())
    sep = "border-left:4px solid #fff;"  # gap between model groups

    def group(tid: str) -> str:
        vals = {bool(cell[(tid, m, l)].solved) for m, l in cols}
        return "all" if vals == {True} else "none" if vals == {False} else "differ"

    titles = {"differ": ("Os resultados divergem", "Alguma combinação de modelo e nível acertou e outra errou."),
              "all": ("Todos acertaram", "Tarefas fáceis para todas as combinações."),
              "none": ("Ninguém acertou", "Tarefas difíceis para todas as combinações.")}
    th = "padding:5px 6px;text-align:center;"
    row1 = ['<tr><th rowspan="2" style="' + th + 'text-align:left;vertical-align:bottom;'
            'border-bottom:2px solid #888">Tarefa</th>']
    for m in models:
        row1.append(f'<th colspan="{len(levels)}" style="{th}{sep}"><div style="background:{colors[m]};'
                    f'color:#fff;border-radius:6px;padding:3px 8px;font-weight:600">{m}</div></th>')
    row2 = ["<tr>"]
    for m, l in cols:
        first = sep if l == levels[0] else ""
        row2.append(f'<th style="{th}{first}font-size:12px;font-weight:600;border-bottom:2px solid #888">{l}</th>')
    rows = ["".join(row1) + "</tr>", "".join(row2) + "</tr>"]
    for key in ("differ", "all", "none"):
        ids = [t for t in tasks if group(t) == key]
        if not ids:
            continue
        title, expl = titles[key]
        rows.append(f'<tr><td colspan="{len(cols) + 1}" style="padding:14px 4px 4px"><b>{title}</b> '
                    f'({len(ids)})<br><span style="font-size:13px;opacity:.75">{expl}</span></td></tr>')
        for tid in ids:
            tds = [f'<td style="padding:4px 8px;font-family:monospace">{tid}</td>']
            for m, l in cols:
                r = cell[(tid, m, l)]
                bg, fg = CELL_STYLE[r.state]
                first = sep if l == levels[0] else "border-left:1px solid #fff;"
                tds.append(f'<td title="{m} · {l}: {r.state}, {r.right}/{r.n_tests} testes certos" '
                           f'style="background:{bg};color:{fg};{first}border-top:1px solid #fff;'
                           f'text-align:center;padding:3px 6px;min-width:70px">'
                           f'<b style="font-size:15px">{STATE_GLYPH[r.state]}</b><br>'
                           f'<span style="font-size:11px">{fmt_int(r.out_tokens)}</span></td>')
            rows.append("<tr>" + "".join(tds) + "</tr>")
    return '<table style="border-collapse:collapse;font-size:14px">' + "".join(rows) + "</table>"


# ---------------------------------------------------------------- run control (local only)
st.session_state.setdefault("api_keys", [])  # extra keys added in this browser session


def mask_key(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "…"


def task_files() -> list[str]:
    """Only tasks*.txt files of the project can be chosen (no free-form paths)."""
    return sorted(f.name for f in ROOT.glob("tasks*.txt"))


def experiment_exists(label: str) -> bool:
    return any(RESULTS_DIR.glob(f"live_{label}_*.jsonl"))


def all_keys() -> list[str]:
    """Keys from .env (GOOGLE_API_KEY, _2, _3, ...) followed by the ones added in this session."""
    keys = [v for _, v in _google_keys()]
    return keys + [k for k in st.session_state.api_keys if k not in keys]


def solver_env(keys: list[str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GOOGLE_API_KEY")}
    env["ARC_NO_DOTENV"] = "1"  # the keys below are the complete list
    for i, key in enumerate(keys):
        env["GOOGLE_API_KEY" if i == 0 else f"GOOGLE_API_KEY_{i + 1}"] = key
    return env


def start_run(levels: list[str], model: str, wait: int, tasks_file: str, label: str, limit: int) -> None:
    if not RUN_MODE:
        raise PermissionError("Esta versão é só para consulta.")
    if tasks_file not in task_files():
        raise ValueError("Arquivo de tarefas inválido.")
    if not LABEL_RE.match(label):
        raise ValueError("Rótulo inválido.")
    if experiment_exists(label):  # main.py starts each live file fresh: never overwrite
        raise FileExistsError(f"Já existe um experimento chamado {label}.")
    keys = all_keys()
    if not keys:
        raise PermissionError("Nenhuma chave da API disponível.")
    RESULTS_DIR.mkdir(exist_ok=True)
    log = open(RESULTS_DIR / f"run_{label}.log", "w", encoding="utf-8")
    cmd = [sys.executable, "-u", "main.py", "--file", tasks_file, "--label", label,
           "--model", model, "--wait", str(wait), "--reasoning", *levels]
    if limit:
        cmd += ["--limit", str(limit)]
    st.session_state.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             env=solver_env(keys))
    log.close()  # the solver has its own handle; keeping ours open would lock the file (Windows)
    ids = read_ids(tasks_file, limit)
    st.session_state.run = {
        "label": label, "levels": levels, "model": model,
        "expected_per_level": sum(tests_in_task(t) for t in ids),
        "n_tasks": len(ids),
    }
    st.session_state.handled_finish = False


def delete_experiment(label: str) -> int:
    """Delete every file of one experiment (live logs, summaries, terminal log)."""
    if not RUN_MODE:
        raise PermissionError("Esta versão é só para consulta.")
    if not LABEL_RE.match(label):
        raise ValueError("Rótulo inválido.")
    victims = [RESULTS_DIR / f"live_{label}_{lvl}.jsonl" for lvl in LEVELS]
    victims.append(RESULTS_DIR / f"run_{label}.log")
    for f in RESULTS_DIR.glob("summary_*.json"):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("label") == label:
                victims.append(f)
        except (json.JSONDecodeError, OSError):
            continue
    removed = 0
    for f in victims:
        try:
            f.unlink()
            removed += 1
        except OSError:  # missing, or still open by a solver that is finishing
            continue
    return removed


def is_running() -> bool:
    p = st.session_state.get("proc")
    return p is not None and p.poll() is None


def render_keys() -> None:
    """Show the .env keys (masked) and allow extra keys for this session."""
    env_keys = [v for _, v in _google_keys()]
    extra = st.session_state.api_keys
    total = len(env_keys) + len(extra)
    with st.expander(f"Chaves da API Google ({total})", expanded=total == 0):
        st.caption(f"{len(env_keys)} do arquivo .env. Chaves adicionadas aqui valem só nesta sessão "
                   f"e não são gravadas. A primeira é usada antes; 429/401/403 passa para a próxima.")
        for i, key in enumerate(env_keys):
            st.markdown(f"`{i + 1}.` `{mask_key(key)}` (.env)")
        for i, key in enumerate(extra):
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.markdown(f"`{len(env_keys) + i + 1}.` `{mask_key(key)}`")
            if c2.button("✕", key=f"del_key_{i}", help="Remover esta chave", disabled=is_running()):
                extra.pop(i)
                st.rerun()
        with st.form("add_key", clear_on_submit=True, border=False):
            new = st.text_input("Nova chave", type="password", placeholder="cole a chave aqui",
                                label_visibility="collapsed")
            if st.form_submit_button("Adicionar chave", width="stretch", disabled=is_running()):
                new = new.strip()
                if not new:
                    st.warning("Cole uma chave antes de adicionar.")
                elif new in env_keys or new in extra:
                    st.warning("Essa chave já está na lista.")
                else:
                    ok, msg = check_google_key(new)
                    if ok:
                        extra.append(new)
                        st.rerun()
                    st.error(f"Chave não adicionada: {msg}")


# ---------------------------------------------------------------- sidebar
st.session_state.setdefault("run_label", f"exp_{datetime.now():%m%d_%H%M}")

with st.sidebar:
    if RUN_MODE:
        st.header("Executar experimento")
        render_keys()
        has_keys = bool(all_keys())
        model_sel = st.selectbox("Modelo (API Gemini)", GOOGLE_MODELS, disabled=is_running(),
                                 help="Modelos de texto com raciocínio disponíveis na API Gemini.")
        level_options = [l for l in LEVELS if not (l == "minimal" and model_sel in NO_MINIMAL)]
        if len(level_options) < len(LEVELS):
            st.caption(f"{model_sel} não aceita o nível «minimal».")
        levels_sel = st.multiselect("Níveis de reasoning (rodam em sequência)", level_options,
                                    default=level_options, disabled=is_running())
        wait_sel = st.selectbox("Espera entre chamadas à API", list(WAIT_OPTIONS), index=0,
                                format_func=WAIT_OPTIONS.get, disabled=is_running(),
                                help="Tempo entre o fim de uma chamada e o início da próxima. Esperas "
                                     "curtas aumentam o risco de limite de taxa (erro 429).")
        files_avail = task_files()
        tasks_file = st.selectbox("Arquivo de tarefas", files_avail, disabled=is_running(),
                                  help="Arquivos tasks*.txt da pasta do projeto.")
        limit = st.number_input("Usar só as N primeiras tarefas (0 = todas)", 0, 400, 5,
                                disabled=is_running())
        label_in = st.text_input("Rótulo do experimento", key="run_label", disabled=is_running())
        label = label_in.strip()
        label_bad = bool(label) and not LABEL_RE.match(label)
        label_taken = not label_bad and bool(label) and experiment_exists(label) and not is_running()
        if label_bad:
            st.caption("Use só letras, números, «_», «-» e «.», sem espaços (até 60 caracteres).")
        elif label_taken:
            st.caption("Já existe um experimento com esse rótulo. Escolha outro.")

        n_ids = 0
        if tasks_file:
            try:
                ids = read_ids(tasks_file, int(limit))
                n_ids = len(ids)
                n_tests = sum(tests_in_task(t) for t in ids)
                min_min = n_tests * len(levels_sel) * wait_sel / 60
                st.caption(f"{n_ids} tarefas, {n_tests} testes por nível. Só a espera de "
                           f"{WAIT_OPTIONS[wait_sel]} entre chamadas já soma cerca de {min_min:.0f} min; "
                           f"com reasoning alto, leva bem mais.")
            except (OSError, ValueError, KeyError):
                n_ids = 0
                st.error(f"Não consegui ler {tasks_file}: confira se todas as tarefas existem em data/.")
        else:
            st.error("Nenhum arquivo tasks*.txt na pasta do projeto.")

        if not has_keys:
            st.caption("Adicione uma chave da API (no .env ou acima) para poder iniciar.")
        c1, c2 = st.columns(2)
        if c1.button("Iniciar", type="primary", width="stretch",
                     disabled=(is_running() or not has_keys or not levels_sel or not n_ids
                               or not label or label_taken or label_bad)):
            ordered = [l for l in LEVELS if l in levels_sel]
            start_run(ordered, model_sel, int(wait_sel), tasks_file, label, int(limit))
            st.session_state.view_label = label
            st.rerun()
        if c2.button("Parar", width="stretch", disabled=not is_running()):
            st.session_state.proc.terminate()
            st.rerun()
        st.divider()
    else:
        st.info("Versão para consulta: os experimentos foram rodados localmente pelo autor. "
                "Para rodar os seus, veja o README do projeto.")

    st.header("Visualizar")
    files = sorted(RESULTS_DIR.glob("live_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    labels = list(dict.fromkeys(f.stem.removeprefix("live_").rsplit("_", 1)[0] for f in files))
    run = st.session_state.get("run")
    if run and run["label"] not in labels:
        labels.insert(0, run["label"])
    if labels:
        if st.session_state.get("view_label") not in labels:
            st.session_state.view_label = labels[0]
        view_label = st.selectbox("Experimento", labels, key="view_label")
    else:
        view_label = None

    if RUN_MODE and view_label:
        busy = is_running() and run and run["label"] == view_label
        with st.popover("Excluir experimento", width="stretch", disabled=bool(busy)):
            st.warning(f"Isso apaga de forma permanente os resultados e as respostas de "
                       f"**{view_label}** (painel e flashback). Não dá para desfazer.")
            if st.button("Sim, excluir", type="primary", key="confirm_delete"):
                n_removed = delete_experiment(view_label)
                st.session_state.pop("flash_label", None)
                if run and run["label"] == view_label:
                    st.session_state.pop("run", None)
                st.toast(f"{view_label}: {n_removed} arquivos removidos")
                st.rerun()
        if busy:
            st.caption("Pare a execução antes de excluir este experimento.")
    view = st.radio("Seção do painel", ["Visão geral", "Por tarefa", "Registro ao vivo"], horizontal=False)


# ---------------------------------------------------------------- main page
st.title("Mais reasoning resolve mais tarefas ARC-AGI-1?")
st.caption("Cada tarefa só conta como resolvida se todos os seus testes estiverem corretos.")
tab_panel, tab_flash, tab_compare = st.tabs(["Painel", "Flashback", "Comparação de modelos"])


def render() -> None:
    run = st.session_state.get("run")
    running = is_running()

    # Finished: trigger one full rerun so the sidebar buttons unlock.
    if run and not running and not st.session_state.get("handled_finish"):
        st.session_state.handled_finish = True
        st.rerun()

    ev_all = load_events()
    ev = ev_all[ev_all["label"] == view_label] if (view_label and not ev_all.empty) else pd.DataFrame()

    if not ev.empty:
        models = list(dict.fromkeys(ev["model"]))
        st.markdown(f"Modelo: **{", ".join(models)}**")

    if run and run["label"] == view_label:
        total = run["expected_per_level"] * len(run["levels"])
        done = len(ev)
        st.progress(min(done / total, 1.0) if total else 0.0,
                    text=f"{'Executando' if running else 'Finalizado'}: {done}/{total} testes "
                         f"({', '.join(run['levels'])})")
        with st.expander("Saída do terminal"):
            lines = (RESULTS_DIR / f"run_{run['label']}.log").read_text(encoding="utf-8").splitlines()
            st.code("\n".join(lines[-15:]) or "(sem saída ainda)")

    if ev.empty:
        st.info("Nenhum resultado ainda. Inicie um experimento na barra lateral ou escolha outro.")
        return

    summary = level_summary(ev)
    tasks = task_table(ev)
    levels_present = [l for l in LEVELS if l in set(ev["reasoning"])]

    if view == "Visão geral":
        # Generated headline: every number comes from the data.
        valid = summary[summary["tasks_done"] > 0]
        if not valid.empty:
            # Highest accuracy; on ties, the one with fewer reasoning tokens.
            best = valid.sort_values(["accuracy", "output_tokens"], ascending=[False, True]).iloc[0]
            cheapest = valid.loc[valid["output_tokens"].idxmin()]
            msg = (f"Até agora, **{best.reasoning}** teve o maior acerto: {fmt_pct(best.accuracy)} "
                   f"({int(best.tasks_solved)}/{int(best.tasks_done)} tarefas), com média de "
                   f"{fmt_int(best.output_tokens)} tokens de saída por teste.")
            if len(valid) > 1:
                msg += (f" O nível mais barato é **{cheapest.reasoning}**, com "
                        f"{fmt_int(cheapest.output_tokens)} tokens de saída por teste.")
            st.markdown(msg)
        excluded = int(summary["tasks_excluded"].sum())
        if excluded:
            st.warning(f"{excluded} tarefas tiveram falha de API (por exemplo 503) e ficaram fora do "
                       f"cálculo de acerto e de custo. Elas aparecem com «!» na seção Por tarefa. "
                       f"Rode de novo para completá-las.")
        cols = st.columns(len(summary))
        for col, r in zip(cols, summary.itertuples()):
            col.metric(r.reasoning, fmt_pct(r.accuracy) if r.tasks_done else "—",
                       help=f"{r.tasks_solved} de {r.tasks_done} tarefas válidas")
            col.caption(f"{r.tasks_done} tarefas válidas, {fmt_int(r.output_tokens)} tokens de saída/teste "
                        f"({fmt_int(r.reasoning_tokens)} de raciocínio + {fmt_int(r.answer_tokens)} de resposta), "
                        f"{r.seconds:.0f} s/teste"
                        + (f", {r.tasks_excluded} fora por erro de API" if r.tasks_excluded else ""))
        left, right = st.columns(2)
        left.altair_chart(accuracy_chart(summary[summary["tasks_done"] > 0]), width="stretch")
        right.altair_chart(
            cost_chart(summary, "output_tokens", "Custo: tokens de saída por teste (média)",
                       "Tokens de saída (raciocínio + resposta)", fmt_int),
            width="stretch")
        st.caption("Com poucas tarefas por nível, os intervalos são largos: diferenças pequenas "
                   "entre níveis podem ser ruído. Todos os níveis usam as mesmas tarefas. "
                   "Em «minimal» a API não informa tokens de raciocínio, mas o modelo raciocina "
                   "no texto da resposta; por isso o custo soma raciocínio e resposta.")
        with st.expander("Tempo médio por teste"):
            st.altair_chart(
                cost_chart(summary, "seconds", "Tempo por teste (média)", "Segundos",
                           lambda v: f"{v:.0f} s"),
                width="stretch")

    elif view == "Por tarefa":
        st.subheader("Quais tarefas dependem de mais reasoning?")
        st.caption("Cada tarefa é classificada pelo padrão de acertos ao longo dos níveis. "
                   "Legenda das células: ✓ acertou · ✗ errou · ! falha de API (fora da conta) · "
                   "… em andamento. Em cada célula, os tokens de saída gastos na tarefa.")
        if len(levels_present) < 2:
            st.info("Este experimento tem só um nível de reasoning, então não há como comparar. "
                    "Rode dois ou mais níveis (ou escolha outro experimento) para ver o padrão por tarefa.")
        kinds = classify_tasks(tasks)
        first_seen = list(dict.fromkeys(ev["task_id"]))
        st.html(task_table_html(tasks, kinds, first_seen, levels_present))
    else:
        st.subheader("Registro ao vivo (mais recente primeiro)")
        log = ev.sort_values("time", ascending=False).assign(
            Resultado=lambda d: d["correct"].map({True: "✓ acertou", False: "✗ errou"}).where(
                d["error"].isna(), "! erro de API"),
            Hora=lambda d: d["time"].dt.strftime("%d/%m %H:%M:%S"),
        )[["Hora", "reasoning", "task_id", "test_index", "Resultado",
           "input_tokens", "reasoning_tokens", "answer_tokens", "seconds", "error"]]
        log.columns = ["Hora", "Reasoning", "Tarefa", "Teste", "Resultado",
                       "Tokens entrada", "Tokens raciocínio", "Tokens resposta", "Segundos", "Erro"]
        st.dataframe(log, width="stretch", hide_index=True, height=520)


def render_flashback() -> None:
    """Replay tab: pick a saved experiment and play it back test by test."""
    ev_all = load_events()
    if ev_all.empty:
        st.info("Ainda não há experimentos salvos para tocar.")
        return
    run = st.session_state.get("run")
    running_label = run["label"] if run and is_running() else None
    info = {}
    for label, e in ev_all.groupby("label"):
        lv = [l for l in LEVELS if l in set(e["reasoning"])]
        suffix = " (em andamento)" if label == running_label else ""
        info[label] = (e["time"].max(), f"{label} · {e['model'].iloc[0]}: {', '.join(lv)}, {len(e)} testes{suffix}")
    labels = sorted(info, key=lambda l: info[l][0], reverse=True)
    pick = st.selectbox("Experimento para tocar", labels, format_func=lambda l: info[l][1],
                        key="flash_label")
    ev = ev_all[ev_all["label"] == pick]
    if "predicted" not in ev.columns:
        st.warning("Este experimento foi gravado antes de guardarmos as respostas; "
                   "o flashback mostra só entrada, saída esperada e acerto/erro.")
    if pick == running_label:
        st.caption("Experimento em andamento: o flashback toca o que já foi gravado. "
                   "Troque de experimento e volte para atualizar.")
    render_player(ev, LEVELS, LEVEL_COLORS)


def render_compare() -> None:
    """Compare models that ran the same tasks (in different experiments) at the same levels."""
    st.subheader("Qual modelo resolve mais tarefas?")
    st.caption("Só entram na comparação as tarefas que todos os modelos escolhidos rodaram, nos mesmos "
               "níveis de reasoning, em experimentos diferentes. Tarefas com falha de API ficam de fora. "
               "Se o mesmo modelo rodou a mesma tarefa mais de uma vez, vale o resultado mais recente.")
    ev_all = load_events()
    res = all_task_results(ev_all) if not ev_all.empty else pd.DataFrame()
    if res.empty:
        st.info("Ainda não há resultados válidos para comparar.")
        return

    present = set(res["model"])
    models_avail = [m for m in GOOGLE_MODELS if m in present] + sorted(present - set(GOOGLE_MODELS))
    colors = model_colors(models_avail)
    cov = (res.pivot_table(index="model", columns="reasoning", values="task_id", aggfunc="count")
           .reindex(index=models_avail, columns=LEVELS).fillna(0).astype(int))
    cov.index.name = "Modelo"
    if len(models_avail) < 2:
        st.info(f"Só há resultados de um modelo ({models_avail[0]}). Para comparar, rode outro modelo "
                f"nas mesmas tarefas e nos mesmos níveis (barra lateral: Modelo + o mesmo arquivo de "
                f"tarefas), com outro rótulo.")
        st.dataframe(cov, width="content")
        return

    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Modelos", models_avail, default=models_avail[:2],
                            help="Com exatamente dois modelos aparece também o confronto tarefa a tarefa.")
    if len(chosen) < 2:
        st.info("Escolha pelo menos dois modelos.")
        return
    chosen = [m for m in models_avail if m in chosen]  # stable order, so colors and bars match
    options = common_levels(res, chosen)
    if not options:
        st.warning("Os modelos escolhidos não têm nenhum nível de reasoning em comum.")
        return
    levels = c2.multiselect("Níveis de reasoning", options, default=options)
    if not levels:
        return
    levels = [l for l in LEVELS if l in levels]
    sub, n_common = common_results(res, chosen, levels)
    if n_common == 0:
        own = {m: int(res[(res["model"] == m) & res["reasoning"].isin(levels)]["task_id"].nunique())
               for m in chosen}
        st.warning("Nenhuma tarefa foi concluída por todos os modelos em todos os níveis escolhidos. "
                   "Tarefas por modelo: " + ", ".join(f"{m}: {n}" for m, n in own.items()))
        return

    s = compare_summary(sub)
    # Generated headline: every number comes from the data.
    wins = {m: 0 for m in chosen}
    ties = 0
    for level in levels:
        row = s[s["reasoning"] == level]
        best = row["accuracy"].max()
        leaders = row[row["accuracy"] == best]["model"].tolist()
        if len(leaders) == 1:
            wins[leaders[0]] += 1
        else:
            ties += 1
    per_model = ", ".join(f"**{m}** em {wins[m]}" for m in chosen)
    st.markdown(f"Comparação em **{n_common} tarefas em comum** × {len(levels)} níveis. "
                f"Melhor acerto por nível: {per_model}" + (f"; empate em {ties}" if ties else "") + ".")
    if n_common < 10:
        st.caption(f"Com {n_common} tarefas, os intervalos de confiança são largos: uma diferença de "
                   f"1 ou 2 tarefas entre modelos pode ser acaso.")

    st.html(model_cards_html(sub, chosen, colors))

    left, right = st.columns(2)
    left.altair_chart(compare_bar_chart(
        s, chosen, colors, "accuracy", "Tarefas resolvidas (%)",
        "Acerto por nível (barra de erro: intervalo de 95%)", fmt_pct, with_ci=True), width="stretch")
    right.altair_chart(compare_bar_chart(
        s, chosen, colors, "out_tokens", "Tokens de saída por tarefa",
        "Custo por nível: tokens de saída por tarefa (média)", fmt_int), width="stretch")

    if len(chosen) == 2:
        a, b = chosen
        st.markdown("**Confronto tarefa a tarefa**")
        st.caption(f"Em cada nível, das {n_common} tarefas: quantas os dois acertaram, só um acertou "
                   f"ou nenhum acertou.")
        st.altair_chart(paired_chart(sub, a, b, levels, colors), width="stretch")

    with st.expander("Resultado de cada tarefa", expanded=True):
        st.caption("✓ acertou · ✗ errou · número = tokens de saída gastos na tarefa.")
        st.html(compare_table_html(sub, chosen, levels, colors))

    with st.expander("Cobertura: tarefas válidas por modelo e nível (todos os experimentos)"):
        st.dataframe(cov, width="content")


# Poll every 3 s only while the solver process is alive.
with tab_panel:
    st.fragment(render, run_every=3 if is_running() else None)()
with tab_flash:
    render_flashback()
with tab_compare:
    render_compare()
