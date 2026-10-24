"""InSite Claim — outage energy-loss assessment. Run: uv run streamlit run app/app.py

UI only: all calculations come from src/app_core.py and src/pipeline.py (unchanged).
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app_core as ac  # noqa: E402
from core import park_of  # noqa: E402
from pipeline import RECIPE  # noqa: E402

APP_NAME = "InSite Claim"
W_LGBM = round(100 * RECIPE.get("blend_weight_lgbm", 0.4))
W_LIN = 100 - W_LGBM
TRAVEL = bool(RECIPE.get("advect"))
st.set_page_config(page_title=APP_NAME, page_icon=str(ROOT / "app" / "icon.png"), layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ design tokens
BLUE, ORANGE, GREEN, INDIGO, PINK, GRAPHITE = "#0071e3", "#ff9f0a", "#34c759", "#5856d6", "#ff2d55", "#8e8e93"
SERIES = [BLUE, ORANGE, GREEN, INDIGO, PINK, GRAPHITE]
TEXT1, TEXT2, TEXT3, HAIR, GRID = "#1d1d1f", "#6e6e73", "#86868b", "#d2d2d7", "#e8e8ed"
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', 'Helvetica Neue', Arial, sans-serif"
Z = {"80": 1.2816, "90": 1.6449}

CSS = f"""
<style>
html, body, .stApp, button, input, textarea, select {{ font-family: {FONT}; -webkit-font-smoothing: antialiased; }}
.stApp {{ background: radial-gradient(1200px 420px at 50% -8%, rgba(0,113,227,.06), transparent 70%), linear-gradient(180deg,#fbfbfd 0,#f5f5f7 520px); }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, [data-testid="stSidebarCollapsedControl"], section[data-testid="stSidebar"] {{ display:none !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 0 !important; padding-bottom: 4rem; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {TEXT1}; }}
[data-testid="stCaptionContainer"] {{ color: {TEXT2} !important; }}

.st-key-navbar {{ position: sticky; top: 0; z-index: 999; background: rgba(251,251,253,.80); backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px); border-bottom: 1px solid {GRID}; padding: 8px 6px; margin-bottom: 18px; }}
.brand {{ display:flex; align-items:center; gap:8px; font-size:19px; font-weight:600; letter-spacing:-0.02em; color:{TEXT1}; white-space:nowrap; }}
.st-key-navbar [data-testid="stButtonGroup"] {{ margin-left: auto; }}

[data-testid="stButtonGroup"] {{ background: rgba(118,118,128,.12); border-radius: 10px; padding: 2px; width: fit-content; max-width: 100%; }}
[data-testid="stButtonGroup"] button {{ border: none !important; background: transparent !important; border-radius: 8px !important; color: {TEXT1} !important;
  min-height: 30px !important; padding: 0 14px !important; box-shadow: none !important; }}
[data-testid="stButtonGroup"] button p {{ font-size: 13px !important; font-weight: 500 !important; color: {TEXT1} !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {{ background: #fff !important; box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 3px 1px rgba(0,0,0,.04) !important; }}

.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {{ border-radius: 980px !important; font-weight: 500; min-height: 38px; box-shadow: none !important; }}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"], [data-testid="stPopover"] button {{ background: {GRID}; border-color: {GRID}; color: {TEXT1}; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPopover"] button:hover {{ filter: brightness(0.97); }}

[data-baseweb="select"] > div, [data-baseweb="input"], .stTextInput input, .stTextArea textarea, .stNumberInput input {{ border-radius: 12px !important; }}
[data-baseweb="tag"] {{ border-radius: 980px !important; background: #e8f2fd !important; }}
[data-baseweb="tag"] span {{ color: {BLUE} !important; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius: 14px; background: #fbfbfd; }}
.stTextArea textarea {{ font-size: 14px; line-height: 1.55; }}

div[class*="st-key-card"] {{ background: #fff; border-radius: 18px; padding: 22px 24px; box-shadow: 0 1px 2px rgba(0,0,0,.04); border: 1px solid rgba(0,0,0,.05); }}
.kpi {{ background:#fff; border-radius:18px; padding:18px 20px; border:1px solid rgba(0,0,0,.05); box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.kpi .l {{ font-size:12px; font-weight:500; color:{TEXT2}; }}
.kpi .v {{ font-size:32px; font-weight:600; letter-spacing:-0.025em; color:{TEXT1}; line-height:1.15; margin-top:6px; font-variant-numeric: tabular-nums; white-space:nowrap; }}
.kpi .v span {{ font-size:16px; font-weight:500; color:{TEXT2}; letter-spacing:0; margin-left:4px; }}
.kpi .c {{ font-size:12px; color:{TEXT3}; margin-top:6px; }}
.kgrid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:16px; margin: 8px 0 20px; }}
.eyebrow {{ font-size:14px; font-weight:600; color:#b25000; margin-bottom:8px; }}
.sect {{ font-size:28px; font-weight:600; letter-spacing:-0.015em; color:{TEXT1}; margin: 30px 0 2px; }}
.sub {{ font-size:15px; color:{TEXT2}; margin-bottom: 14px; }}
.ctitle {{ font-size:17px; font-weight:600; letter-spacing:-0.02em; color:{TEXT1}; }}
.csub {{ font-size:13px; color:{TEXT2}; margin: 2px 0 10px; line-height:1.5; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; border-radius:980px; padding:3px 10px; font-size:12px; font-weight:600; white-space:nowrap; }}
.badge:before {{ content:""; width:6px; height:6px; border-radius:50%; background: currentColor; }}
.b-red {{ background:#fdecee; color:#d70015; }} .b-green {{ background:#e9f6ec; color:#248a3d; }}
.b-gray {{ background:#f2f2f4; color:{TEXT2}; }} .b-blue {{ background:#e8f2fd; color:{BLUE}; }} .b-orange {{ background:#fff4e5; color:#b25000; }}
table.t {{ width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric: tabular-nums; border:none !important; margin:0; }}
table.t th, table.t td {{ border-left:none !important; border-right:none !important; border-top:none !important; background:transparent; }}
table.t th {{ text-align:left; font-size:12px; font-weight:600; color:{TEXT2}; padding:10px 12px; border-bottom:1px solid {GRID} !important; white-space:nowrap; }}
table.t td {{ padding:12px; border-bottom:1px solid {GRID} !important; color:{TEXT1}; vertical-align:top; line-height:1.45; }}
table.t tr:last-child td {{ border-bottom:none !important; }}
table.t td.n, table.t th.n {{ text-align:right; white-space:nowrap; }}
table.t tr.total td {{ font-weight:600; }}
table.t tbody tr:hover td {{ background:#fafafc; }}
.tcard {{ overflow-x:auto; }}
.step {{ display:flex; gap:16px; padding:16px 0; border-bottom:1px solid {GRID}; }}
.step:last-child {{ border-bottom:none; }}
.step .num {{ flex:0 0 28px; height:28px; border-radius:50%; background:#e8f2fd; color:{BLUE}; font-weight:600; font-size:14px; display:flex; align-items:center; justify-content:center; }}
.step b {{ display:block; font-size:16px; font-weight:600; color:{TEXT1}; margin-bottom:2px; letter-spacing:-0.01em; }}
.step span {{ font-size:14px; color:{TEXT2}; line-height:1.55; }}
.hero h1 {{ font-size:60px; line-height:1.04; font-weight:600; letter-spacing:-0.035em; margin:0 0 14px; padding:0; color:{TEXT1}; }}
.hero p {{ font-size:21px; line-height:1.42; color:{TEXT2}; letter-spacing:-0.01em; max-width:560px; margin:0; }}
.empty {{ text-align:center; padding:28px 10px; color:{TEXT2}; font-size:15px; }}
.empty b {{ display:block; font-size:17px; color:{TEXT1}; margin:10px 0 4px; }}
.facts {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin-top:6px; }}
.facts > div {{ background:#f5f5f7; border-radius:12px; padding:12px 14px; }}
.facts .l {{ font-size:12px; color:{TEXT2}; }} .facts .v {{ font-size:17px; font-weight:600; color:{TEXT1}; margin-top:2px; }}
.tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px; margin-top:4px; }}
.tile {{ background:#fff; border-radius:16px; padding:16px 18px; border:1px solid rgba(0,0,0,.05); }}
.tile .n {{ font-size:15px; font-weight:600; color:{TEXT1}; margin-bottom:2px; }} .tile .m {{ font-size:12px; color:{TEXT2}; margin:4px 0 10px; }}
@media (max-width: 760px) {{ .hero h1 {{ font-size:40px; }} .hero p {{ font-size:17px; }} .kpi .v {{ font-size:26px; }} }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.rotor {{ transform-box: fill-box; transform-origin: center; animation: spin 14s linear infinite; }}
.rotor.s2 {{ animation-duration: 18s; }} .rotor.s3 {{ animation-duration: 22s; }}
@media (prefers-reduced-motion: reduce) {{ .rotor {{ animation: none; }} }}
</style>
"""


def turbine_svg(x, y, s, cls, opacity):
    blades = "".join(f'<path d="M0,0 L-1.7,-3 L0,-38 L1.7,-3 Z" transform="rotate({a})" fill="#fff" stroke="{TEXT1}" stroke-width="1.2" stroke-linejoin="round"/>'
                     for a in (0, 120, 240))
    return (f'<g transform="translate({x},{y}) scale({s})" opacity="{opacity}">'
            f'<path d="M-2.4,2 L-1.1,100 L1.1,100 L2.4,2 Z" fill="none" stroke="{TEXT1}" stroke-width="1.2"/>'
            f'<g class="rotor {cls}"><circle r="38" fill="none" stroke="none"/>{blades}<circle r="3.4" fill="#fff" stroke="{TEXT1}" stroke-width="1.3"/></g></g>')


HERO_ART = ('<svg viewBox="0 0 320 230" width="100%" style="max-width:380px" role="img" aria-label="Wind turbines">'
            + turbine_svg(60, 120, 0.6, "s3", 0.3) + turbine_svg(125, 98, 0.82, "s2", 0.5) + turbine_svg(225, 64, 1.18, "", 1)
            + f'<line x1="0" y1="182" x2="320" y2="182" stroke="{HAIR}" stroke-width="1"/></svg>')

LOGO = (f'<svg width="24" height="24" viewBox="-20 -22 40 46" aria-hidden="true"><path d="M-1.4,0 L-0.7,22 L0.7,22 L1.4,0 Z" fill="{TEXT1}"/>'
        + "".join(f'<path d="M0,0 L-1.5,-2 L0,-19 L1.5,-2 Z" transform="rotate({a})" fill="{BLUE}"/>' for a in (0, 120, 240))
        + f'<circle r="2.4" fill="{TEXT1}"/></svg>')


# ------------------------------------------------------------------ helpers
def esc(x):
    return html.escape(str(x))


def kpis(cards):
    st.markdown("<div class='kgrid'>" + "".join(
        f"<div class='kpi'><div class='l'>{esc(a)}</div><div class='v'>{b}</div><div class='c'>{esc(c)}</div></div>" for a, b, c in cards)
        + "</div>", unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f"<div class='sect'>{esc(title)}</div>" + (f"<div class='sub'>{esc(sub)}</div>" if sub else "<div style='height:10px'></div>"),
                unsafe_allow_html=True)


def card_head(title, sub=None):
    st.markdown(f"<div class='ctitle'>{esc(title)}</div>" + (f"<div class='csub'>{esc(sub)}</div>" if sub else "<div style='height:8px'></div>"),
                unsafe_allow_html=True)


def badge(text, color):
    return f"<span class='badge b-{color}'>{esc(text)}</span>"


def table_html(df, numeric=frozenset(), total_last=False, raw=frozenset()):
    head = "".join(f"<th class='{'n' if c in numeric else ''}'>{esc(c)}</th>" for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        cls = " class='total'" if total_last and i == len(df) - 1 else ""
        tds = "".join(f"<td class='{'n' if c in numeric else ''}'>{r[c] if c in raw else esc(r[c])}</td>" for c in df.columns)
        rows.append(f"<tr{cls}>{tds}</tr>")
    return f"<div class='tcard'><table class='t'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=48, r=16, t=36 if legend else 12, b=36), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=TEXT2), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TEXT2)),
        hoverlabel=dict(bgcolor="#fff", bordercolor=HAIR, font=dict(family=FONT, size=13, color=TEXT1)),
        bargap=0.35, barcornerradius=6,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", tickfont=dict(color=TEXT3))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, ticks="", tickfont=dict(color=TEXT3), title_font=dict(color=TEXT2, size=12))
    return fig


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def _seg_keep(key):
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get("_last_" + key)


def segmented(label, options, key, default=None, fmt=None):
    """Single-choice segmented control that never ends up empty (clicking the active segment keeps it)."""
    last = st.session_state.get("_last_" + key, default if default is not None else options[0])
    if last not in options:
        last = default if default in options else options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = last
    kw = {"format_func": fmt} if fmt else {}
    v = st.segmented_control(label, options, key=key, on_change=_seg_keep, args=(key,), label_visibility="collapsed", **kw)
    v = v if v in options else last
    st.session_state["_last_" + key] = v
    return v


# ------------------------------------------------------------------ persistent settings (survive page switches)
DEFAULTS = {"source": "Example dataset", "window_days": RECIPE["training_window_days"], "level": "90", "density": False,
            "min_gap_h": 6.0, "outage": None, "targets": None, "fp": None, "uploads_w": (), "uploads_c": (), "park": None}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def keep(k):
    v = st.session_state.get("_" + k)
    if v is not None and v != []:
        st.session_state[k] = v


def keep_uploads(kind):
    files = st.session_state.get("_up_" + kind) or []
    st.session_state["uploads_" + kind] = tuple((f.name, f.getvalue()) for f in files)


# ------------------------------------------------------------------ data + analysis (shared by all pages)
@st.cache_data(show_spinner=False)
def load(files, curves):
    long = ac.parse_wind_files(list(files))
    return ac.to_wide(long), ac.test_flags(long), ac.parse_power_curves(list(curves))


@st.cache_data(show_spinner=False)
def demo_files():
    raw = ROOT / "data" / "raw"
    return (tuple((p.name, p.read_bytes()) for p in sorted(raw.glob("*_data.xlsx"))),
            tuple((p.name, p.read_bytes()) for p in sorted(raw.glob("*_power_curve.xlsx"))))


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(_wide, _curves, key, targets, start, end, window_days, level, density, test_window):
    return ac.run_analysis(_wide, list(targets), (start, end), _curves, window_days, level, density, test_window)


@st.cache_data(show_spinner=False)
def portfolio():
    try:
        return json.loads((ROOT / "app" / "portfolio.json").read_text())
    except Exception:  # noqa: BLE001
        return []


CTX: dict = {"ok": False}


def build_context():
    if st.session_state.source == "Example dataset":
        wf, cf = demo_files()
    else:
        wf, cf = st.session_state.uploads_w, st.session_state.uploads_c
    if not wf:
        CTX["msg"] = ("No data yet", "Upload wind data for the affected park and its neighbours, plus power curves.")
        return
    try:
        wide, flags, curves = load(wf, cf)
    except Exception as e:  # noqa: BLE001
        CTX["msg"] = ("Couldn't read the files", str(e))
        return
    fp = hashlib.md5(b"".join(n.encode() + d for n, d in wf + cf)).hexdigest()
    det = ac.detect_outage(wide, min_hours=st.session_state.min_gap_h)
    turbines = list(wide["wind_speed"].columns)
    CTX.update(wide=wide, flags=flags, curves=curves, det=det, turbines=turbines)
    if st.session_state.fp != fp:  # new data: reset outage + targets to the detected ones
        st.session_state.fp, st.session_state.outage, st.session_state.targets = fp, None, None
    if not det:
        CTX["msg"] = ("No outage found", "No data gap is long enough to count as an outage. Lower the gap threshold in Settings.")
        return
    if not curves:
        CTX["msg"] = ("Power curves missing", "Upload the power curves of the affected turbines.")
        return
    targets = [t for t in (st.session_state.targets or det["targets"]) if t in turbines] or det["targets"]
    start, end = st.session_state.outage or (det["start"], det["end"])
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        start, end = det["start"], det["end"]
    test_window = ac.detect_test_window(wide, flags, targets, end)
    key = json.dumps([fp, str(start), str(end), targets, st.session_state.window_days, st.session_state.level,
                      st.session_state.density, str(test_window)])
    try:
        with st.spinner("Rebuilding the missing wind from neighbouring parks…"):
            results = analyse(wide, curves, key, tuple(targets), start, end, st.session_state.window_days,
                              st.session_state.level, st.session_state.density, test_window)
    except Exception as e:  # noqa: BLE001
        CTX["msg"] = ("The analysis couldn't run for this selection", f"{e}. Try a longer training window or a different outage window.")
        return
    total, lo, hi = ac.park_range(results, st.session_state.level)
    CTX.update(ok=True, results=results, targets=targets, start=start, end=end, test_window=test_window,
               total=total, lo=lo, hi=hi, hours=(end - start).total_seconds() / 3600 + 1 / 6,
               color={r["target"]: SERIES[i % len(SERIES)] for i, r in enumerate(results)})


def empty_state(title, text, key):
    with st.container(key=f"card_empty_{key}"):
        st.markdown(f"<div class='empty'><svg width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='{TEXT3}' stroke-width='1.5' "
                    "stroke-linecap='round'><path d='M3 8h10a3 3 0 1 0-3-3'/><path d='M3 12h15a3 3 0 1 1-3 3'/><path d='M3 16h7'/></svg>"
                    f"<b>{esc(title)}</b>{esc(text)}</div>", unsafe_allow_html=True)


def need_results(where):
    if CTX.get("ok"):
        return True
    t, m = CTX.get("msg", ("No data yet", "Choose a data source on the Overview page."))
    empty_state(t, m, where)
    _, mid, _ = st.columns([2, 1, 2])
    if mid.button("Go to Overview", key=f"go_overview_{where}", width="stretch"):
        st.switch_page(PAGES["Overview"])
    return False


# ------------------------------------------------------------------ pages
def page_overview():
    c1, c2 = st.columns([1.4, 1], vertical_alignment="center")
    with c1:
        st.markdown("<div class='hero'><div class='eyebrow'>Outage energy assessment</div><h1>InSite Claim</h1>"
                    "<p>Know what a wind park lost while it was offline. InSite Claim rebuilds the missing wind from neighbouring parks "
                    "and turns it into a defensible kWh figure for your insurance claim.</p></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 1.6])
        if b1.button("Open analysis", type="primary", key="hero_open", width="stretch"):
            st.switch_page(PAGES["Analysis"])
        if b2.button("Claim report", key="hero_report", width="stretch"):
            st.switch_page(PAGES["Claim report"])
    with c2:
        st.markdown(f"<div style='display:flex;justify-content:center;padding-top:10px'>{HERO_ART}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if CTX.get("ok"):
        kpis([
            ("Connected parks", f"{len({park_of(t) for t in CTX['turbines']})}", f"{len(CTX['turbines'])} turbines with live data"),
            ("Open outage", f"{CTX['hours']:.0f}<span>h</span>", f"{CTX['start']:%d %b %H:%M} → {CTX['end'] + pd.Timedelta(minutes=10):%d %b %H:%M}"),
            ("Estimated loss", f"{CTX['total']:,.0f}<span>kWh</span>",
             f"{len(CTX['results'])} turbines · {st.session_state.level} % range {CTX['lo']/1000:,.0f}–{CTX['hi']/1000:,.0f} MWh"),
        ])

    # ---------------- portfolio
    section("Portfolio", "Parks with live data can be analysed. The rest of the BayWa r.e. portfolio is listed for reference.")
    farms = portfolio()
    data_parks = []
    if CTX.get("wide") is not None:
        tparks = {park_of(t) for t in (CTX.get("targets") or (CTX.get("det") or {}).get("targets", []))}
        for p in sorted({park_of(t) for t in CTX["turbines"]}, key=lambda p: (p not in tparks, p)):
            members = [t for t in CTX["turbines"] if park_of(t) == p]
            data_parks.append({"label": f"Wind park {p}", "park": p, "members": members,
                               "role": "Affected park" if p in tparks else "Reference park"})
    options = [d["label"] for d in data_parks] + [f"{f['name']} · {f['country']}" for f in farms]
    if not options:
        empty_state("No parks yet", "Choose a data source below to add parks.", "portfolio")
    else:
        if st.session_state.park not in options:
            st.session_state.park = options[0]
        with st.container(key="card_portfolio"):
            cc1, cc2 = st.columns([1.3, 2], vertical_alignment="bottom")
            cc1.selectbox("Wind park", options, index=options.index(st.session_state.park), key="_park", on_change=keep, args=("park",))
            sel = st.session_state.park
            dp = next((d for d in data_parks if d["label"] == sel), None)
            if dp:
                cc2.markdown(f"<div style='padding-bottom:12px'>{badge(dp['role'], 'orange' if dp['role'] == 'Affected park' else 'blue')} &nbsp;"
                             f"<span style='color:{TEXT2};font-size:13px'>{len(dp['members'])} turbines · live data</span></div>", unsafe_allow_html=True)
                wide = CTX["wide"]
                rows = []
                for t in dp["members"]:
                    s = wide[("wind_speed", t)]
                    if CTX.get("ok") and t in CTX["targets"]:
                        status, note = badge("Communication lost", "red"), f"Outage {CTX['start']:%d %b} – {CTX['end']:%d %b}"
                    elif dp["role"] == "Affected park":
                        status, note = badge("Reporting", "green"), "Not in this analysis"
                    else:
                        status, note = badge("Reporting", "green"), "Reference candidate"
                    rows.append({"Turbine": t, "Status": status, "Note": note, "Data coverage": f"{100 * s.notna().mean():.1f} %",
                                 "Mean wind": f"{s.mean():.1f} m/s",
                                 "Last value": f"{s.last_valid_index():%d %b %Y, %H:%M}" if s.notna().any() else "—"})
                st.markdown(table_html(pd.DataFrame(rows), numeric={"Data coverage", "Mean wind"}, raw={"Status"}), unsafe_allow_html=True)
            else:
                f = next(f for f in farms if f"{f['name']} · {f['country']}" == sel)
                stat = "In development" if f.get("status") == "development" else "No data connected"
                cc2.markdown(f"<div style='padding-bottom:12px'>{badge(stat, 'gray')} &nbsp;"
                             f"<span style='color:{TEXT2};font-size:13px'>BayWa r.e. portfolio</span></div>", unsafe_allow_html=True)
                facts = [("Country", f.get("country") or "—"),
                         ("Capacity", f"{f['capacity_mw']:g} MW" if f.get("capacity_mw") else "—"),
                         ("Turbines", f"{f['turbines']}" if f.get("turbines") else "—"),
                         ("BayWa r.e. role", f.get("relationship") or "—")]
                st.markdown("<div class='facts'>" + "".join(f"<div><div class='l'>{esc(a)}</div><div class='v'>{esc(b)}</div></div>" for a, b in facts)
                            + "</div>", unsafe_allow_html=True)
                src = f"&nbsp;<a href='{esc(f['evidence_url'])}' target='_blank'>Source ›</a>" if f.get("evidence_url") else ""
                why = ("This park is still in development." if f.get("status") == "development"
                       else "No turbine data is connected for this park yet, so it can't be analysed. Upload its data below to include it.")
                st.markdown(f"<div style='margin-top:14px;font-size:13px;color:{TEXT2}'>{why}{src}</div>", unsafe_allow_html=True)
        if farms:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='tiles'>" + "".join(
                f"<div class='tile'><div class='n'>{esc(f['name'])}</div><div class='m'>{esc(f['country'])}"
                + (f" · {f['capacity_mw']:g} MW" if f.get("capacity_mw") else "") + f" · {esc(f.get('relationship', ''))}</div>"
                + badge("In development" if f.get("status") == "development" else "No data connected", "gray") + "</div>"
                for f in farms) + "</div>", unsafe_allow_html=True)

    # ---------------- data source
    section("Data source", "One file per turbine (CSV or Excel), plus the power curves of the affected turbines.")
    with st.container(key="card_data"):
        src = segmented("Source", ["Example dataset", "Upload files"], "_source", default=st.session_state.source)
        if src != st.session_state.source:
            st.session_state.source = src
            st.rerun()
        if st.session_state.source == "Upload files":
            u1, u2 = st.columns(2)
            u1.file_uploader("Wind data", type=["csv", "xlsx"], accept_multiple_files=True, key="_up_w", on_change=keep_uploads, args=("w",),
                             help="Columns: turbine_name, timestamp, wind_speed. Optional: wind_direction, ambient_temperature, air_pressure, wind_speed_stddev, test_set.")
            u2.file_uploader("Power curves", type=["csv", "xlsx"], accept_multiple_files=True, key="_up_c", on_change=keep_uploads, args=("c",),
                             help="Columns: turbine_name, wind_speed (m/s), power (kW).")
            if st.session_state.uploads_w:
                st.caption(f"{len(st.session_state.uploads_w)} wind files and {len(st.session_state.uploads_c)} power curves loaded.")
            else:
                st.caption("Turbine names start with a park letter, e.g. T1, H3, O2.")
        else:
            st.caption("Hackathon dataset · wind park T (affected), H and O · July – September 2026 · 10-minute data.")
        if CTX.get("msg") and not CTX.get("ok"):
            st.markdown(f"<div style='margin-top:8px'>{badge(CTX['msg'][0], 'orange')} <span style='color:{TEXT2};font-size:13px'>"
                        f"{esc(CTX['msg'][1])}</span></div>", unsafe_allow_html=True)


def settings_popover():
    with st.popover("Settings", width="content"):
        st.markdown(f"<div class='ctitle'>Settings</div><div class='csub'>Changes apply to every page.</div>", unsafe_allow_html=True)
        st.slider("Training days before the outage", 7, 60, int(st.session_state.window_days), key="_window_days", on_change=keep, args=("window_days",),
                  help="The model learns the link between the parks only from this period.")
        st.caption("Confidence of the range")
        lv = segmented("Confidence of the range", ["80", "90"], "_level", default=st.session_state.level, fmt=lambda x: f"{x} %")
        if lv != st.session_state.level:
            st.session_state.level = lv
            st.rerun()
        st.toggle("Air density correction", value=bool(st.session_state.density), key="_density", on_change=keep, args=("density",),
                  help="Adjusts the power curve for air density from temperature and pressure (IEC 61400-12). Changes results by about 2 %.")
        st.number_input("Gaps longer than this count as an outage (hours)", 1.0, 48.0, float(st.session_state.min_gap_h), 1.0,
                        key="_min_gap_h", on_change=keep, args=("min_gap_h",))


def page_analysis():
    h1, h2 = st.columns([5, 1], vertical_alignment="bottom")
    with h1:
        section("Analysis", "Rebuilt wind and lost energy for the outage.")
    with h2:
        settings_popover()
    if not need_results("analysis"):
        return
    wide, det, results, color = CTX["wide"], CTX["det"], CTX["results"], CTX["color"]
    with st.container(key="card_outage"):
        card_head("Outage window", "Detected automatically from the missing data. Drag to adjust.")
        o1, o2 = st.columns([2, 1])
        with o1:
            t0, t1 = wide.index.min().to_pydatetime(), wide.index.max().to_pydatetime()
            st.slider("Outage window", min_value=t0, max_value=t1, value=(CTX["start"].to_pydatetime(), CTX["end"].to_pydatetime()),
                      step=pd.Timedelta(minutes=10).to_pytimedelta(), format="DD MMM YYYY, HH:mm", key="_outage", on_change=keep, args=("outage",),
                      label_visibility="collapsed")
            tr_from = CTX["start"] - pd.Timedelta(days=int(st.session_state.window_days))
            moved = (CTX["start"], CTX["end"]) != (det["start"], det["end"])
            st.caption(f"Training data {tr_from:%d %b} → {CTX['start']:%d %b %Y}"
                       + (f" · detected {det['start']:%d %b %H:%M} → {det['end']:%d %b %H:%M}" if moved else ""))
        with o2:
            st.multiselect("Affected turbines", CTX["turbines"], default=CTX["targets"], key="_targets", on_change=keep, args=("targets",),
                           label_visibility="collapsed", placeholder="Choose affected turbines")
            refs = [t for t in CTX["turbines"] if park_of(t) not in {park_of(x) for x in CTX["targets"]}]
            st.caption(f"Reference pool: {', '.join(refs)}")
    mean_wind = float(np.mean(np.concatenate([r["wind_pred"] for r in results])))
    kpis([
        ("Estimated energy loss", f"{CTX['total']:,.0f}<span>kWh</span>", f"{CTX['total']/1000:,.1f} MWh across {len(results)} turbines"),
        (f"{st.session_state.level} % range", f"{CTX['lo']/1000:,.0f}–{CTX['hi']/1000:,.0f}<span>MWh</span>", "Calibrated on 84 simulated outages"),
        ("Outage duration", f"{CTX['hours']:.0f}<span>h</span>", f"{CTX['start']:%d %b %H:%M} → {CTX['end'] + pd.Timedelta(minutes=10):%d %b %H:%M}"),
        ("Mean rebuilt wind", f"{mean_wind:.1f}<span>m/s</span>", "References " + ", ".join(sorted({x for r in results for x in r["refs"]}))),
    ])

    with st.container(key="card_wind"):
        a1, a2 = st.columns([3, 2], vertical_alignment="center")
        with a1:
            card_head("Wind speed", "Measured before and after, rebuilt during the outage.")
        with a2:
            sel = segmented("Turbine", [r["target"] for r in results], "wind_sel")
        r = next(x for x in results if x["target"] == sel)
        show_refs = st.toggle("Show reference turbines", value=False, key="show_refs")
        pad = pd.Timedelta(days=2)
        view = wide.loc[CTX["start"] - pad: CTX["end"] + pad]
        fig = go.Figure()
        fig.add_vrect(x0=CTX["start"], x1=CTX["end"], fillcolor="rgba(255,159,10,.10)", line_width=0, layer="below",
                      annotation_text="Outage", annotation_position="top left", annotation_font=dict(size=12, color="#b25000"))
        if show_refs:
            for ref in r["refs"][:3]:
                fig.add_scatter(x=view.index, y=view[("wind_speed", ref)], name=f"{ref} reference", line=dict(color=GRAPHITE, width=1, dash="dot"), opacity=0.8)
        fig.add_scatter(x=view.index, y=view[("wind_speed", sel)], name="Measured", line=dict(color=TEXT1, width=1.5))
        band = Z[st.session_state.level] * r["wobble_sd"]
        ts = r["timestamps"]
        fig.add_scatter(x=np.concatenate([ts, ts[::-1]]), y=np.concatenate([r["wind_pred"] + band, np.clip(r["wind_pred"] - band, 0, None)[::-1]]),
                        fill="toself", fillcolor="rgba(0,113,227,.12)", line=dict(width=0), name=f"{st.session_state.level} % band", hoverinfo="skip")
        fig.add_scatter(x=ts, y=r["wind_pred"], name="Rebuilt", line=dict(color=BLUE, width=2))
        fig.update_yaxes(title_text="m/s")
        chart(style(fig, 400), key="wind_chart")
        st.caption(f"The band shows where a single 10-minute value typically lands (±{band:.1f} m/s), learned from the training days.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    e1, e2 = st.columns(2, gap="medium")
    with e1, st.container(key="card_eturb"):
        card_head("Lost energy per turbine", f"Whiskers show the {st.session_state.level} % range.")
        f = go.Figure()
        for r in results:
            a, b = r["range_kwh"]
            f.add_bar(x=[r["target"]], y=[r["energy_kwh"] / 1000], marker_color=color[r["target"]], name=r["target"],
                      error_y=dict(type="data", symmetric=False, array=[(b - r["energy_kwh"]) / 1000], arrayminus=[(r["energy_kwh"] - a) / 1000],
                                   color=TEXT2, thickness=1.2, width=6),
                      hovertemplate="%{x}: %{y:,.1f} MWh<extra></extra>")
        f.update_yaxes(title_text="MWh")
        f.update_layout(hovermode="closest")
        chart(style(f, 320, legend=False), key="eturb_chart")
    with e2, st.container(key="card_ecum"):
        card_head("Cumulative loss", "How the loss built up during the outage.")
        f = go.Figure()
        cum_total = None
        for r in results:
            cum = np.cumsum(r["power_kw"]) / 6 / 1000
            cum_total = cum if cum_total is None else cum_total + cum
            f.add_scatter(x=r["timestamps"], y=cum, name=r["target"], line=dict(color=color[r["target"]], width=2))
        if len(results) > 1:
            f.add_scatter(x=results[0]["timestamps"], y=cum_total, name="Total", line=dict(color=TEXT1, width=2, dash="dash"))
        f.update_yaxes(title_text="MWh")
        chart(style(f, 320), key="ecum_chart")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(key="card_bands"):
        card_head("Where the loss comes from", "Lost energy by rebuilt wind speed. Below 3 m/s a turbine makes no power; above 11.5 m/s it runs at full power.")
        bt = ac.band_table(results)
        f = go.Figure()
        for r in results:
            d = bt[bt.turbine == r["target"]]
            f.add_bar(x=d.band.astype(str), y=d.kwh / 1000, name=r["target"], marker_color=color[r["target"]],
                      hovertemplate="%{x}: %{y:,.1f} MWh<extra>" + r["target"] + "</extra>")
        f.update_layout(barmode="stack", hovermode="closest")
        f.update_yaxes(title_text="MWh")
        chart(style(f, 300), key="bands_chart")
        agg = bt.groupby("band", observed=False).agg(hours=("hours", "sum"), kwh=("kwh", "sum")).reset_index()
        n = len(results)
        tbl = pd.DataFrame({"Wind band": agg.band.astype(str), "Hours per turbine": (agg.hours / n).map(lambda x: f"{x:.1f}"),
                            "Lost energy": agg.kwh.map(lambda x: f"{x:,.0f} kWh"),
                            "Share of loss": (100 * agg.kwh / max(agg.kwh.sum(), 1e-9)).map(lambda x: f"{x:.1f} %")})
        st.markdown(table_html(tbl, numeric={"Hours per turbine", "Lost energy", "Share of loss"}), unsafe_allow_html=True)
        st.caption(f"Hours are averaged over the {n} affected turbine{'s' if n > 1 else ''}, so they add up to the {CTX['hours']:.0f}-hour outage.")


def page_accuracy():
    section("Accuracy", "How close the method gets when the truth is known.")
    if not need_results("accuracy"):
        return
    results, color = CTX["results"], CTX["color"]
    tw = CTX["test_window"]
    with st.container(key="card_test"):
        if tw and all("test_metrics" in r for r in results):
            card_head(f"Test period · {tw[0]:%d %b} → {tw[1]:%d %b %Y}",
                      "Measured data exists here, so the same model predicts it and we compare. These are the judging metrics.")
            rows, tt, tp = [], 0.0, 0.0
            for r in results:
                m = r["test_metrics"]
                tt += m["energy_true_kwh"]
                tp += m["energy_pred_kwh"]
                rows.append([r["target"], m["wind_rmse"], m["wind_mae"], m["power_rmse"], m["power_mae"], m["energy_true_kwh"], m["energy_pred_kwh"], m["energy_dev_pct"]])
            M = pd.DataFrame(rows, columns=["Turbine", "wr", "wm", "pr", "pm", "et", "ep", "dev"])
            M.loc[len(M)] = ["All", M.wr.mean(), M.wm.mean(), M.pr.mean(), M.pm.mean(), tt, tp, 100 * (tp - tt) / tt if tt else float("nan")]
            show = pd.DataFrame({"Turbine": M.Turbine, "Wind RMSE": M.wr.map(lambda x: f"{x:.2f} m/s"), "Wind MAE": M.wm.map(lambda x: f"{x:.2f} m/s"),
                                 "Power RMSE": M.pr.map(lambda x: f"{x:,.0f} kW"), "Power MAE": M.pm.map(lambda x: f"{x:,.0f} kW"),
                                 "Energy measured": M.et.map(lambda x: f"{x:,.0f} kWh"), "Energy estimated": M.ep.map(lambda x: f"{x:,.0f} kWh"),
                                 "Deviation": M.dev.map(lambda x: f"{x:+.1f} %")})
            st.markdown(table_html(show, numeric=set(show.columns) - {"Turbine"}, total_last=True), unsafe_allow_html=True)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            sel = segmented("Turbine", [r["target"] for r in results], "acc_sel")
            r = next(x for x in results if x["target"] == sel)
            x = r["extra"]["test"]
            f = go.Figure()
            f.add_scatter(x=x["timestamps"], y=x["wind_true"], name="Measured", line=dict(color=TEXT1, width=1.5))
            f.add_scatter(x=x["timestamps"], y=x["wind_pred"], name="Estimated", line=dict(color=color[sel], width=2))
            f.update_yaxes(title_text="m/s")
            chart(style(f, 340), key="acc_chart")
        else:
            st.markdown("<div class='empty'><b>No test period</b>There is no measured data after the outage to compare against.</div>", unsafe_allow_html=True)

    section("Back-test inside the training days")
    with st.container(key="card_inner"):
        card_head("Rehearsal on the days before the outage",
                  "The training days are cut into 3-day pieces. Each piece is hidden in turn, the model is trained on the rest and predicts it. "
                  "This measures how much real wind wobbles around the estimate, and it works for any upload. Energy here is before the energy correction.")
        inner = pd.DataFrame([{"Turbine": r["target"], "Wind RMSE": f"{r['inner_metrics']['wind_rmse']:.2f} m/s",
                               "Wind MAE": f"{r['inner_metrics']['wind_mae']:.2f} m/s", "Power RMSE": f"{r['inner_metrics']['power_rmse']:,.0f} kW",
                               "Power MAE": f"{r['inner_metrics']['power_mae']:,.0f} kW", "Energy deviation": f"{r['inner_metrics']['energy_dev_pct']:+.1f} %",
                               "Wobble": f"±{r['wobble_sd']:.2f} m/s"} for r in results])
        st.markdown(table_html(inner, numeric=set(inner.columns) - {"Turbine"}), unsafe_allow_html=True)

    section("How the method was chosen")
    with st.container(key="card_val"):
        card_head("Validation on 84 simulated outages, July – August", "Every 3-day window was hidden and rebuilt using only the 30 days before it.")
        val = pd.DataFrame([
            ["<b>Simple baseline</b>", "Linear regression on the current wind speed of all seven reference turbines. Wind put straight through the power curve.",
             "677 kW", "1.19 m/s", "−13.7 %"],
            ["<b>Without energy correction</b>", "Top five references by correlation. Inputs: their wind now, each park's wind 10, 30 and 60 minutes earlier and later, "
             "each park's wind at the travel-time lag learned for the current wind direction, wind direction, turbulence and agreement between references. "
             f"Trained on labels smoothed over 90 minutes. {W_LIN} % linear model, {W_LGBM} % gradient-boosted trees.", "625 kW", "1.11 m/s", "−11.4 %"],
            ["<b>InSite Claim</b>", "Same wind, but the power curve is averaged over the typical wobble of real wind around the estimate, about ±1.1 m/s.",
             "615 kW", "1.11 m/s", "−2.3 %"]], columns=["Setup", "What it does", "Power RMSE", "Wind RMSE", "Energy deviation"])
        st.markdown(table_html(val, numeric={"Power RMSE", "Wind RMSE", "Energy deviation"}, raw={"Setup"}), unsafe_allow_html=True)
        st.caption("Wind RMSE is the same in the last two rows because only the step from wind to energy differs.")


def page_method():
    section("Method", "Five steps from missing data to a claim figure.")
    if not need_results("method"):
        return
    results, color, wide = CTX["results"], CTX["color"], CTX["wide"]
    lv = st.session_state.level
    steps = [
        ("Pick reference turbines automatically", f"For each affected turbine, every turbine in another park is ranked by how closely its wind followed the "
         f"affected turbine over the {int(st.session_state.window_days)} training days (correlation). The top five are used."),
        ("Learn the link between the parks", "Two models learn the affected turbine's wind from the references' wind, the wind 10, 30 and 60 minutes "
         "earlier and later (wind travels between parks), wind direction, turbulence and how much the references agree. "
         + ("Wind needs time to travel between parks, and how long depends on the wind direction, so for each direction the travel-time "
            "lag that fits best is learned from the training days. " if TRAVEL else "")
         + f"A linear model ({W_LIN} %) and gradient-boosted trees ({W_LGBM} %) are averaged."),
        ("Rebuild the missing wind", "The reference parks kept measuring during the outage. Their data goes into the model, which returns the affected "
         "turbine's wind for every 10 minutes."),
        ("Turn wind into energy", "Real wind wobbles around any estimate and the power curve is steep, so the curve is averaged over the typical "
         "wobble learned from the training days instead of read at a single point. Without this, losses come out about 10 % too low."),
        ("Put a range on it", f"Back-testing on 84 simulated outages showed how far a 3-day estimate can be off: "
         f"±{100 * RECIPE['energy']['range_turbine_rel'][lv]:.0f} % per turbine and ±{100 * RECIPE['energy']['range_park_rel'][lv]:.0f} % for a whole park "
         f"at {lv} % confidence."),
    ]
    with st.container(key="card_steps"):
        st.markdown("".join(f"<div class='step'><div class='num'>{i}</div><div><b>{esc(a)}</b><span>{esc(b)}</span></div></div>"
                            for i, (a, b) in enumerate(steps, 1)), unsafe_allow_html=True)

    section("Reference selection")
    sel = segmented("Turbine", [r["target"] for r in results], "m_sel")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    r = next(x for x in results if x["target"] == sel)
    rk = r["ranking"].copy()
    h = 60 + 36 * len(rk)
    m1, m2 = st.columns(2, gap="medium")
    with m1, st.container(key="card_rank"):
        card_head("Correlation over the training days", "Blue bars are used. Turbines in the affected park are never used.")
        used = rk.reference.isin(r["refs"])
        f = go.Figure(go.Bar(x=rk["corr"], y=rk.reference, orientation="h", marker_color=[BLUE if u else HAIR for u in used],
                             text=rk["corr"].map(lambda v: f"{v:.2f}"), textposition="outside", textfont=dict(color=TEXT2),
                             hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        f.update_layout(hovermode="closest", bargap=0.3)
        f.update_xaxes(range=[0, 1.08], showgrid=True, gridcolor=GRID)
        f.update_yaxes(autorange="reversed", showgrid=False)
        chart(style(f, h, legend=False), key="rank_chart")
    with m2, st.container(key="card_scatter"):
        best = r["refs"][0]
        d = wide["wind_speed"].iloc[r["train_idx"]][[best, sel]].dropna()
        card_head(f"{sel} against {best}", "Every 10-minute pair from the training days.")
        f = go.Figure(go.Scattergl(x=d[best], y=d[sel], mode="markers", marker=dict(size=4, color=color[sel], opacity=0.3),
                                   hovertemplate=f"{best} %{{x:.1f}} · {sel} %{{y:.1f}} m/s<extra></extra>"))
        mx = float(max(d.max().max(), 1))
        f.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=GRAPHITE, dash="dash", width=1), hoverinfo="skip")
        f.update_layout(hovermode="closest")
        f.update_xaxes(title_text=f"{best} (m/s)", title_font=dict(color=TEXT2, size=12))
        f.update_yaxes(title_text=f"{sel} (m/s)")
        chart(style(f, h, legend=False), key="scatter_chart")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(key="card_heat"):
        card_head("Affected turbines against reference turbines",
                  "Correlation over the training days. The higher the value, the better that turbine works as a reference.")
        tgts = [x["target"] for x in results]
        refs_all = [t for t in CTX["turbines"] if park_of(t) not in {park_of(x) for x in tgts}]
        C = wide["wind_speed"].iloc[r["train_idx"]][tgts + refs_all].corr().loc[tgts, refs_all]
        f = go.Figure(go.Heatmap(z=C.values, x=list(C.columns), y=list(C.index), zmin=0.5, zmax=0.85, text=C.round(2).values, texttemplate="%{text}",
                                 colorscale=[[0, "#f5f5f7"], [0.5, "#9ec9f5"], [1, BLUE]], showscale=False, xgap=3, ygap=3,
                                 hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
        f.update_layout(hovermode="closest")
        f.update_yaxes(autorange="reversed", showgrid=False)
        chart(style(f, 90 + 72 * len(tgts), legend=False), key="heat_chart")
    if st.session_state.density and r.get("air_density") is not None:
        st.caption(f"Air density during the outage, estimated from the reference parks: {np.mean(r['air_density']):.3f} kg/m³ (standard air 1.225).")


def report_html(results, park_name, insurer, policy):
    """Self-contained claim report. Opens in any browser; print to PDF from there."""
    lv = st.session_state.level
    start, end = CTX["start"], CTX["end"]
    f1 = go.Figure()
    for r in results:
        f1.add_scatter(x=r["timestamps"], y=r["wind_pred"], name=r["target"], line=dict(color=CTX["color"][r["target"]], width=2))
    f1.update_yaxes(title_text="Rebuilt wind (m/s)")
    f2 = go.Figure()
    for r in results:
        a, b = r["range_kwh"]
        f2.add_bar(x=[r["target"]], y=[r["energy_kwh"] / 1000], marker_color=CTX["color"][r["target"]], name=r["target"],
                   error_y=dict(type="data", symmetric=False, array=[(b - r["energy_kwh"]) / 1000], arrayminus=[(r["energy_kwh"] - a) / 1000], color=TEXT2))
    f2.update_yaxes(title_text="Lost energy (MWh)")
    c1 = style(f1, 300).to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False, "staticPlot": True})
    c2 = style(f2, 300, legend=False).to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "staticPlot": True})
    rows = "".join(f"<tr><td>{esc(r['target'])}</td><td class='n'>{np.mean(r['wind_pred']):.1f} m/s</td><td class='n'>{r['energy_kwh']:,.0f}</td>"
                   f"<td class='n'>{r['range_kwh'][0]:,.0f} – {r['range_kwh'][1]:,.0f}</td><td>{esc(', '.join(r['refs']))}</td></tr>" for r in results)
    acc = ""
    tw = CTX["test_window"]
    if tw and all("test_metrics" in r for r in results):
        acc_rows = "".join(f"<tr><td>{esc(r['target'])}</td><td class='n'>{r['test_metrics']['wind_rmse']:.2f}</td><td class='n'>{r['test_metrics']['wind_mae']:.2f}</td>"
                           f"<td class='n'>{r['test_metrics']['power_rmse']:,.0f}</td><td class='n'>{r['test_metrics']['power_mae']:,.0f}</td>"
                           f"<td class='n'>{r['test_metrics']['energy_dev_pct']:+.1f} %</td></tr>" for r in results)
        acc = (f"<h2>Accuracy check</h2><p>The same model was applied to {tw[0]:%d %b} – {tw[1]:%d %b %Y}, where measured data exists.</p>"
               f"<table><tr><th>Turbine</th><th class='n'>Wind RMSE (m/s)</th><th class='n'>Wind MAE (m/s)</th><th class='n'>Power RMSE (kW)</th>"
               f"<th class='n'>Power MAE (kW)</th><th class='n'>Energy deviation</th></tr>{acc_rows}</table>")
    to = (" · for " + esc(insurer)) if insurer else ""
    pol = (" · policy " + esc(policy)) if policy else ""
    dens = ", corrected for air density" if st.session_state.density else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(APP_NAME)} · claim report · {esc(park_name)}</title>
<style>
body{{font-family:{FONT};color:{TEXT1};background:#f5f5f7;margin:0;-webkit-font-smoothing:antialiased}}
.page{{max-width:900px;margin:32px auto;background:#fff;border-radius:18px;padding:48px 56px}}
.brand{{font-size:15px;font-weight:600;color:{TEXT1};display:flex;gap:8px;align-items:center}}
h1{{font-size:36px;letter-spacing:-0.02em;margin:22px 0 6px;font-weight:600}} h2{{font-size:21px;margin:36px 0 8px;font-weight:600;letter-spacing:-0.01em}}
p{{color:{TEXT2};font-size:15px;line-height:1.55;margin:6px 0}}
.k{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}}
.k div{{background:#f5f5f7;border-radius:14px;padding:16px}} .k .l{{font-size:12px;color:{TEXT2}}} .k .v{{font-size:24px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;margin-top:8px}}
th{{text-align:left;font-size:12px;color:{TEXT2};font-weight:600;padding:8px 10px;border-bottom:1px solid {GRID}}}
td{{padding:10px;border-bottom:1px solid {GRID}}} .n{{text-align:right;white-space:nowrap}}
ol{{color:{TEXT2};font-size:15px;line-height:1.6;padding-left:20px}} .foot{{font-size:12px;color:{TEXT3};margin-top:36px}}
@media print{{body{{background:#fff}} .page{{margin:0;padding:24px;border-radius:0}} h2{{break-after:avoid}}}}
</style></head><body><div class="page">
<div class="brand">{LOGO}{esc(APP_NAME)}</div>
<h1>Energy yield loss report</h1>
<p>{esc(park_name)} · communication outage {start:%d %B %Y, %H:%M} – {end + pd.Timedelta(minutes=10):%d %B %Y, %H:%M} ({CTX['hours']:.0f} hours){to}{pol}</p>
<div class="k"><div><div class="l">Estimated energy loss</div><div class="v">{CTX['total']:,.0f} kWh</div></div>
<div><div class="l">{lv} % range</div><div class="v">{CTX['lo']:,.0f} – {CTX['hi']:,.0f} kWh</div></div>
<div><div class="l">Affected turbines</div><div class="v">{len(results)}</div></div></div>
<h2>Per turbine</h2><table><tr><th>Turbine</th><th class="n">Mean rebuilt wind</th><th class="n">Lost energy (kWh)</th><th class="n">{lv} % range (kWh)</th><th>Reference turbines</th></tr>{rows}</table>
<h2>Rebuilt wind during the outage</h2>{c1}
<h2>Lost energy</h2>{c2}
{acc}
<h2>Method</h2><ol>
<li>Reference turbines in neighbouring parks were selected automatically by correlation over the {int(st.session_state.window_days)} days before the outage.</li>
<li>A blend of a linear model ({W_LIN} %) and gradient-boosted trees ({W_LGBM} %) learned the affected turbines' wind from the references' wind, time-shifted wind{', direction-dependent travel time' if TRAVEL else ''}, direction and turbulence.</li>
<li>The missing 10-minute wind speeds were rebuilt from the references, which kept measuring during the outage.</li>
<li>Energy was computed with the turbines' power curves, averaged over the typical spread of real wind around the estimate{dens}.</li>
<li>The {lv} % range comes from back-testing the method on 84 simulated 3-day outages.</li></ol>
<div class="foot">Generated by {esc(APP_NAME)}. All figures are estimates based on reference-park measurements.</div>
</div></body></html>"""


def page_report():
    section("Claim report", "Everything the insurer needs, ready to send.")
    if not need_results("report"):
        return
    results = CTX["results"]
    with st.container(key="card_details"):
        card_head("Claim details", "These appear in the email and the report.")
        c1, c2 = st.columns(2)
        park_name = c1.text_input("Wind park name", value=f"Wind park {'/'.join(sorted({park_of(t) for t in CTX['targets']}))}", key="park_name")
        insurer = c2.text_input("Addressee", placeholder="Claims department", key="insurer")
        policy = c1.text_input("Policy number", placeholder="Optional", key="policy")
        sender = c2.text_input("Your name", placeholder="Optional", key="sender")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1.2, 1], gap="medium")
    table = ac.results_table(results)
    with r1, st.container(key="card_email"):
        card_head("Insurance email", "Edit freely before sending.")
        text = ac.email_text(results, (CTX["start"], CTX["end"]), st.session_state.level, park_name, insurer, policy, sender, st.session_state.density)
        edited = st.text_area("Email", value=text, height=460, label_visibility="collapsed", key=f"email_{hashlib.md5(text.encode()).hexdigest()[:8]}")
        st.download_button("Download email", edited, file_name="insite_claim_email.txt", mime="text/plain", key="dl_email")
    with r2, st.container(key="card_downloads"):
        card_head("Claim report", "A one-page report with figures, charts, accuracy and method. Open it in a browser and choose Print › Save as PDF.")
        st.download_button("Download claim report", lambda: report_html(results, park_name, insurer, policy), file_name="insite_claim_report.html",
                           mime="text/html", type="primary", key="dl_report", width="stretch")
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        card_head("Data", f"{len(table):,} rows · every 10-minute step for each affected turbine.")
        st.download_button("Predicted time series (CSV)", table.to_csv(index=False), file_name="insite_claim_timeseries.csv", mime="text/csv",
                           key="dl_csv", width="stretch")
        summary = {"outage_start": str(CTX["start"]), "outage_end": str(CTX["end"]), "level_pct": st.session_state.level,
                   "air_density": bool(st.session_state.density), "total_kwh": CTX["total"], "range_kwh": [CTX["lo"], CTX["hi"]],
                   "turbines": {r["target"]: {"energy_kwh": r["energy_kwh"], "range_kwh": list(r["range_kwh"]), "references": r["refs"],
                                              **({"test_metrics": r["test_metrics"]} if "test_metrics" in r else {})} for r in results}}
        st.download_button("Summary (JSON)", json.dumps(summary, indent=2, default=float), file_name="insite_claim_summary.json",
                           mime="application/json", key="dl_json", width="stretch")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container(key="card_series"):
        n_per = table.groupby("turbine").size()
        card_head("Predicted time series", f"{', '.join(f'{t}: {n} rows' for t, n in n_per.items())}. Scroll inside the table to see every row.")
        st.dataframe(table, hide_index=True, width="stretch", height=380,
                     column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm"),
                                    "turbine": "Turbine",
                                    "predicted_wind_speed_ms": st.column_config.NumberColumn("Wind (m/s)", format="%.2f"),
                                    "expected_power_kw": st.column_config.NumberColumn("Power (kW)", format="%.0f"),
                                    "energy_kwh": st.column_config.NumberColumn("Energy (kWh)", format="%.1f")})


# ------------------------------------------------------------------ app shell
PAGES = {
    "Overview": st.Page(page_overview, title="Overview", url_path="overview", default=True),
    "Analysis": st.Page(page_analysis, title="Analysis", url_path="analysis"),
    "Accuracy": st.Page(page_accuracy, title="Accuracy", url_path="accuracy"),
    "Method": st.Page(page_method, title="Method", url_path="method"),
    "Claim report": st.Page(page_report, title="Claim report", url_path="report"),
}
pg = st.navigation(list(PAGES.values()), position="hidden")
st.html(CSS)


def on_nav():
    st.session_state["_goto"] = st.session_state.get("nav")


with st.container(key="navbar"):
    n1, n2 = st.columns([1, 2.6], vertical_alignment="center")
    n1.markdown(f"<div class='brand'>{LOGO}{APP_NAME}</div>", unsafe_allow_html=True)
    with n2:
        goto = st.session_state.pop("_goto", None)
        if not goto:
            st.session_state["nav"] = pg.title
        st.segmented_control("Navigation", list(PAGES), key="nav", on_change=on_nav, label_visibility="collapsed")
if goto and goto != pg.title and goto in PAGES:
    st.switch_page(PAGES[goto])

build_context()
pg.run()
