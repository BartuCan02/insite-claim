"""InSite Claim — outage energy-loss assessment. Run: uv run streamlit run app/app.py

UI only: all calculations come from src/app_core.py and src/pipeline.py (unchanged).
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import sys
import time
import zipfile
from email.message import EmailMessage
from urllib.parse import quote
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app_core as ac  # noqa: E402
import claim as cl  # noqa: E402
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



CSS2 = """
<style>
.kic { width:34px; height:34px; border-radius:10px; background:#e8f2fd; display:flex; align-items:center; justify-content:center; margin-bottom:10px; }
.flow { display:grid; grid-template-columns: 1fr 22px 1fr 22px 1fr 22px 1fr; gap:6px; align-items:stretch; margin:16px 0 6px; }
.flow .st { background:#f5f5f7; border-radius:14px; padding:14px 16px; }
.flow .st:last-child { background:#e8f2fd; }
.flow .l { font-size:12px; color:#6e6e73; font-weight:500; } .flow .v { font-size:22px; font-weight:600; color:#1d1d1f; margin:4px 0 2px; font-variant-numeric:tabular-nums; }
.flow .c { font-size:12px; color:#86868b; line-height:1.4; } .flow .arr { display:flex; align-items:center; justify-content:center; color:#86868b; font-size:22px; }
.trail .it { display:flex; gap:14px; align-items:flex-start; padding:12px 0; border-bottom:1px solid #e8e8ed; position:relative; }
.trail .it:last-child { border-bottom:none; }
.trail .dot { flex:0 0 26px; height:26px; border-radius:50%; background:#e9f6ec; color:#248a3d; font-size:12px; font-weight:700; display:flex; align-items:center; justify-content:center; }
.trail .tx { flex:1; } .trail .tx b { display:block; font-size:15px; font-weight:600; color:#1d1d1f; } .trail .tx span { font-size:13px; color:#6e6e73; line-height:1.5; }
.trail .tag { font-size:12px; font-weight:600; color:#248a3d; background:#e9f6ec; border-radius:980px; padding:3px 10px; white-space:nowrap; }
.atth { font-size:13px; font-weight:600; color:#6e6e73; margin:14px 0 6px; }
div[class*="st-key-att_row_"] { background:#fafafc; border:1px solid #e8e8ed; border-radius:12px; padding:4px 10px; margin-bottom:6px; }
div[class*="st-key-att_row_"] [data-testid="stHorizontalBlock"] { gap:8px; }
div[class*="st-key-att_row_"] button { min-height:30px !important; padding:0 8px !important; color:#6e6e73 !important; font-size:18px !important; }
.attn { font-size:14px; font-weight:500; color:#1d1d1f; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.atts { font-size:13px; color:#86868b; text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
.st-key-card_composer [data-testid="stTextInput"] label p, .st-key-card_composer [data-testid="stTextArea"] label p { font-size:12px; color:#6e6e73; }
.st-key-dl_eml button { background: linear-gradient(180deg,#1a8cff,#0071e3) !important; border:none !important; color:#fff !important; }

/* presentation mode */
.st-key-pz_ui, .st-key-pz_uic, .st-key-pz_exit, .st-key-pz_cap, .st-key-pz_float, .st-key-pz_load { position:fixed !important; z-index:1000; }
.st-key-pz_ui { right:28px; bottom:26px; width:auto !important; background:rgba(255,255,255,.86); backdrop-filter:blur(20px) saturate(180%);
  -webkit-backdrop-filter:blur(20px) saturate(180%); border:1px solid rgba(0,0,0,.06); border-radius:980px; padding:6px 8px 6px 10px; box-shadow:0 12px 40px rgba(0,0,0,.14); align-items:center; }
.st-key-pz_uic { left:50%; transform:translateX(-50%); bottom:64px; width:auto !important; align-items:center; gap:18px !important; }
.st-key-pz_uic button { background:rgba(20,22,26,.30) !important; border:1px solid rgba(255,255,255,.65) !important; color:#fff !important;
  backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); min-width:170px; min-height:48px !important; font-size:17px !important; }
.st-key-pz_uic button p { font-size:17px !important; color:#fff !important; }
.st-key-pz_exit { right:24px; top:18px; width:auto !important; align-items:center; }
.st-key-pz_exit button { color:#6e6e73 !important; }
.pzfs { background:transparent; border:none; color:#6e6e73; font:500 13px -apple-system,BlinkMacSystemFont,Arial; cursor:pointer; padding:6px 10px; border-radius:980px; }
.pzfs:hover { background:rgba(0,0,0,.05); }
.pzprog { font:500 13px -apple-system,BlinkMacSystemFont,Arial; color:#6e6e73; padding:0 8px; white-space:nowrap; font-variant-numeric:tabular-nums; display:flex; gap:8px; align-items:center; }
.pzprog span { display:inline-flex; gap:4px; } .pzprog i { width:6px; height:6px; border-radius:50%; background:#d2d2d7; display:inline-block; } .pzprog i.on { background:#0071e3; }
.st-key-pz_uic .pzprog { color:rgba(255,255,255,.8); }
.st-key-pz_cap { top:22px; left:50%; transform:translateX(-50%); width:auto !important; }
.pzcap { background:rgba(29,29,31,.88); color:#fff; padding:11px 22px; border-radius:980px; font:500 17px -apple-system,BlinkMacSystemFont,Arial;
  letter-spacing:-0.01em; box-shadow:0 10px 30px rgba(0,0,0,.18); white-space:nowrap; backdrop-filter:blur(10px); animation: pzin .45s ease both; }
.pzcap b { color:#64b5ff; font-weight:600; }
@keyframes pzin { from { opacity:0; transform:translateY(-6px) scale(.98); } to { opacity:1; transform:none; } }
.st-key-pz_cine { position:fixed !important; inset:0; z-index:900; }
.pzc { position:fixed; inset:0; z-index:900; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; overflow:hidden; animation: pzfade .6s ease both; }
@keyframes pzfade { from { opacity:0; } to { opacity:1; } }
.pzc h1 { font:300 76px/1.08 -apple-system,BlinkMacSystemFont,'SF Pro Display','Inter',Arial; letter-spacing:-0.035em; margin:0; position:relative; z-index:2; }
.pzc h1 b { font-weight:700; }
.pzc p { font:400 21px/1.4 -apple-system,BlinkMacSystemFont,Arial; margin:22px 0 0; position:relative; z-index:2; }
.pzc svg.sil { position:absolute; left:0; right:0; bottom:0; width:100%; height:62%; z-index:1; }
.pz1 { background: radial-gradient(1200px 500px at 50% 82%, rgba(210,215,224,.28), transparent 70%), linear-gradient(180deg,#1f2b3c 0%,#3b4b62 38%,#6f7b8c 64%,#3a3f47 82%,#1a1d22 100%); }
.pz1 h1 { color:#f5f5f7; } .pz1 p { color:rgba(245,245,247,.7); }
.pz2 { background: radial-gradient(900px 340px at 62% 60%, rgba(255,214,170,.65), transparent 70%), radial-gradient(1400px 380px at 50% 88%, rgba(255,255,255,.75), transparent 70%),
  linear-gradient(180deg,#7fa6d4 0%,#c9d8ea 34%,#f3e2d1 58%,#e9cdb6 68%,#93a08d 84%,#5f6d5c 100%); }
.pz2 h1 { color:#1d1d1f; } .pz2 h1 b { color:#1d1d1f; } .pz2 p { color:#3a3a3c; }
.st-key-pz_float { left:0; right:0; margin:0 auto; top:76px; width:min(780px, 92vw) !important; max-height:calc(100vh - 160px); overflow-y:auto;
  background:rgba(255,255,255,.80); backdrop-filter:blur(30px) saturate(180%); -webkit-backdrop-filter:blur(30px) saturate(180%);
  border:1px solid rgba(255,255,255,.7); border-radius:28px; padding:28px 30px !important; box-shadow:0 40px 100px rgba(16,24,40,.22); animation: pzin .45s ease both; }
.st-key-pz_float.wide, .st-key-pz_tech { width:min(1060px, 94vw) !important; }
div[class*="st-key-pz_r"] { background:#fff; border-radius:18px; padding:14px 18px; box-shadow:0 1px 2px rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.05); }
.pzic { width:44px; height:44px; border-radius:50%; background:#f2f2f4; display:flex; align-items:center; justify-content:center; }
.pzrl { font-size:12px; color:#6e6e73; font-weight:500; margin-bottom:2px; }
.pzh { font:600 30px/1.15 -apple-system,BlinkMacSystemFont,'SF Pro Display',Arial; letter-spacing:-0.02em; color:#1d1d1f; margin:2px 0 6px; }
.pzs { font-size:15px; color:#6e6e73; margin-bottom:16px; }
.st-key-pz_analyze button { background:linear-gradient(180deg,#2d95ff,#0071e3) !important; border:none !important; min-height:54px !important; font-size:18px !important; box-shadow:0 10px 26px rgba(0,113,227,.35) !important; }
.st-key-pz_analyze button p { font-size:18px !important; font-weight:600 !important; }
.st-key-pz_load { inset:0; width:100vw !important; height:100vh !important; display:flex !important; align-items:center !important; justify-content:center !important; background:rgba(245,245,247,.35); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
.st-key-pz_load [data-testid='stElementContainer'] { width:auto !important; }
.pzl { width:340px; background:#fff; border-radius:28px; padding:36px 28px 30px; text-align:center; box-shadow:0 40px 100px rgba(16,24,40,.22); animation: pzin .4s ease both; }
.pzring { position:relative; width:132px; height:132px; margin:0 auto 18px; }
.pzring svg { transform:rotate(-90deg); }
.pzring .arc { stroke-dasharray: 340; stroke-dashoffset: 340; animation: pzarc 2.4s cubic-bezier(.4,0,.2,1) forwards; }
@keyframes pzarc { to { stroke-dashoffset: 0; } }
.pzring .num { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font:600 28px -apple-system,Arial; color:#1d1d1f; font-variant-numeric:tabular-nums; }
.pzring .num::after { content:"0%"; animation: pznum 2.4s linear forwards; }
@keyframes pznum { 0% { content:"0%"; } 8% { content:"12%"; } 16% { content:"27%"; } 24% { content:"41%"; } 32% { content:"54%"; } 40% { content:"64%"; } 50% { content:"73%"; } 60% { content:"81%"; } 70% { content:"88%"; } 80% { content:"94%"; } 90% { content:"98%"; } 100% { content:"100%"; } }
.pzl b { display:block; font:500 18px -apple-system,Arial; color:#1d1d1f; }
.pzrot { position:relative; height:20px; margin-top:8px; font-size:13px; color:#86868b; }
.pzrot span { position:absolute; left:0; right:0; opacity:0; animation: pzrot 2.4s linear forwards; }
.pzrot span:nth-child(2) { animation-delay: .8s; } .pzrot span:nth-child(3) { animation-delay: 1.6s; animation-name: pzrotlast; }
@keyframes pzrot { 0% { opacity:0 } 10% { opacity:1 } 30% { opacity:1 } 40% { opacity:0 } 100% { opacity:0 } }
@keyframes pzrotlast { 0% { opacity:0 } 10% { opacity:1 } 100% { opacity:1 } }
.pztech { display:grid; grid-template-columns: 1fr 1fr; gap:22px; }
.pztech .col { background:#fff; border-radius:20px; padding:22px 24px; border:1px solid rgba(0,0,0,.05); }
.pztech h3 { font:600 19px -apple-system,Arial; margin:0 0 12px; color:#1d1d1f; letter-spacing:-0.01em; }
.pzstep { display:flex; gap:12px; align-items:flex-start; padding:9px 0; }
.pzstep .n { flex:0 0 24px; height:24px; border-radius:50%; background:#e8f2fd; color:#0071e3; font:600 12px Arial; display:flex; align-items:center; justify-content:center; }
.pzstep b { display:block; font-size:15px; color:#1d1d1f; } .pzstep span { font-size:13px; color:#6e6e73; }
.pzbar { display:grid; grid-template-columns: 130px 1fr 60px; gap:10px; align-items:center; padding:5px 0; font-size:13px; color:#1d1d1f; }
.pzbar .tr { height:10px; background:#f2f2f4; border-radius:6px; overflow:hidden; } .pzbar .fl { height:10px; border-radius:6px; }
.pzbar .vv { text-align:right; color:#6e6e73; font-variant-numeric:tabular-nums; }
.pzsmall { font-size:12px; color:#86868b; margin-top:10px; line-height:1.5; }
</style>
"""


# ------------------------------------------------------------------ helpers
def esc(x):
    return html.escape(str(x))


def kpis(cards):
    st.markdown("<div class='kgrid'>" + "".join(
        f"<div class='kpi'>{(c[3] if len(c) > 3 else '')}<div class='l'>{esc(c[0])}</div><div class='v'>{c[1]}</div><div class='c'>{esc(c[2])}</div></div>"
        for c in cards) + "</div>", unsafe_allow_html=True)


def kicon(path):
    return (f"<div class='kic'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#0071e3' stroke-width='2' "
            f"stroke-linecap='round' stroke-linejoin='round'>{path}</svg></div>")


IC_BOLT = "<path d='M13 2 4 14h7l-1 8 9-12h-7z'/>"
IC_COIN = "<ellipse cx='12' cy='6' rx='8' ry='3'/><path d='M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6'/><path d='M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6'/>"
IC_SHIELD = "<path d='M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z'/><path d='m9 12 2 2 4-4'/>"
IC_CLOCK = "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>"


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


def _bound_change(key, state_key):
    v = st.session_state.get(key)
    if v is None:
        st.session_state[key] = st.session_state[state_key]
    else:
        st.session_state[state_key] = v


def segmented_bound(label, options, key, state_key, fmt=None):
    """Segmented control tied to a persistent setting (stays in sync across pages and presentation mode)."""
    if st.session_state.get(key) != st.session_state[state_key]:
        st.session_state[key] = st.session_state[state_key]
    kw = {"format_func": fmt} if fmt else {}
    st.segmented_control(label, options, key=key, on_change=_bound_change, args=(key, state_key), label_visibility="collapsed", **kw)
    return st.session_state[state_key]


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
            "min_gap_h": 6.0, "outage": None, "targets": None, "fp": None, "uploads_w": (), "uploads_c": (), "park": None,
            "contract_src": "Sample contract", "contract_up": None, "ovs": {}, "stage": 1, "sub": 0, "stage_t": 0.0,
            "att_removed": [], "att_extra": [], "att_n": 0}
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
    CTX.update(ok=True, key=key, results=results, targets=targets, start=start, end=end, test_window=test_window,
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
        segmented_bound("Source", ["Example dataset", "Upload files"], "_source", "source")
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
        segmented_bound("Confidence of the range", ["80", "90"], "_level", "level", fmt=lambda x: f"{x} %")
        st.toggle("Air density correction", value=bool(st.session_state.density), key="_density", on_change=keep, args=("density",),
                  help="Adjusts the power curve for air density from temperature and pressure (IEC 61400-12). Changes results by about 2 %.")
        st.number_input("Gaps longer than this count as an outage (hours)", 1.0, 48.0, float(st.session_state.min_gap_h), 1.0,
                        key="_min_gap_h", on_change=keep, args=("min_gap_h",))


def kpi_cards():
    results, lv = CTX["results"], st.session_state.level
    _, calc, _ = contract_and_calc()
    cards = [("Estimated energy loss", f"{CTX['total']:,.0f}<span>kWh</span>",
              f"{lv} % range {CTX['lo']/1000:,.0f}–{CTX['hi']/1000:,.0f} MWh · {len(results)} turbines", kicon(IC_BOLT))]
    if calc and calc.get("rate"):
        exposure = calc["total_kwh"] * calc["rate"]
        cards.append(("Claim estimate", f"EUR {calc['amount']:,.0f}",
                      f"of EUR {exposure:,.0f} lost revenue · after the {calc['waiting_h']:g} h waiting period · subject to coverage review", kicon(IC_COIN)))
    else:
        cards.append(("Claim estimate", "—", "Add the insurance contract on the Claim report page", kicon(IC_COIN)))
    tm = [r["test_metrics"] for r in results if "test_metrics" in r]
    tw = CTX["test_window"]
    if tm and tw and len(tm) == len(results):
        mae = float(np.mean([m["wind_mae"] for m in tm]))
        et, ep = sum(m["energy_true_kwh"] for m in tm), sum(m["energy_pred_kwh"] for m in tm)
        cards.append(("Validated accuracy", f"{mae:.2f}<span>m/s</span>",
                      f"wind MAE against measured data, {tw[0]:%d}–{tw[1]:%d %b} · summed energy {100 * (ep - et) / et:+.1f} %", kicon(IC_SHIELD)))
    else:
        mae = float(np.mean([r["inner_metrics"]["wind_mae"] for r in results]))
        cards.append(("Back-tested accuracy", f"{mae:.2f}<span>m/s</span>", "wind MAE on the training days, each piece hidden in turn", kicon(IC_SHIELD)))
    cards.append(("Outage duration", f"{CTX['hours']:.0f}<span>h</span>",
                  f"{CTX['start']:%d %b %H:%M} → {CTX['end'] + pd.Timedelta(minutes=10):%d %b %H:%M}", kicon(IC_CLOCK)))
    with st.container(key="kpiwrap"):
        kpis(cards)


def page_analysis(present=False):
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
    kpi_cards()

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
        top_t = max(results, key=lambda x: x["energy_kwh"])["target"]
        for r in results:
            a, b = r["range_kwh"]
            mc = (BLUE if r["target"] == top_t else HAIR) if present else color[r["target"]]
            f.add_bar(x=[r["target"]], y=[r["energy_kwh"] / 1000], marker_color=mc, name=r["target"],
                      error_y=dict(type="data", symmetric=False, array=[(b - r["energy_kwh"]) / 1000], arrayminus=[(r["energy_kwh"] - a) / 1000],
                                   color=TEXT2, thickness=1.2, width=6),
                      hovertemplate="%{x}: %{y:,.1f} MWh<extra></extra>")
        if present:
            tot_e = sum(r["energy_kwh"] for r in results)
            srt = sorted(results, key=lambda x: -x["energy_kwh"])
            f = go.Figure(go.Bar(y=[r["target"] for r in srt], x=[100 * r["energy_kwh"] / tot_e for r in srt], orientation="h",
                                 marker_color=[BLUE if r["target"] == top_t else HAIR for r in srt],
                                 text=[f"{100 * r['energy_kwh'] / tot_e:.0f} %" for r in srt], textposition="outside",
                                 hovertemplate="%{y}: %{x:.0f} % of the loss<extra></extra>"))
            f.update_xaxes(range=[0, 60], title_text="share of the loss (%)")
            f.update_yaxes(autorange="reversed", showgrid=False)
        else:
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
        if present:
            tot = bt.groupby("band", observed=False).kwh.sum()
            top_b = tot.idxmax()
            f.add_bar(x=tot.index.astype(str), y=tot.values / 1000, marker_color=[BLUE if b == top_b else HAIR for b in tot.index],
                      text=[f"{v / 1000:,.0f} MWh" if v > 0 else "" for v in tot.values], textposition="outside",
                      hovertemplate="%{x}: %{y:,.1f} MWh<extra></extra>", showlegend=False)
        else:
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


def report_html(results, park_name, insurer, policy, contract_html=""):
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
{contract_html}
{acc}
<h2>Method</h2><ol>
<li>Reference turbines in neighbouring parks were selected automatically by correlation over the {int(st.session_state.window_days)} days before the outage.</li>
<li>A blend of a linear model ({W_LIN} %) and gradient-boosted trees ({W_LGBM} %) learned the affected turbines' wind from the references' wind, time-shifted wind{', direction-dependent travel time' if TRAVEL else ''}, direction and turbulence.</li>
<li>The missing 10-minute wind speeds were rebuilt from the references, which kept measuring during the outage.</li>
<li>Energy was computed with the turbines' power curves, averaged over the typical spread of real wind around the estimate{dens}.</li>
<li>The {lv} % range comes from back-testing the method on 84 simulated 3-day outages.</li></ol>
<div class="foot">Generated by {esc(APP_NAME)}. All figures are estimates based on reference-park measurements.</div>
</div></body></html>"""


@st.cache_data(show_spinner=False, max_entries=4)
def report_cached(key, _results, park_name, insurer, policy, contract_html):
    """Built during the page run (it needs the session's settings) and cached per input."""
    return report_html(_results, park_name, insurer, policy, contract_html)


SAMPLE_CONTRACT = ROOT / "data" / "sample_contract" / "sample_bi_policy_wind_park_T.pdf"


@st.cache_data(show_spinner=False)
def parse_contract_cached(name, data):
    return cl.parse_contract(name, data)


def keep_contract():
    f = st.session_state.get("_contract_up")
    st.session_state.contract_up = (f.name, f.getvalue()) if f is not None else None


def term_value(t):
    if t.value is None:
        return "Not found"
    if t.key == "waiting_h":
        return f"{t.value:g} hours"
    if t.key == "rate_eur_kwh":
        return f"EUR {t.value:.4f} per kWh"
    if t.key == "cap_eur":
        return f"EUR {t.value:,.0f}"
    if t.key in ("notice_days", "payment_days"):
        return f"{t.value:.0f} days"
    return "Found"


def contract_source():
    src = st.session_state.contract_src
    if src == "Sample contract" and SAMPLE_CONTRACT.exists():
        return SAMPLE_CONTRACT.name, SAMPLE_CONTRACT.read_bytes()
    if src == "Upload contract" and st.session_state.contract_up:
        return st.session_state.contract_up
    return None, None


def _ov(name):
    v = st.session_state.ovs.get(name)
    return float(v) if v else None


def keep_ov(name):
    st.session_state.ovs[name] = st.session_state.get(name)


def contract_and_calc():
    """(contract, calc, fingerprint) for the current contract choice; (None, None, '') without one."""
    name, data = contract_source()
    if not data or not CTX.get("ok"):
        return None, None, ""
    try:
        contract = parse_contract_cached(name, data)
    except Exception:  # noqa: BLE001
        return None, None, ""
    fp = hashlib.md5(data).hexdigest()[:10]
    ov = {"rate_eur_kwh": _ov(f"ov_rate_{fp}"), "waiting_h": _ov(f"ov_wait_{fp}"), "cap_eur": _ov(f"ov_cap_{fp}"),
          "notice_days": _ov(f"ov_notice_{fp}")}
    rel = (RECIPE["energy"]["range_park_rel"] if len(CTX["results"]) > 1 else RECIPE["energy"]["range_turbine_rel"])[st.session_state.level]
    calc = cl.compute_claim(CTX["results"], (CTX["start"], CTX["end"]), contract, ov, rel)
    return contract, calc, fp + json.dumps(ov, sort_keys=True)


def contract_card():
    """Contract choice, the clauses found in it, and the claim it implies. Returns (contract, calc, no_exclusion, fingerprint)."""
    with st.container(key="card_contract"):
        card_head("Insurance contract", "The email and report cite your policy's clauses, so the claim rests on what was agreed.")
        src = segmented_bound("Contract", ["Sample contract", "Upload contract", "No contract"], "_contract_src", "contract_src")
        if src == "Sample contract" and SAMPLE_CONTRACT.exists():
            st.caption("A sample business interruption policy for wind park T, with fictional parties.")
            st.download_button("View sample contract (PDF)", SAMPLE_CONTRACT.read_bytes(), file_name=SAMPLE_CONTRACT.name, mime="application/pdf",
                               key="dl_sample_contract")
        elif src == "Upload contract":
            st.file_uploader("Contract", type=["pdf", "docx", "txt", "md"], key="_contract_up", on_change=keep_contract,
                             label_visibility="collapsed", help="PDF, Word or text. Clauses should be numbered, for example 4.1.")
        contract, calc, cfp = contract_and_calc()
        if src == "Upload contract" and st.session_state.contract_up and contract is None:
            st.markdown(f"{badge('Could not read the contract', 'orange')}", unsafe_allow_html=True)
        if contract is None:
            st.markdown(f"<div style='margin-top:10px'>{badge('Coverage review required', 'orange')} <span style='color:{TEXT2};font-size:13px'>"
                        "Without a contract the email is a preliminary notification, not a claim for a specific amount.</span></div>",
                        unsafe_allow_html=True)
            return None, None, False, ""
        fp = cfp[:10]
        pre, T = contract.preamble, contract.terms
        st.markdown(f"<div style='margin:10px 0 6px'>{badge('Contract read', 'green')} &nbsp;<span style='color:{TEXT2};font-size:13px'>"
                    f"{esc(pre.get('policy', 'policy number not found'))} · {esc(pre.get('insurer', 'insurer not found'))} · "
                    f"{len(contract.clauses)} clauses</span></div>", unsafe_allow_html=True)
        rows = [{"Term": t.label, "Value": term_value(t), "Clause": t.clause or "—",
                 "Contract text": (t.quote[:150] + "…") if len(t.quote) > 150 else (t.quote or "—")} for t in T.values()]
        st.markdown(table_html(pd.DataFrame(rows), numeric=set()), unsafe_allow_html=True)
        with st.expander("Adjust contract terms"):
            a1, a2, a3, a4 = st.columns(4)
            spec = [(a1, "ov_rate", "rate_eur_kwh", "Rate (EUR per kWh)", 10.0, 0.001, "%.4f"), (a2, "ov_wait", "waiting_h", "Waiting period (h)", 240.0, 1.0, None),
                    (a3, "ov_cap", "cap_eur", "Limit per event (EUR)", 1e9, 1000.0, None), (a4, "ov_notice", "notice_days", "Notify within (days)", 365.0, 1.0, None)]
            for col, pfx, term, label, mx, stp, fmt in spec:
                k = f"{pfx}_{fp}"
                cur = _ov(k) if _ov(k) is not None else float(T[term].value or 0.0)
                kw = {"format": fmt} if fmt else {}
                col.number_input(label, 0.0, mx, cur, stp, key=k, on_change=keep_ov, args=(k,), **kw)
        nk = f"noexcl_{fp}"
        no_excl = st.checkbox("Confirm: no planned maintenance and no curtailment ordered by the grid operator during the outage"
                              + (f" (clause {T['exclusions'].clause})" if T["exclusions"].clause else ""),
                              value=bool(st.session_state.ovs.get(nk)), key=nk, on_change=keep_ov, args=(nk,))
        contract, calc, cfp = contract_and_calc()
        c = lambda k: f"clause {T[k].clause}" if T.get(k) and T[k].clause else "per contract"  # noqa: E731
        steps = [("Potential generation", f"{calc['total_kwh']:,.0f} kWh", "rebuilt for the outage"),
                 ("Not covered", f"−{calc['waiting_kwh']:,.0f} kWh", f"{calc['waiting_h']:g} h waiting period · {c('waiting_h')}"),
                 ("Claimable energy", f"{calc['claimable_kwh']:,.0f} kWh", f"{st.session_state.level} % range {calc['claimable_lo']:,.0f} – {calc['claimable_hi']:,.0f}")]
        if calc.get("rate"):
            steps.append(("Proposed claim", f"EUR {calc['amount']:,.0f}",
                          f"EUR {calc['rate']:.4f}/kWh · {c('rate_eur_kwh')}" + (" · capped" if calc.get("capped") else "") + " · subject to insurer review"))
        st.markdown("<div class='flow'>" + "<div class='arr'>›</div>".join(
            f"<div class='st'><div class='l'>{esc(a_)}</div><div class='v'>{esc(b_)}</div><div class='c'>{esc(c_)}</div></div>" for a_, b_, c_ in steps)
            + "</div>", unsafe_allow_html=True)
        if calc.get("notify_by"):
            st.caption(f"Notify the insurer by {calc['notify_by']:%d %B %Y} ({calc['notice_days']:.0f} days after the outage, {c('notice_days')}).")
        if calc.get("below_waiting"):
            st.warning("The outage is shorter than the waiting period, so nothing is claimable under this contract.")
        return contract, calc, no_excl, cfp + str(no_excl)


def evidence_card(calc):
    res, wide = CTX["results"], CTX["wide"]
    tg = [r["target"] for r in res]
    refs = sorted({x for r in res for x in r["refs"]})
    te = (wide.index >= CTX["start"]) & (wide.index <= CTX["end"])
    cov = 100 * float(wide.loc[te, "wind_speed"][refs].notna().mean().mean())
    curve = res[0]["power_curve"]
    cut_in = float(curve.wind_speed[curve.power > 0].min())
    rated = float(curve.power.max())
    n_tr = len(res[0]["train_idx"])
    n_out = sum(len(r["timestamps"]) for r in res)
    tm = [r["test_metrics"] for r in res if "test_metrics" in r]
    tw = CTX["test_window"]
    if tm and tw and len(tm) == len(res):
        et, ep = sum(m["energy_true_kwh"] for m in tm), sum(m["energy_pred_kwh"] for m in tm)
        val = (f"Measured {tw[0]:%d}–{tw[1]:%d %b}: wind MAE {np.mean([m['wind_mae'] for m in tm]):.2f} m/s, summed energy {100 * (ep - et) / et:+.1f} %. "
               f"Range calibrated on 84 simulated 3-day outages.")
    else:
        val = f"Back-test on the training days: wind MAE {np.mean([r['inner_metrics']['wind_mae'] for r in res]):.2f} m/s."
    prod = f"{CTX['total']:,.0f} kWh lost"
    if calc and calc.get("rate"):
        prod += f" → {calc['claimable_kwh']:,.0f} kWh claimable → EUR {calc['amount']:,.0f} proposed"
    items = [
        ("Incident", f"{CTX['start']:%d %b %Y, %H:%M} → {CTX['end'] + pd.Timedelta(minutes=10):%d %b %Y, %H:%M} · {CTX['hours']:.0f} h · "
                     f"{', '.join(tg)} without data", "Detected"),
        ("Operational measurements", f"{n_tr:,} ten-minute values per affected turbine from the {int(st.session_state.window_days)} days before the outage", "Used"),
        ("Reference measurements", f"{', '.join(refs)} · {cov:.0f} % of values present during the outage", "Used"),
        ("Reconstructed wind speed", f"{n_out:,} ten-minute values ({len(res)} turbines × {len(res[0]['timestamps'])})", "Generated"),
        ("Power curves", f"{', '.join(tg)} · cut-in {cut_in:g} m/s · rated {rated:,.0f} kW", "Used"),
        ("Validation", val, "Measured"),
        ("Estimated production", prod, "Calculated"),
    ]
    with st.container(key="card_evidence"):
        card_head("Evidence", "Every number in the claim traces back to these sources.")
        st.markdown("<div class='trail'>" + "".join(
            f"<div class='it'><div class='dot'>{i}</div><div class='tx'><b>{esc(a_)}</b><span>{esc(b_)}</span></div>"
            f"<div class='tag'>{esc(c_)}</div></div>" for i, (a_, b_, c_) in enumerate(items, 1)) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- email composer
EXT_COLORS = {"pdf": "#e5484d", "doc": "#2f6fed", "docx": "#2f6fed", "csv": "#30a46c", "xlsx": "#30a46c", "xls": "#30a46c",
              "zip": "#6e6e73", "html": "#7d5bd0", "txt": "#8e8e93", "json": "#8e8e93", "md": "#8e8e93"}
MIMES = {"pdf": "application/pdf", "zip": "application/zip", "csv": "text/csv", "html": "text/html", "txt": "text/plain", "json": "application/json",
         "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "md": "text/markdown",
         "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def file_icon(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    col = EXT_COLORS.get(ext, "#8e8e93")
    return (f"<svg width='26' height='32' viewBox='0 0 26 32' aria-hidden='true'><path d='M3 1h13l8 8v20a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2z' fill='{col}'/>"
            f"<path d='M16 1v6a2 2 0 0 0 2 2h6' fill='rgba(255,255,255,.45)'/>"
            f"<text x='12.5' y='24' font-size='7' font-weight='700' fill='#fff' text-anchor='middle' font-family='-apple-system,Arial'>{esc(ext.upper()[:4])}</text></svg>")


def human_size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(n / 1e3, 0.1):.0f} KB" if n >= 1e3 else f"{n} B"


@st.cache_data(show_spinner=False, max_entries=4)
def generated_attachments(key, _contract, _calc, contract_name, _contract_bytes, park_name, policy, _summary):
    """Files produced from the current analysis (cached per case)."""
    res, lv = CTX["results"], st.session_state.level
    ctxd = {k: CTX[k] for k in ("start", "end", "hours", "total", "lo", "hi", "test_window")}
    out = []
    try:
        import claim_docs as cd
        out.append(("InSite_Claim_Report.pdf", cd.report_pdf(res, ctxd, _contract, _calc, lv, park_name, (_contract.preamble.get("insurer", "") if _contract else ""),
                                                             policy, int(st.session_state.window_days), W_LIN, W_LGBM, TRAVEL), MIMES["pdf"]))
        if contract_name:
            out.append((contract_name, _contract_bytes, MIMES.get(contract_name.rsplit(".", 1)[-1].lower(), "application/octet-stream")))
        out.append(("Supporting_Evidence.zip", cd.evidence_zip(CTX["wide"], res, ctxd, _calc, contract_name, _contract_bytes, _summary), MIMES["zip"]))
        if _calc:
            out.append(("Claim_Calculation.csv", cd.calculation_csv(_calc, res), MIMES["csv"]))
    except ImportError:
        chtml = cl.report_section_html(_contract, _calc, lv, esc) if _contract else ""
        out.append(("InSite_Claim_Report.html", report_html(res, park_name, "", policy, chtml).encode(), MIMES["html"]))
        if contract_name:
            out.append((contract_name, _contract_bytes, MIMES.get(contract_name.rsplit(".", 1)[-1].lower(), "application/octet-stream")))
        out.append(("Reconstructed_Wind.csv", ac.results_table(res).to_csv(index=False).encode(), MIMES["csv"]))
    return out


def rm_att(name):
    extras = st.session_state.att_extra
    if any(n == name for n, _, _ in extras):
        st.session_state.att_extra = [x for x in extras if x[0] != name]
    else:
        st.session_state.att_removed = st.session_state.att_removed + [name]


def add_att(key):
    for f in st.session_state.get(key) or []:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        st.session_state.att_extra = [x for x in st.session_state.att_extra if x[0] != f.name] + [(f.name, f.getvalue(), MIMES.get(ext, "application/octet-stream"))]
        st.session_state.att_removed = [n for n in st.session_state.att_removed if n != f.name]
    st.session_state.att_n += 1  # fresh uploader next run


def build_eml(to, cc, subject, body, atts):
    m = EmailMessage()
    if to:
        m["To"] = to
    if cc:
        m["Cc"] = cc
    m["Subject"] = subject
    m["X-Unsent"] = "1"  # opens as a draft in Outlook and Apple Mail
    m.set_content(body)
    for name, data, mime in atts:
        maintype, subtype = (mime or "application/octet-stream").split("/", 1)
        m.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return m.as_bytes()


def build_zip(subject, body, atts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("email.txt", f"Subject: {subject}\n\n{body}")
        for name, data, _ in atts:
            z.writestr(name, data)
    return buf.getvalue()


def composer_card(contract, calc, no_excl, cfp):
    res, lv = CTX["results"], st.session_state.level
    park_name = f"Wind park {'/'.join(sorted({park_of(t) for t in CTX['targets']}))}"
    pre = contract.preamble if contract else {}
    refs = sorted({x for r in res for x in r["refs"]})
    sender = "Wind Operations Team"
    if contract:
        text = cl.claim_email(res, (CTX["start"], CTX["end"]), contract, calc, lv, park_name, pre.get("insurer", ""), sender, no_excl, refs,
                              int(st.session_state.window_days))
    else:
        text = ac.email_text(res, (CTX["start"], CTX["end"]), lv, park_name, "", "", sender, st.session_state.density)
        text = text.replace("Subject: Business interruption claim", "Subject: Preliminary notification of a business interruption")
    first, _, body = text.partition("\n")
    subject = first.removeprefix("Subject: ").strip()
    body = body.lstrip("\n")
    summary = {"outage_start": str(CTX["start"]), "outage_end": str(CTX["end"] + pd.Timedelta(minutes=10)), "level_pct": lv,
               "air_density": bool(st.session_state.density), "total_kwh": CTX["total"], "range_kwh": [CTX["lo"], CTX["hi"]],
               "turbines": {r["target"]: {"energy_kwh": r["energy_kwh"], "range_kwh": list(r["range_kwh"]), "references": r["refs"],
                                          **({"test_metrics": r["test_metrics"]} if "test_metrics" in r else {})} for r in res}}
    if calc:
        summary["claim"] = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in calc.items()}
        summary["claim"]["policy"] = pre.get("policy")
    cname, cbytes = contract_source() if contract else (None, None)
    with st.spinner("Preparing attachments…"):
        gen = generated_attachments(CTX["key"] + cfp, contract, calc, cname, cbytes, park_name, pre.get("policy", ""), summary)
    atts = [a for a in gen if a[0] not in st.session_state.att_removed] + list(st.session_state.att_extra)
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    with st.container(key="card_composer"):
        t1, t2 = st.columns([4, 1], vertical_alignment="center")
        t1.markdown("<div class='ctitle' style='font-size:22px'>Prepare Insurance Claim</div>"
                    "<div class='csub'>Review and edit. InSite Claim never sends email itself.</div>", unsafe_allow_html=True)
        t2.markdown(f"<div style='text-align:right'>{badge('Draft', 'blue')}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        to = c1.text_input("To", placeholder="claims@your-insurer.example", key="mail_to")
        cc = c2.text_input("Cc", placeholder="Optional", key="mail_cc")
        subj = st.text_input("Subject", value=subject, key=f"mail_subj_{h}")
        msg = st.text_area("Message", value=body, height=330, key=f"mail_body_{h}")
        st.markdown(f"<div class='atth'>Attachments ({len(atts)})</div>", unsafe_allow_html=True)
        for i, (name, data, mime) in enumerate(atts):
            with st.container(key=f"att_row_{i}"):
                a1, a2, a3, a4, a5 = st.columns([0.45, 7, 1.4, 0.55, 0.55], vertical_alignment="center", gap="small")
                a1.markdown(file_icon(name), unsafe_allow_html=True)
                a2.markdown(f"<div class='attn'>{esc(name)}</div>", unsafe_allow_html=True)
                a3.markdown(f"<div class='atts'>{human_size(len(data))}</div>", unsafe_allow_html=True)
                a4.download_button("↓", data, file_name=name, mime=mime, key=f"att_dl_{i}_{name}", type="tertiary", help=f"Download {name}")
                a5.button("×", key=f"att_rm_{i}_{name}", type="tertiary", on_click=rm_att, args=(name,), help="Remove attachment")
        ak = f"_att_add_{st.session_state.att_n}"
        st.file_uploader("Add attachment", accept_multiple_files=True, key=ak, on_change=add_att, args=(ak,), label_visibility="collapsed")
        if st.session_state.att_removed:
            st.button("Restore generated attachments", key="att_restore", type="tertiary",
                      on_click=lambda: st.session_state.update(att_removed=[]))
        b1, b2, b3 = st.columns([1.1, 1.1, 2], vertical_alignment="center")
        mailto = f"mailto:{quote(to)}?subject={quote(subj)}&body={quote(msg[:1800])}" + (f"&cc={quote(cc)}" if cc else "")
        b1.link_button("Open in Mail", mailto, width="stretch", help="Opens your mail app with the text. Attach the files from the draft package.")
        b2.download_button("All files (.zip)", lambda: build_zip(subj, msg, atts), file_name="insite_claim_package.zip", mime="application/zip",
                           key="dl_pkg", width="stretch")
        b3.download_button("Prepare claim email →", lambda: build_eml(to, cc, subj, msg, atts), file_name="insite_claim_draft.eml",
                           mime="message/rfc822", type="primary", key="dl_eml", width="stretch")
        st.caption("The .eml draft opens in Mail or Outlook with every attachment in place, ready for you to review and send."
                   + ("" if contract else " Without a contract this is a preliminary notification, not a claim for a specific amount."))


def page_report():
    section("Claim report", "Evidence, policy and a ready-to-send claim.")
    if not need_results("report"):
        return
    results = CTX["results"]
    evidence_card(contract_and_calc()[1])
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    contract, calc, no_excl, cfp = contract_card()
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    composer_card(contract, calc, no_excl, cfp)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    table = ac.results_table(results)
    with st.container(key="card_series"):
        n_per = table.groupby("turbine").size()
        card_head("Predicted time series", f"{', '.join(f'{t}: {n} rows' for t, n in n_per.items())}. Scroll inside the table to see every row.")
        st.dataframe(table, hide_index=True, width="stretch", height=380,
                     column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm"),
                                    "turbine": "Turbine",
                                    "predicted_wind_speed_ms": st.column_config.NumberColumn("Wind (m/s)", format="%.2f"),
                                    "expected_power_kw": st.column_config.NumberColumn("Power (kW)", format="%.0f"),
                                    "energy_kwh": st.column_config.NumberColumn("Energy (kWh)", format="%.1f")})



# ------------------------------------------------------------------ presentation mode
STAGE_SUBS = {6: 3, 7: 2}
KEEP_KEYS = ["pz_ui", "pz_uic", "pz_exit", "pz_cap", "pz_float", "pz_cine", "pz_load", "pz_tech"]


def pres_go(stage, sub=0):
    st.session_state.stage, st.session_state.sub, st.session_state.stage_t = stage, sub, time.time()


def pres_next():
    s_, k = st.session_state.stage, st.session_state.sub
    if k + 1 < STAGE_SUBS.get(s_, 1):
        pres_go(s_, k + 1)
    elif s_ < 8:
        pres_go(s_ + 1)


def pres_back():
    s_, k = st.session_state.stage, st.session_state.sub
    if k > 0:
        pres_go(s_, k - 1)
    elif s_ == 5:
        pres_go(3)
    elif s_ > 1:
        pres_go(s_ - 1, STAGE_SUBS.get(s_ - 1, 1) - 1)


def pres_exit(page="Overview"):
    st.session_state["_pres_exit"] = page


def keep_p(state_key, widget_key):
    v = st.session_state.get(widget_key)
    if v is not None and v != []:
        st.session_state[state_key] = v


def on_park_p():
    v = st.session_state.get("_p_park")
    if not v:
        return
    st.session_state.park = v
    members = st.session_state.get("_park_members", {}).get(v)
    if members:
        st.session_state.targets = members


def keep_contract_p():
    f = st.session_state.get("_p_contract_up")
    st.session_state.contract_up = (f.name, f.getvalue()) if f is not None else None


def keep_uploads_p(kind):
    files = st.session_state.get("_p_up_" + kind) or []
    st.session_state["uploads_" + kind] = tuple((f.name, f.getvalue()) for f in files)


def silhouettes(fill, rotor_fill, op_far=.35):
    def t(x, y, s, cls, op):
        bl = "".join(f'<path d="M0,0 L-3,-5 L0,-120 L3,-5 Z" transform="rotate({a})" fill="{rotor_fill}"/>' for a in (0, 120, 240))
        return (f'<g transform="translate({x},{y}) scale({s})" opacity="{op}"><path d="M-5,0 L-2.5,330 L2.5,330 L5,0 Z" fill="{fill}"/>'
                f'<g class="rotor {cls}">{bl}<circle r="6" fill="{fill}"/></g></g>')
    return ('<svg class="sil" viewBox="0 0 1600 600" preserveAspectRatio="xMidYMax slice" aria-hidden="true">'
            + t(1330, 170, 1.35, "", 1) + t(380, 300, .8, "s2", op_far + .2) + t(150, 360, .55, "s3", op_far) + t(1450, 330, .6, "s2", op_far)
            + t(860, 360, .45, "s3", op_far - .1) + '</svg>')


def pz_nav(center=False):
    s_, k = st.session_state.stage, st.session_state.sub
    subs = STAGE_SUBS.get(s_, 1)
    dots = "".join(f"<i class='{'on' if j == k else ''}'></i>" for j in range(subs)) if subs > 1 else ""
    prog = f"<div class='pzprog'>{s_:02d} / 08{('<span>' + dots + '</span>') if dots else ''}</div>"
    with st.container(key="pz_uic" if center else "pz_ui", horizontal=True, gap="small"):
        if not center and (s_ > 1 or k > 0):
            st.button("← Back", key="pz_back", type="tertiary", on_click=pres_back)
        st.markdown(prog, unsafe_allow_html=True)
        if s_ == 8:
            st.button("Restart", key="pz_restart", on_click=pres_go, args=(1,))
            st.button("Finish", key="pz_finish", type="primary", on_click=pres_exit, args=("Claim report",))
        elif s_ not in (3, 4):
            st.button("Next →", key="pz_next", type="primary", on_click=pres_next)
    with st.container(key="pz_exit", horizontal=True, gap="small"):
        st.html("<button class='pzfs' onclick=\"document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()\">Full screen</button>",
                unsafe_allow_javascript=True)
        st.button("Exit", key="pz_exitbtn", type="tertiary", on_click=pres_exit)


def pz_caption(html_text):
    with st.container(key="pz_cap"):
        st.markdown(f"<div class='pzcap'>{html_text}</div>", unsafe_allow_html=True)


def spotlight(target):
    t = target or "pz_none"
    keep = "".join(f":not(.st-key-{k} *)" for k in KEEP_KEYS)
    focus = (f".stMain .st-key-{t} {{ position:relative; z-index:20; transform:scale(1.03); }} .st-key-{t} .kpi {{ box-shadow:0 24px 60px rgba(16,24,40,.16); }}"
             if t == "kpiwrap" else
             f".stMain .st-key-{t} {{ position:relative; z-index:20; transform:scale(1.02); box-shadow:0 30px 80px rgba(16,24,40,.18), 0 0 0 2px rgba(0,113,227,.35) !important; }}")
    st.html(f"""<style>
.stApp {{ background: linear-gradient(180deg,#eef1f6,#e4e8ef) !important; }}
[data-testid="stSpinner"] {{ display:none !important; }}
.stMain div[class*="st-key-card"], .stMain .st-key-kpiwrap {{ transition: opacity .45s ease, filter .45s ease, transform .5s cubic-bezier(.2,.8,.2,1), box-shadow .5s ease; }}
.stMain div[class*="st-key-card"]:not(.st-key-{t}), .stMain .st-key-kpiwrap:not(.st-key-{t}) {{ opacity:.22; filter:blur(4px) saturate(.7); pointer-events:none; }}
.stMain [data-testid="stElementContainer"]:not(div[class*="st-key-card"] *):not(.st-key-kpiwrap *):not(.st-key-{t} *){keep} {{ opacity:.22; filter:blur(4px); }}
{focus}
</style>""")


def pz_scroll(target):
    token = f"{st.session_state.stage}.{st.session_state.sub}.{target}"
    st.html(f"""<script>/* {token} */
(function(){{
  if (!window.__insiteKeys) {{
    window.__insiteKeys = true;
    document.addEventListener('keydown', function(e) {{
      const a = document.activeElement; if (a && ['INPUT','TEXTAREA','SELECT'].includes(a.tagName)) return;
      const btns = Array.from(document.querySelectorAll('.st-key-pz_ui button, .st-key-pz_uic button, .st-key-pz_exit button, .st-key-pz_float button'));
      const hit = (labels) => {{ const b = btns.find(x => labels.includes(x.innerText.trim())); if (b) {{ b.click(); e.preventDefault(); }} }};
      if (e.key === 'ArrowRight') hit(['Next →', 'Analyze →', 'Finish']);
      else if (e.key === 'ArrowLeft') hit(['← Back']);
      else if (e.key === 'Escape') hit(['Exit']);
    }});
  }}
  const sel = {json.dumps('.st-key-' + target) if target else 'null'};
  if (!sel) {{ window.scrollTo({{top: 0, behavior: 'smooth'}}); return; }}
  let n = 0;
  const go = () => {{ const el = document.querySelector(sel); if (el) {{ const r = el.getBoundingClientRect(); const tall = r.height > window.innerHeight * 0.78; const y = window.scrollY + r.top - (tall ? 84 : (window.innerHeight - r.height) / 2); window.scrollTo({{top: Math.max(0, y), behavior: 'smooth'}}); }} else if (n++ < 50) setTimeout(go, 100); }};
  setTimeout(go, 250);
}})();
</script>""", unsafe_allow_javascript=True)


def backdrop():
    st.markdown("<div class='hero'><div class='eyebrow'>Outage energy assessment</div><h1>InSite Claim</h1>"
                "<p>Know what a wind park lost while it was offline.</p></div>", unsafe_allow_html=True)
    farms = portfolio()
    st.markdown("<div class='tiles' style='margin-top:40px'>" + "".join(
        f"<div class='tile'><div class='n'>{esc(f['name'])}</div><div class='m'>{esc(f['country'])}</div>{badge('No data connected', 'gray')}</div>"
        for f in farms[:12]) + "</div>", unsafe_allow_html=True)


def stage_cinema(n):
    if n == 1:
        body = "<h1>How much is<br>missing wind data<br><b>really worth?</b></h1>"
        sil = silhouettes("#11161d", "#11161d", .3)
        cls = "pz1"
    else:
        body = ("<h1>Calculate it. Prove it. <b>Claim it.</b></h1>"
                "<p>InSite Claim — from missing data to defensible claims.</p>")
        sil = silhouettes("#3d4652", "#3d4652", .35)
        cls = "pz2"
    with st.container(key="pz_cine"):
        st.markdown(f"<div class='pzc {cls}'>{sil}<div style='position:relative;z-index:2;margin-top:-6vh'>{body}</div></div>", unsafe_allow_html=True)
    backdrop()
    pz_nav(center=True)
    pz_scroll(None)


ICONS = {
    "turbine": "<path d='M12 13v8'/><path d='M12 11 6 4'/><path d='m12 11 7-2'/><path d='m12 11-1 7'/><circle cx='12' cy='11' r='1.6'/>",
    "doc": "<path d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z'/><path d='M14 3v5h5'/><path d='M9 13h6M9 17h4'/>",
    "ref": "<circle cx='6' cy='12' r='2.5'/><circle cx='18' cy='6' r='2.5'/><circle cx='18' cy='18' r='2.5'/><path d='M8.3 11 15.7 7M8.3 13l7.4 4'/>",
    "shield": "<path d='M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z'/><path d='m9 12 2 2 4-4'/>",
}


def pzrow(key, icon, label):
    c = st.container(key=key)
    a, b = c.columns([0.11, 1], vertical_alignment="center")
    a.markdown(f"<div class='pzic'><svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='#1d1d1f' stroke-width='1.7' "
               f"stroke-linecap='round' stroke-linejoin='round'>{ICONS[icon]}</svg></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='pzrl'>{esc(label)}</div>", unsafe_allow_html=True)
    return b


def stage_inputs():
    backdrop()
    with st.container(key="pz_float"):
        st.markdown("<div class='eyebrow' style='color:#0071e3'>InSite Claim</div><div class='pzh'>Select the turbine. Upload the evidence.</div>"
                    "<div class='pzs'>Everything below is the live product. Change anything, then analyse.</div>", unsafe_allow_html=True)
        turbines = CTX.get("turbines") or []
        parks = sorted({park_of(t) for t in turbines})
        members = {f"Wind park {p}": [t for t in turbines if park_of(t) == p] for p in parks}
        st.session_state["_park_members"] = members
        farms = portfolio()
        options = list(members) + [f"{f['name']} · {f['country']}" for f in farms]
        tparks = {park_of(t) for t in (CTX.get("targets") or (CTX.get("det") or {}).get("targets", []))}
        default = next((f"Wind park {p}" for p in tparks), options[0] if options else None)
        if st.session_state.park not in options:
            st.session_state.park = default
        b = pzrow("pz_r1", "turbine", "Wind farm and turbines")
        c1, c2 = b.columns([1, 1.2])
        if options:
            c1.selectbox("Wind farm", options, index=options.index(st.session_state.park), key="_p_park", on_change=on_park_p, label_visibility="collapsed")
        sel = st.session_state.park
        has_data = sel in members
        if has_data:
            opts = members[sel]
            cur = [t for t in (CTX.get("targets") or []) if t in opts] or opts
            c2.multiselect("Turbines", opts, default=cur, key="_p_targets", on_change=keep_p, args=("targets", "_p_targets"), label_visibility="collapsed")
        else:
            c2.markdown(f"<div style='padding-top:8px'>{badge('No data connected', 'gray')}</div>", unsafe_allow_html=True)
        b = pzrow("pz_r2", "doc", "Operational data")
        with b:
            segmented_bound("Source", ["Example dataset", "Upload files"], "_p_source", "source", fmt=lambda x: "Hackathon sample data" if x == "Example dataset" else x)
        if st.session_state.source == "Upload files":
            u1, u2 = b.columns(2)
            u1.file_uploader("Wind data (all turbines)", type=["csv", "xlsx"], accept_multiple_files=True, key="_p_up_w", on_change=keep_uploads_p, args=("w",))
            u2.file_uploader("Power curves", type=["csv", "xlsx"], accept_multiple_files=True, key="_p_up_c", on_change=keep_uploads_p, args=("c",))
        tg = CTX.get("targets") or []
        if turbines:
            b.caption(f"Measurements for {', '.join(tg) or '—'} (affected) and {len([t for t in turbines if park_of(t) not in {park_of(x) for x in tg}])} reference turbines"
                      + (" · clearly labelled sample data" if st.session_state.source == "Example dataset" else ""))
        b = pzrow("pz_r3", "ref", "Reference wind farms")
        refparks = [p for p in parks if p not in {park_of(x) for x in tg}]
        b.markdown(" &nbsp;".join(badge(f"Wind park {p} · {len(members[f'Wind park {p}'])} turbines", "blue") for p in refparks)
                   + f" <span style='color:{TEXT2};font-size:13px'>&nbsp;best five picked automatically by correlation</span>", unsafe_allow_html=True)
        b = pzrow("pz_r4", "shield", "Insurance contract")
        with b:
            segmented_bound("Contract", ["Sample contract", "Upload contract", "No contract"], "_p_contract_src", "contract_src")
        if st.session_state.contract_src == "Upload contract":
            b.file_uploader("Contract", type=["pdf", "docx", "txt", "md"], key="_p_contract_up", on_change=keep_contract_p, label_visibility="collapsed")
        cname, cdata = contract_source()
        if cdata:
            b.markdown(f"<div style='display:flex;gap:10px;align-items:center;margin-top:6px'>{file_icon(cname)}<span style='font-size:14px'>{esc(cname)}</span>"
                       f"<span style='color:{TEXT3};font-size:13px'>{human_size(len(cdata))}</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        ok = has_data and bool(CTX.get("wide") is not None)
        st.button("Analyze →", key="pz_analyze", type="primary", width="stretch", on_click=pres_go, args=(4,), disabled=not ok)
        if not ok:
            st.caption(CTX.get("msg", ("", "Choose a wind farm with data."))[1] if CTX.get("msg") else "Choose a wind farm with connected data to analyse.")
    pz_nav()
    pz_scroll(None)


def stage_loading():
    backdrop()
    with st.container(key="pz_load"):
        st.markdown("""<div class='pzl'><div class='pzring'><svg width='132' height='132' viewBox='0 0 132 132'>
<circle cx='66' cy='66' r='54' fill='none' stroke='#eef0f4' stroke-width='10'/>
<circle class='arc' cx='66' cy='66' r='54' fill='none' stroke='#0071e3' stroke-width='10' stroke-linecap='round'/></svg><div class='num'></div></div>
<b>Analyzing loss potential…</b><div class='pzrot'><span>Reconstructing missing wind speeds</span><span>Estimating potential production</span>
<span>Preparing the evidence package</span></div></div>""", unsafe_allow_html=True)
    t0 = st.session_state.stage_t or time.time()
    build_context()
    left = 2.4 - (time.time() - t0)
    if left > 0:
        time.sleep(left)
    if CTX.get("ok"):
        pres_go(5)
        st.rerun()
    with st.container(key="pz_float"):
        t_, m_ = CTX.get("msg", ("The analysis could not run", "Check the data and try again."))
        st.markdown(f"<div class='pzh'>{esc(t_)}</div><div class='pzs'>{esc(m_)}</div>", unsafe_allow_html=True)
        st.button("Back to inputs", key="pz_retry", type="primary", on_click=pres_go, args=(3,))


def stage_tech():
    bars = [("Time shifts", -17, BLUE), ("Rolling averages", -14, BLUE), ("Wind direction", -14, BLUE), ("Turbulence", -2, BLUE),
            ("Hour of day", 1, ORANGE), ("Air pressure", 2, ORANGE), ("Temperature", 5, ORANGE)]
    rows = "".join(f"<div class='pzbar'><div>{esc(n)}</div><div class='tr'><div class='fl' style='width:{abs(v) / 17 * 100:.0f}%;background:{c}'></div></div>"
                   f"<div class='vv'>{v:+d} kW</div></div>" for n, v, c in bars)
    steps = [("5 reference turbines", "other parks, ranked by correlation over the training days"),
             ("Wind in time", "now, ±10–60 min" + (", plus a travel time learned per wind direction" if TRAVEL else "")),
             (f"{W_LIN} % linear + {W_LGBM} % trees", "ridge regression and LightGBM, trained on the 30 days before the outage"),
             ("Power curve over the wobble", "averages over the real spread of wind, fixing a 10 % under-count")]
    st_html = "".join(f"<div class='pzstep'><div class='n'>{i}</div><div><b>{esc(a)}</b><span>{esc(b)}</span></div></div>" for i, (a, b) in enumerate(steps, 1))
    with st.container(key="pz_float"):
        st.markdown("<div class='pzh'>How the wind is rebuilt — and how we chose it</div>"
                    "<div class='pzs'>Staged grid search: 65,000+ models, each tested on 84 simulated 3-day outages.</div>"
                    f"<div class='pztech'><div class='col'><h3>The model</h3>{st_html}</div>"
                    f"<div class='col'><h3>What mattered</h3>{rows}<div class='pzsmall'>Change in power error when a feature group is added. "
                    "Wind travels between parks and the link depends on direction. Temperature hurt, so it is used for air density instead.</div></div></div>",
                    unsafe_allow_html=True)


def page_present():
    if "_pres_exit" in st.session_state:
        st.switch_page(PAGES[st.session_state.pop("_pres_exit")])
    s_, k = st.session_state.stage, st.session_state.sub
    if s_ in (1, 2):
        stage_cinema(s_)
        return
    if s_ == 3:
        spotlight(None)
        stage_inputs()
        return
    if s_ == 4:
        spotlight(None)
        stage_loading()
        return
    if not CTX.get("ok"):
        pres_go(3)
        st.rerun()
    res = CTX["results"]
    if s_ in (5, 6):
        target, cap = None, ""
        if s_ == 5:
            target, cap = "kpiwrap", "Here’s the estimated loss <b>and its claim value.</b>"
        elif k == 0:
            bt = ac.band_table(res)
            tot = bt.groupby("band", observed=False).kwh.sum()
            top = str(tot.idxmax())
            target, cap = "card_bands", f"Highest estimated loss at <b>{esc(top)}</b> · {100 * tot.max() / tot.sum():.0f} % of the total"
        elif k == 1:
            top = max(res, key=lambda r: r["energy_kwh"])
            share = 100 * top["energy_kwh"] / sum(r["energy_kwh"] for r in res)
            target, cap = "card_eturb", f"Turbine <b>{esc(top['target'])}</b> most affected · {share:.0f} % of the loss"
        else:
            cap = "Reconstructed from reference parks, <b>chosen by evidence.</b>"
        spotlight(target)
        if k == 2 and s_ == 6:
            stage_tech()
        pz_caption(cap)
        page_analysis(present=True)
    else:
        if s_ == 7 and k == 0:
            target, cap = "card_evidence", "Every result is backed by <b>traceable evidence.</b>"
        elif s_ == 7:
            contract, _, _ = contract_and_calc()
            n = sum(1 for t in contract.terms.values() if t.clause) if contract else 0
            target = "card_contract"
            cap = f"Policy clauses identified · <b>{n} terms</b> matched to the claim" if contract else "Coverage review required · <b>no contract attached</b>"
        else:
            target, cap = "card_composer", "From missing data to <b>a defensible claim.</b>"
        spotlight(target)
        pz_caption(cap)
        page_report()
    pz_nav()
    pz_scroll(target)


# ------------------------------------------------------------------ app shell
PAGES = {
    "Overview": st.Page(page_overview, title="Overview", url_path="overview", default=True),
    "Analysis": st.Page(page_analysis, title="Analysis", url_path="analysis"),
    "Accuracy": st.Page(page_accuracy, title="Accuracy", url_path="accuracy"),
    "Method": st.Page(page_method, title="Method", url_path="method"),
    "Claim report": st.Page(page_report, title="Claim report", url_path="report"),
    "Present": st.Page(page_present, title="Present", url_path="present"),
}
NAV = [k for k in PAGES if k != "Present"]
pg = st.navigation(list(PAGES.values()), position="hidden")
st.html(CSS)
st.html(CSS2)


def start_presentation():
    pres_go(1)
    st.session_state["_goto"] = "Present"


def on_nav():
    st.session_state["_goto"] = st.session_state.get("nav")


goto = st.session_state.pop("_goto", None)
if goto and goto != pg.title and goto in PAGES:
    st.switch_page(PAGES[goto])
if pg.title == "Present":
    if st.session_state.stage != 4:
        build_context()
    pg.run()
else:
    with st.container(key="navbar"):
        n1, n2, n3 = st.columns([1, 2.6, 0.5], vertical_alignment="center")
        n1.markdown(f"<div class='brand'>{LOGO}{APP_NAME}</div>", unsafe_allow_html=True)
        with n2:
            st.session_state["nav"] = pg.title
            st.segmented_control("Navigation", NAV, key="nav", on_change=on_nav, label_visibility="collapsed")
        n3.button("Present", key="present_btn", type="primary", on_click=start_presentation, width="stretch")
    build_context()
    pg.run()
