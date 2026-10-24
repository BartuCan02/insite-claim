"""Outage energy-loss estimator. Run: uv run streamlit run app/app.py"""
from __future__ import annotations

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

st.set_page_config(page_title="Outage Energy Loss Estimator", page_icon="🌬️", layout="wide")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
GREY = "#9a988f"
Z = {"80": 1.2816, "90": 1.6449}


def kwh(x):
    return f"{x:,.0f} kWh"


# ---------------- sidebar: data + settings ----------------
with st.sidebar:
    st.header("1 · Data")
    wind_files = st.file_uploader("Wind data, one file per turbine (CSV or Excel)", type=["csv", "xlsx"], accept_multiple_files=True,
                                  help="Columns: turbine_name, timestamp, wind_speed. Optional: wind_direction, ambient_temperature, air_pressure, wind_speed_stddev, test_set.")
    curve_files = st.file_uploader("Power curves of the affected turbines", type=["csv", "xlsx"], accept_multiple_files=True,
                                   help="Columns: turbine_name, wind_speed (m/s), power (kW).")
    use_demo = st.toggle("Use the hackathon example data", value=not wind_files)

    st.header("2 · Settings")
    window_days = st.slider("Training window before the outage (days)", 7, 60, RECIPE["training_window_days"],
                            help="The model learns the link between the parks only from this period before the outage.")
    level = st.radio("Confidence level of the range", ["80", "90"], index=1, horizontal=True, format_func=lambda x: f"{x} %")
    density = st.toggle("Air density correction", value=False,
                        help="Adjusts the power curve for air density from temperature and pressure (IEC 61400-12). "
                             "Changes the result by about 2 %. Off = plain power curve.")
    min_gap_h = st.number_input("Treat gaps longer than (hours) as an outage", 1.0, 48.0, 6.0, 1.0)


@st.cache_data(show_spinner=False)
def load(files: tuple[tuple[str, bytes], ...], curves: tuple[tuple[str, bytes], ...]):
    long = ac.parse_wind_files(list(files))
    return ac.to_wide(long), ac.test_flags(long), ac.parse_power_curves(list(curves))


def demo_files():
    raw = ROOT / "data" / "raw"
    return (tuple((p.name, p.read_bytes()) for p in sorted(raw.glob("*_data.xlsx"))),
            tuple((p.name, p.read_bytes()) for p in sorted(raw.glob("*_power_curve.xlsx"))))


if use_demo:
    wf, cf = demo_files()
else:
    wf = tuple((f.name, f.getvalue()) for f in wind_files or [])
    cf = tuple((f.name, f.getvalue()) for f in curve_files or [])

st.title("Outage energy loss estimator")
st.caption("Reconstructs the wind at a wind park that lost communication, from neighbouring parks, and turns it into lost energy in kWh.")

if not wf:
    st.info("Upload wind data for the affected park and for neighbouring reference parks in the sidebar, or switch on the example data.")
    st.markdown("""
**How it works**
1. Upload one file per turbine. Turbine names start with a park letter, e.g. `T1`, `H3`, `O2`.
2. The outage is found automatically from missing data; you can adjust it.
3. A model learns how the affected turbines' wind relates to the other parks in the days before the outage.
4. It rebuilds the missing wind, applies the power curve, and reports the lost energy with an uncertainty range.
""")
    st.stop()

try:
    with st.spinner("Reading files…"):
        wide, flags, curves = load(wf, cf)
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't read the files: {e}")
    st.stop()

turbines = list(wide["wind_speed"].columns)
parks = sorted({park_of(t) for t in turbines})

# ---------------- outage ----------------
st.subheader("Outage period")
det = ac.detect_outage(wide, min_hours=min_gap_h)
if not det:
    st.warning("No gap long enough to be an outage was found. Lower the gap threshold or check the data.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c2:
    targets = st.multiselect("Affected turbines", turbines, default=det["targets"],
                             help="Detected automatically: turbines that share the longest data gap.")
    if not targets:
        st.stop()
    ref_pool = [t for t in turbines if park_of(t) not in {park_of(x) for x in targets}]
    st.caption(f"Reference pool (other parks): {', '.join(ref_pool)}")
    missing_curve = [t for t in targets if t not in curves]
    if not curves:
        st.error("Upload power curves for the affected turbines.")
        st.stop()
    if missing_curve:
        st.warning(f"No power curve for {', '.join(missing_curve)}; using {next(iter(curves))}'s.")
with c1:
    t0, t1 = wide.index.min().to_pydatetime(), wide.index.max().to_pydatetime()
    start, end = st.slider("Outage window (detected automatically, drag to adjust)", min_value=t0, max_value=t1,
                           value=(det["start"].to_pydatetime(), det["end"].to_pydatetime()),
                           step=pd.Timedelta(minutes=10).to_pytimedelta(), format="DD MMM YYYY HH:mm")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hours = (end - start).total_seconds() / 3600 + 1 / 6
    tr_from = start - pd.Timedelta(days=window_days)
    st.markdown(f"**{start:%d %b %Y %H:%M} → {end + pd.Timedelta(minutes=10):%d %b %Y %H:%M}** · {hours:.0f} hours · "
                f"training data: {tr_from:%d %b} → {start:%d %b %Y}")
    if (start, end) != (det["start"], det["end"]):
        st.caption(f"Detected: {det['start']:%d %b %H:%M} → {det['end']:%d %b %H:%M}")
    if tr_from < wide.index.min():
        st.warning("The training window reaches before the start of the data; fewer days are used.")

test_window = ac.detect_test_window(wide, flags, targets, end)

# ---------------- run ----------------
import hashlib  # noqa: E402

fp = hashlib.md5(b"".join(n.encode() + d for n, d in wf + cf)).hexdigest()
key = json.dumps([fp, str(start), str(end), targets, window_days, level, density, str(test_window)])


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(_wide, _curves, key, targets, start, end, window_days, level, density, test_window):
    return ac.run_analysis(_wide, targets, (start, end), _curves, window_days, level, density, test_window)


with st.spinner(f"Learning from the {window_days} days before the outage and rebuilding the wind…"):
    results = analyse(wide, curves, key, targets, start, end, window_days, level, density, test_window)
color = {r["target"]: COLORS[i % len(COLORS)] for i, r in enumerate(results)}

# ---------------- headline ----------------
total, lo, hi = ac.park_range(results, level)
mean_wind = np.mean(np.concatenate([r["wind_pred"] for r in results]))
cards = [
    ("Estimated energy loss", f"{total:,.0f} kWh", f"{total/1000:,.1f} MWh across {len(targets)} turbine{'s' if len(targets) > 1 else ''}"),
    (f"{level} % range", f"{lo:,.0f} – {hi:,.0f} kWh", "calibrated on 84 simulated outages"),
    ("Outage duration", f"{hours:.0f} hours", f"{start:%d %b %H:%M} → {end + pd.Timedelta(minutes=10):%d %b %H:%M}"),
    ("Mean estimated wind", f"{mean_wind:.1f} m/s", f"references: {', '.join(sorted({x for r in results for x in r['refs']}))}"),
]
st.markdown(
    "<div style='display:flex;flex-wrap:wrap;gap:12px;margin:0.5rem 0 1rem'>"
    + "".join(
        f"<div style='flex:1 1 210px;background:rgba(29,158,117,0.07);border-radius:10px;padding:14px 16px'>"
        f"<div style='font-size:0.85rem;opacity:0.75'>{a}</div>"
        f"<div style='font-size:1.6rem;font-weight:600;line-height:1.3'>{b}</div>"
        f"<div style='font-size:0.8rem;opacity:0.65'>{c}</div></div>"
        for a, b, c in cards
    )
    + "</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Wind speed", "Energy loss", "Accuracy", "Method", "Insurance email", "Download"])

# ---------------- wind ----------------
with tabs[0]:
    sel = st.radio("Turbine", [r["target"] for r in results], horizontal=True, key="wind_sel")
    r = next(x for x in results if x["target"] == sel)
    pad = pd.Timedelta(days=2)
    ctx = wide.loc[start - pad: end + pad]
    fig = go.Figure()
    fig.add_vrect(x0=start, x1=end, fillcolor="#e1e0d9", opacity=0.35, line_width=0, annotation_text="outage", annotation_position="top left")
    show_refs = st.toggle("Show reference turbines", value=False)
    if show_refs:
        for ref in r["refs"][:3]:
            fig.add_scatter(x=ctx.index, y=ctx[("wind_speed", ref)], name=f"{ref} (reference)", line=dict(color=GREY, width=1, dash="dot"), opacity=0.7)
    fig.add_scatter(x=ctx.index, y=ctx[("wind_speed", sel)], name=f"{sel} measured", line=dict(color="#444441", width=1.5))
    band = Z[level] * r["wobble_sd"]
    ts = r["timestamps"]
    fig.add_scatter(x=np.concatenate([ts, ts[::-1]]), y=np.concatenate([r["wind_pred"] + band, np.clip(r["wind_pred"] - band, 0, None)[::-1]]),
                    fill="toself", fillcolor=color[sel], opacity=0.15, line=dict(width=0), name=f"{level} % band (10-min values)", hoverinfo="skip")
    fig.add_scatter(x=ts, y=r["wind_pred"], name=f"{sel} estimated", line=dict(color=color[sel], width=2))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Wind speed (m/s)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The band shows how far a single 10-minute value typically lands from the estimate (±{band:.1f} m/s), learned from the training days.")

# ---------------- energy ----------------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lost energy per turbine**")
        f = go.Figure()
        for r in results:
            a, b = r["range_kwh"]
            f.add_bar(x=[r["target"]], y=[r["energy_kwh"] / 1000], marker_color=color[r["target"]], name=r["target"],
                      error_y=dict(type="data", symmetric=False, array=[(b - r["energy_kwh"]) / 1000], arrayminus=[(r["energy_kwh"] - a) / 1000], color="#444441"),
                      text=[f"{r['energy_kwh']/1000:,.1f} MWh"], textposition="inside", insidetextanchor="start",
                      textfont=dict(color="white"))
        f.update_layout(height=340, showlegend=False, yaxis_title="MWh", margin=dict(l=10, r=10, t=10, b=10), bargap=0.5)
        st.plotly_chart(f, width="stretch")
        st.caption(f"Whiskers: {level} % range.")
    with c2:
        st.markdown("**Cumulative loss during the outage**")
        f = go.Figure()
        cum_total = None
        for r in results:
            cum = np.cumsum(r["power_kw"]) / 6 / 1000
            cum_total = cum if cum_total is None else cum_total + cum
            f.add_scatter(x=r["timestamps"], y=cum, name=r["target"], line=dict(color=color[r["target"]], width=2))
        if len(results) > 1:
            f.add_scatter(x=results[0]["timestamps"], y=cum_total, name="Total", line=dict(color="#444441", width=2, dash="dash"))
        f.update_layout(height=340, yaxis_title="MWh", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, width="stretch")
    st.markdown("**Where the loss comes from: wind speed bands**")
    bt = ac.band_table(results)
    f = go.Figure()
    for r in results:
        d = bt[bt.turbine == r["target"]]
        f.add_bar(x=d.band.astype(str), y=d.kwh / 1000, name=r["target"], marker_color=color[r["target"]])
    f.update_layout(barmode="stack", height=320, yaxis_title="MWh", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f, width="stretch")
    agg = bt.groupby("band", observed=False).agg(hours=("hours", "sum"), kwh=("kwh", "sum"))
    agg["share of loss"] = (100 * agg.kwh / agg.kwh.sum()).round(1).astype(str) + " %"
    agg["kwh"] = agg.kwh.map(lambda x: f"{x:,.0f}")
    agg["hours"] = agg.hours.round(1)
    agg = agg.reset_index()
    agg["band"] = agg["band"].astype(str)
    st.dataframe(agg.rename(columns={"band": "wind band", "hours": "turbine-hours", "kwh": "lost energy (kWh)"}), hide_index=True, width="stretch")
    st.caption("Below 3 m/s the turbine makes no power; above 11.5 m/s it is at full power. Errors in the steep middle bands matter most.")

# ---------------- accuracy ----------------
with tabs[2]:
    if test_window and all("test_metrics" in r for r in results):
        a, b = test_window
        st.markdown(f"**Test period {a:%d %b} → {b:%d %b %Y}** · measured data exists here, so we predict it with the same model and compare.")
        rows = []
        for r in results:
            m = r["test_metrics"]
            rows.append({"turbine": r["target"], "wind RMSE (m/s)": m["wind_rmse"], "wind MAE (m/s)": m["wind_mae"],
                         "power RMSE (kW)": m["power_rmse"], "power MAE (kW)": m["power_mae"],
                         "energy measured (kWh)": m["energy_true_kwh"], "energy estimated (kWh)": m["energy_pred_kwh"],
                         "energy deviation": m["energy_dev_pct"]})
        T = pd.DataFrame(rows)
        tot_t, tot_p = T["energy measured (kWh)"].sum(), T["energy estimated (kWh)"].sum()
        T.loc[len(T)] = {"turbine": "All", **{c: T[c].mean() for c in T.columns[1:5]}, "energy measured (kWh)": tot_t,
                         "energy estimated (kWh)": tot_p, "energy deviation": 100 * (tot_p - tot_t) / tot_t}
        st.dataframe(T.style.format({c: "{:.2f}" for c in T.columns[1:3]} | {c: "{:.0f}" for c in T.columns[3:5]}
                                    | {c: "{:,.0f}" for c in T.columns[5:7]} | {"energy deviation": "{:+.1f} %"}), hide_index=True, width="stretch")
        sel = st.radio("Turbine", [r["target"] for r in results], horizontal=True, key="acc_sel")
        r = next(x for x in results if x["target"] == sel)
        x = r["extra"]["test"]
        f = go.Figure()
        f.add_scatter(x=x["timestamps"], y=x["wind_true"], name="measured", line=dict(color="#444441", width=1.5))
        f.add_scatter(x=x["timestamps"], y=x["wind_pred"], name="estimated", line=dict(color=color[sel], width=2))
        f.update_layout(height=320, yaxis_title="Wind speed (m/s)", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, width="stretch")
    else:
        st.info("No measured data after the outage to test against.")
    st.markdown("**Back-test inside the training window**")
    st.caption("A rehearsal inside the training days: the training window is cut into 3-day pieces; each piece is hidden in turn, "
               "the model is trained on the rest and predicts it. This measures how much real wind wobbles around our estimate "
               "(used for the energy correction) and works for any upload, even without measured data after the outage. "
               "Energy deviation here is before the energy correction (wind put straight through the power curve).")
    st.dataframe(pd.DataFrame([{"turbine": r["target"], "wind RMSE (m/s)": round(r["inner_metrics"]["wind_rmse"], 2),
                                "wind MAE (m/s)": round(r["inner_metrics"]["wind_mae"], 2),
                                "power RMSE (kW)": round(r["inner_metrics"]["power_rmse"]),
                                "power MAE (kW)": round(r["inner_metrics"]["power_mae"]),
                                "energy deviation": f"{r['inner_metrics']['energy_dev_pct']:+.1f} %"} for r in results]),
                 hide_index=True, width="stretch")
    st.markdown("**Validation on 84 simulated outages (July–August)**")
    st.caption("How the method was chosen: every 3-day window in July/August was hidden and rebuilt, using only the 30 days before it.")
    st.markdown("""
| setup | what it does | power RMSE | wind RMSE | total energy deviation |
|---|---|---|---|---|
| **Simple baseline** | Linear regression (ridge) on the current wind speed of all 7 reference turbines. Wind put straight through the power curve. | 677 kW | 1.19 m/s | −13.7 % |
| **This tool, without energy correction** | Top-5 references picked by correlation. Inputs: their wind now, each park's wind 10/30/60 min earlier and later, wind direction, turbulence, how much the references agree. Prediction = 60 % linear model + 40 % LightGBM (many small decision trees). Wind put straight through the power curve. | 631 kW | 1.13 m/s | −11.2 % |
| **This tool** | Same wind prediction, but the power curve is averaged over the typical wobble of real wind around the estimate (about ±1.1 m/s). This removes the systematic under-estimate of energy. | 621 kW | 1.13 m/s | −2.0 % |
""")
    st.caption("Wind RMSE is the same in the last two rows because only the step from wind to energy differs.")

# ---------------- method ----------------
with tabs[3]:
    st.markdown(f"""
**1. Pick reference turbines automatically.** For each affected turbine, every turbine in another park is ranked by how closely its wind
followed the affected turbine over the {window_days} training days (correlation). The top 5 are used.

**2. Learn the link between the parks.** Two models learn the affected turbine's wind from the references' wind, the wind 10, 30 and 60 minutes
earlier and later (wind travels between parks), wind direction, turbulence and how much the references agree. A linear model (60 %) and
gradient-boosted trees (40 %) are averaged.

**3. Rebuild the missing wind** during the outage from the references, which kept measuring.

**4. Turn wind into energy.** Real wind wobbles around any estimate, and the power curve is steep, so the curve is averaged over the typical
wobble (learned from the training days) instead of read at a single point. Without this, losses come out about 10 % too low.

**5. Put a range on it.** Back-testing on 84 simulated outages showed how far a 3-day estimate can be off: ±{100*RECIPE['energy']['range_turbine_rel'][level]:.0f} %
per turbine and ±{100*RECIPE['energy']['range_park_rel'][level]:.0f} % for a whole park at {level} % confidence.
""")
    sel = st.radio("Turbine", [r["target"] for r in results], horizontal=True, key="m_sel")
    r = next(x for x in results if x["target"] == sel)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reference ranking (correlation over the training days)**")
        rk = r["ranking"].copy()
        rk["used"] = rk.reference.isin(r["refs"])
        f = go.Figure(go.Bar(x=rk["corr"], y=rk.reference, orientation="h",
                             marker_color=[color[sel] if u else "#d3d1c7" for u in rk.used], text=rk["corr"].round(2), textposition="outside"))
        f.update_layout(height=60 + 34 * len(rk), xaxis=dict(range=[0, 1], title="correlation"), yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=30, t=10, b=10))
        st.plotly_chart(f, width="stretch")
        st.caption("Coloured = used. References from the affected park are never used.")
    with c2:
        best = r["refs"][0]
        tr_idx = r["train_idx"]
        d = wide["wind_speed"].iloc[tr_idx][[best, sel]].dropna()
        st.markdown(f"**{sel} vs its best reference {best} (training days)**")
        f = go.Figure(go.Scattergl(x=d[best], y=d[sel], mode="markers", marker=dict(size=4, color=color[sel], opacity=0.35)))
        mx = float(max(d.max().max(), 1))
        f.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False)
        f.update_layout(height=320, xaxis_title=f"{best} wind (m/s)", yaxis_title=f"{sel} wind (m/s)", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(f, width="stretch")
    st.markdown("**Correlation between all turbines (training days)**")
    C = wide["wind_speed"].iloc[r["train_idx"]].corr()
    f = go.Figure(go.Heatmap(z=C.values, x=C.columns, y=C.index, colorscale="Blues", zmin=0.5, zmax=1, text=C.round(2).values,
                             texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    f.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(f, width="stretch")
    if density and r.get("air_density") is not None:
        st.caption(f"Air density during the outage (estimated from reference parks): {np.mean(r['air_density']):.3f} kg/m³ (standard 1.225).")

# ---------------- email ----------------
with tabs[4]:
    c1, c2 = st.columns(2)
    park_name = c1.text_input("Wind park name", value=f"Wind park {'/'.join(sorted({park_of(t) for t in targets}))}")
    insurer = c2.text_input("Addressee", placeholder="Claims department")
    policy = c1.text_input("Policy number", placeholder="optional")
    sender = c2.text_input("Your name", placeholder="optional")
    text = ac.email_text(results, (start, end), level, park_name, insurer, policy, sender, density)
    edited = st.text_area("Draft email (editable)", value=text, height=460)
    st.download_button("Download email as .txt", edited, file_name="outage_claim_email.txt", mime="text/plain")

# ---------------- download ----------------
with tabs[5]:
    table = ac.results_table(results)
    st.dataframe(table.head(200), hide_index=True, width="stretch")
    st.download_button("Download predicted time series (CSV)", table.to_csv(index=False), file_name="outage_predictions.csv", mime="text/csv", type="primary")
    summary = {"outage_start": str(start), "outage_end": str(end), "level_pct": level, "air_density": density,
               "total_kwh": total, "range_kwh": [lo, hi],
               "turbines": {r["target"]: {"energy_kwh": r["energy_kwh"], "range_kwh": list(r["range_kwh"]), "references": r["refs"],
                                          **({"test_metrics": r["test_metrics"]} if "test_metrics" in r else {})} for r in results}}
    st.download_button("Download summary (JSON)", json.dumps(summary, indent=2, default=float), file_name="outage_summary.json", mime="application/json")
