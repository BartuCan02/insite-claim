"""Logic behind the app, kept separate from the page so it can be tested."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from core import detect_gaps, park_of, rank_references, training_mask
from data import VARS
from pipeline import RECIPE, estimate_outage
from power import INTERVAL_HOURS, PowerCurve

BANDS = [0, 3, 5, 7, 9, 11.5, 40]
BAND_LABELS = ["0–3 m/s", "3–5 m/s", "5–7 m/s", "7–9 m/s", "9–11.5 m/s", "11.5+ m/s"]


# ---------- reading uploads ----------
def read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    raw = data.decode("utf-8-sig", errors="replace")
    sep = ";" if raw.count(";") > raw.count(",") else ","
    decimal = "," if sep == ";" else "."
    return pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal)


def parse_wind_files(files: list[tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, data in files:
        df = read_table(name, data)
        df.columns = [c.strip().lower() for c in df.columns]
        if "turbine_name" not in df.columns:
            df["turbine_name"] = Path(name).stem.split("_")[0]
        if "timestamp" not in df.columns or "wind_speed" not in df.columns:
            raise ValueError(f"{name}: needs at least 'timestamp' and 'wind_speed' columns")
        for v in VARS + ["test_set"]:
            if v not in df.columns:
                df[v] = np.nan
        frames.append(df[["turbine_name", "timestamp", *VARS, "test_set"]])
    long = pd.concat(frames, ignore_index=True)
    long["timestamp"] = pd.to_datetime(long["timestamp"]).dt.floor("10min")
    long["turbine_name"] = long["turbine_name"].astype(str).str.strip()
    for v in VARS:
        long[v] = pd.to_numeric(long[v], errors="coerce")
    # a column that never changes for a turbine is a placeholder, not a measurement
    for (t, v), s in long.groupby("turbine_name")[VARS].agg(lambda x: x.nunique(dropna=True)).stack().items():
        if s <= 1:
            long.loc[long.turbine_name == t, v] = np.nan
    return long


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot_table(index="timestamp", columns="turbine_name", values=VARS, aggfunc="mean", dropna=False)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="10min")
    wide = wide.reindex(grid).rename_axis("timestamp")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([VARS, sorted(long.turbine_name.unique())]))
    return wide


def test_flags(long: pd.DataFrame) -> pd.Series:
    t = long.dropna(subset=["test_set"])
    if t.empty:
        return pd.Series(dtype=bool)
    flag = t["test_set"].astype(str).str.lower().isin(["true", "1", "yes"])
    return flag.groupby(t["timestamp"]).any()


def parse_power_curves(files: list[tuple[str, bytes]]) -> dict[str, pd.DataFrame]:
    curves = {}
    for name, data in files:
        df = read_table(name, data)
        df.columns = [c.strip().lower() for c in df.columns]
        if "turbine_name" not in df.columns:
            df["turbine_name"] = Path(name).stem.split("_")[0]
        for t, g in df.groupby("turbine_name"):
            curves[str(t)] = g[["wind_speed", "power"]].astype(float).sort_values("wind_speed").reset_index(drop=True)
    return curves


# ---------- outage + test window ----------
def detect_outage(wide: pd.DataFrame, min_hours: float = 6) -> dict:
    """Longest gap per turbine; the affected park is the one whose turbines share the longest gap."""
    ws = wide["wind_speed"]
    gaps = {}
    for t in ws.columns:
        g = [x for x in detect_gaps(ws[t], min_steps=int(min_hours * 6))]
        if g:
            gaps[t] = max(g, key=lambda x: x.n_steps)
    if not gaps:
        return {}
    longest = max(gaps.values(), key=lambda g: g.n_steps)
    affected = [t for t, g in gaps.items() if g.start <= longest.end and g.end >= longest.start and g.hours >= 0.5 * longest.hours]
    parks = {park_of(t) for t in affected}
    start = min(gaps[t].start for t in affected)
    end = max(gaps[t].end for t in affected)
    return {"targets": sorted(affected), "parks": sorted(parks), "start": start, "end": end, "gaps": gaps}


def detect_test_window(wide: pd.DataFrame, flags: pd.Series, targets: list[str], outage_end: pd.Timestamp):
    """Flagged test rows after the outage where the targets have data; else the 6 days after the outage."""
    have = wide["wind_speed"][targets].notna().all(axis=1)
    if not flags.empty:
        f = flags.reindex(wide.index, fill_value=False) & have & (wide.index > outage_end)
        if f.any():
            return f[f].index.min(), f[f].index.max()
    after = have & (wide.index > outage_end) & (wide.index <= outage_end + pd.Timedelta(days=6))
    return (after[after].index.min(), after[after].index.max()) if after.any() else None


# ---------- analysis ----------
def _one(wide, target, outage, window_days, level, density, curve, test_window):
    pc = PowerCurve(frame=curve)
    extra = {"test": test_window} if test_window else None
    r = estimate_outage(wide, target, outage, window_days, level, density, pc=pc, extra_windows=extra)
    r["power_curve"] = curve
    tr = np.zeros(len(wide), bool)
    tr[r["train_idx"]] = True
    te = (wide.index >= outage[0]) & (wide.index <= outage[1])
    r["ranking"] = rank_references(wide["wind_speed"], target, tr, te)
    r["inner_metrics"] = metrics(pc, r["inner"]["wind_true"], r["inner"]["wind_pred"])
    if "test" in r["extra"]:
        x = r["extra"]["test"]
        r["test_metrics"] = metrics(pc, x["wind_true"], x["wind_pred"], power_pred=x["power_kw"])
    return r


def run_analysis(wide, targets, outage, curves, window_days=30, level="90", density=False, test_window=None):
    jobs = []
    for t in targets:
        curve = curves.get(t) if t in curves else next(iter(curves.values()))
        jobs.append(delayed(_one)(wide, t, outage, window_days, level, density, curve, test_window))
    return Parallel(n_jobs=min(len(jobs), 4))(jobs)


def metrics(pc: PowerCurve, y_true, y_pred, power_pred=None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    pt = pc.power_kw(y_true[m])
    pp = pc.power_kw(y_pred[m]) if power_pred is None else np.asarray(power_pred)[m]
    et, ep = pt.sum() * INTERVAL_HOURS, pp.sum() * INTERVAL_HOURS
    return {
        "n": int(m.sum()),
        "wind_rmse": float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2))),
        "wind_mae": float(np.mean(np.abs(y_pred[m] - y_true[m]))),
        "power_rmse": float(np.sqrt(np.mean((pp - pt) ** 2))),
        "power_mae": float(np.mean(np.abs(pp - pt))),
        "energy_true_kwh": float(et),
        "energy_pred_kwh": float(ep),
        "energy_dev_pct": float(100 * (ep - et) / et) if et > 0 else float("nan"),
    }


def band_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        b = pd.cut(r["wind_pred"], BANDS, labels=BAND_LABELS, right=False)
        df = pd.DataFrame({"band": b, "kwh": r["power_kw"] * INTERVAL_HOURS})
        g = df.groupby("band", observed=False)
        for band, kwh in g.kwh.sum().items():
            rows.append({"turbine": r["target"], "band": band, "kwh": kwh, "hours": g.size()[band] / 6})
    out = pd.DataFrame(rows)
    out["band"] = pd.Categorical(out["band"], categories=BAND_LABELS, ordered=True)
    return out


def park_range(results, level: str) -> tuple[float, float, float]:
    total = sum(r["energy_kwh"] for r in results)
    rel = RECIPE["energy"]["range_park_rel"][level] if len(results) > 1 else RECIPE["energy"]["range_turbine_rel"][level]
    return total, total * (1 - rel), total * (1 + rel)


def results_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        for t, w, p in zip(r["timestamps"], r["wind_pred"], r["power_kw"]):
            rows.append({"timestamp": t, "turbine": r["target"], "predicted_wind_speed_ms": round(float(w), 3),
                         "expected_power_kw": round(float(p), 1), "energy_kwh": round(float(p) * INTERVAL_HOURS, 2)})
    return pd.DataFrame(rows)


# ---------- insurance email ----------
def email_text(results, outage, level, park_name, insurer, policy, sender, density) -> str:
    total, lo, hi = park_range(results, level)
    hours = (outage[1] - outage[0]).total_seconds() / 3600 + 1 / 6
    lines = [f"Subject: Business interruption claim – energy yield loss, {park_name}, {outage[0]:%d %b %Y} to {outage[1]:%d %b %Y}", "",
             f"Dear {insurer or 'Sir or Madam'},", "",
             f"We are notifying you of a communication outage at {park_name} that interrupted feed-in "
             f"from {outage[0]:%d %B %Y, %H:%M} to {outage[1] + pd.Timedelta(minutes=10):%d %B %Y, %H:%M} ({hours:.0f} hours)."
             + (f" Policy number: {policy}." if policy else ""), "",
             f"Estimated energy yield loss: {total:,.0f} kWh ({total/1000:,.1f} MWh), "
             f"with a {level}% range of {lo:,.0f} – {hi:,.0f} kWh.", "", "Per turbine:"]
    for r in results:
        a, b = r["range_kwh"]
        lines.append(f"  • {r['target']}: {r['energy_kwh']:,.0f} kWh (range {a:,.0f} – {b:,.0f} kWh), mean estimated wind speed {np.mean(r['wind_pred']):.1f} m/s")
    refs = sorted({x for r in results for x in r["refs"]})
    lines += ["", "Method:",
              f"  • Wind speeds during the outage were reconstructed from {len(refs)} reference turbines in neighbouring wind parks ({', '.join(refs)}), "
              "selected automatically by correlation with the affected turbines.",
              f"  • The relationship was learned from the {RECIPE['training_window_days']} days before the outage (blend of a linear model and gradient-boosted trees, "
              "using wind speed, time-shifted wind speed, wind direction and turbulence of the reference parks).",
              "  • Energy was computed with the manufacturer power curve, accounting for the typical spread of real wind around the estimate."
              + (" Air density was taken into account." if density else ""),
              f"  • The {level}% range is based on back-testing the method on 84 simulated 3-day outages.", "",
              "We are happy to provide the underlying time series on request.", "", "Kind regards,", sender or "[Name]"]
    return "\n".join(lines)
