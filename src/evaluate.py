"""Step 1 validation, following the challenge rules:
train only on the 30 days before an outage, references only from other parks.

Simulated outages: consecutive 3-day blocks in August (each has 30 days of
history before it). Final check: model trained before the real outage
(1-3 Sep) and scored on the shown test window (4-9 Sep).
"""
from __future__ import annotations

import pandas as pd

from core import longest_gap, rank_references, select_references, training_mask
from data import SHOWN, TARGETS, TRAIN_END, load_wide
from metrics import score
from models import FleetRatio, LinearRefs, SectorLinear
from power import PowerCurve

WINDOW_DAYS = 30
MODELS = {"fleet_ratio": FleetRatio, "linear": LinearRefs, "sector_linear": SectorLinear}
REF_RULES = {"all_refs": None, "top3": 3}


def simulated_outages(wide, days=3):
    first = wide.index[0] + pd.Timedelta(days=WINDOW_DAYS)
    starts = pd.date_range(first, TRAIN_END - pd.Timedelta(days=days) + pd.Timedelta(minutes=10), freq=f"{days}D")
    return [(s, s + pd.Timedelta(days=days) - pd.Timedelta(minutes=10)) for s in starts]


def run_case(wide, target, train_mask, test_mask, pc, case):
    ws = wide["wind_speed"]
    ranking = rank_references(ws, target, train_mask, test_mask)
    rows = []
    for rule, k in REF_RULES.items():
        refs = select_references(ranking, k=k)
        for mname, M in MODELS.items():
            model = M(refs).fit(wide[train_mask], target)
            pred = model.predict(wide[test_mask])
            r = score(pc, wide.loc[test_mask, ("wind_speed", target)], pred)
            rows.append({**case, "target": target, "refs": rule, "model": mname, "ref_list": ",".join(refs), **r})
    return rows


def main():
    wide = load_wide()
    pc = PowerCurve("T1")
    idx = wide.index
    rows = []
    for start, end in simulated_outages(wide):
        test = (idx >= start) & (idx <= end)
        if wide.loc[test, "wind_speed"].notna().mean().min() < 0.5:
            print(f"skip block {start:%m-%d}: under half the timestamps have data")
            continue
        train = training_mask(idx, start, WINDOW_DAYS)
        for t in TARGETS:
            rows += run_case(wide, t, train, test, pc, {"case": f"sim_{start:%m-%d}"})
    for t in TARGETS:
        gap = longest_gap(wide[("wind_speed", t)])
        train = training_mask(idx, gap.start, WINDOW_DAYS)
        test = (idx >= SHOWN[0]) & (idx <= SHOWN[1])
        rows += run_case(wide, t, train, test, pc, {"case": "shown_sep"})
    res = pd.DataFrame(rows)
    res.to_csv("outputs/step1_results.csv", index=False)

    def summarise(df):
        tot = 100 * (df.e_pred_kwh.sum() - df.e_true_kwh.sum()) / df.e_true_kwh.sum()
        return pd.Series(
            {
                "ws_rmse": df.ws_rmse.mean(),
                "ws_mae": df.ws_mae.mean(),
                "p_rmse_kw": df.p_rmse_kw.mean(),
                "p_mae_kw": df.p_mae_kw.mean(),
                "abs_e_dev_pct": df.e_dev_pct.abs().mean(),
                "total_e_dev_pct": tot,
            }
        )

    pd.set_option("display.width", 200)
    sim = res[res.case != "shown_sep"]
    print(f"Simulated 3-day outages: {sim.case.nunique()} blocks x {len(TARGETS)} targets\n")
    print(sim.groupby(["refs", "model"]).apply(summarise, include_groups=False).round(2).to_string())
    print("\nShown test window 4-9 Sep (trained on 2-31 Aug):\n")
    print(res[res.case == "shown_sep"].groupby(["refs", "model"]).apply(summarise, include_groups=False).round(2).to_string())
    print("\nAuto-detected outages and top-3 references:")
    for t in TARGETS:
        g = longest_gap(wide[("wind_speed", t)])
        tr = training_mask(idx, g.start, WINDOW_DAYS)
        te = (idx >= g.start) & (idx <= g.end)
        r = rank_references(wide["wind_speed"], t, tr, te)
        print(f"  {t}: {g.start} -> {g.end} ({g.hours:.0f} h)  ranking: " + ", ".join(f"{a} {c:.2f}" for a, c in zip(r.reference, r["corr"])))


if __name__ == "__main__":
    main()
