"""Step 2 validation on the same 84 fake outages, then the 4-9 Sep window."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from core import longest_gap, training_mask
from data import SHOWN, TARGETS, load_wide
from energy import Wobble, energy_interval, energy_kwh, expected_power, simulate_energy
from pipeline import BlendModel, inner_backtest
from power import PowerCurve
from search_lib import experiments

METHODS = ["none", "gauss_global", "gauss_bin", "empirical_bin"]
CHUNKS = {"10min": 1, "6h": 36, "24h": 144}
LEVELS = [0.8, 0.9]
CENTER_METHOD_FOR_SIMS = "empirical_bin"


def one(wide, case):
    pc = PowerCurve(case["target"])
    y, tr, te = case["y"], case["tr"], case["te"]
    m = BlendModel(case["target"]).fit(wide, tr)
    pred = m.predict(te)
    ip, it, io = inner_backtest(wide, case["target"], tr, m.refs)
    wob = Wobble().fit(ip, it, io)
    yt = y[te]
    p_true = pc.power_kw(yt)
    e_true = energy_kwh(p_true)
    rows = []
    for meth in METHODS:
        pp = expected_power(pc, pred, wob, meth)
        rows.append({"id": case["id"], "target": case["target"], "kind": "method", "method": meth,
                     "e_true": e_true, "e_pred": energy_kwh(pp),
                     "p_rmse": float(np.sqrt(np.mean((pp - p_true) ** 2))), "p_mae": float(np.mean(np.abs(pp - p_true))),
                     "inner_sd": wob.global_sd, "refs": ",".join(m.refs)})
    center = energy_kwh(expected_power(pc, pred, wob, CENTER_METHOD_FOR_SIMS))
    for cname, L in CHUNKS.items():
        sims = simulate_energy(pc, pred, wob, L)
        for lev in LEVELS:
            lo, hi = energy_interval(center, sims, lev)
            rows.append({"id": case["id"], "target": case["target"], "kind": "interval", "method": f"{cname}@{int(lev*100)}",
                         "e_true": e_true, "e_pred": center, "lo": lo, "hi": hi,
                         "covered": bool(lo <= e_true <= hi), "width_pct": 100 * (hi - lo) / center})
    return rows


def main():
    wide = load_wide()
    exps = experiments(wide)
    idx = wide.index
    shown = []
    for t in TARGETS:
        y = wide[("wind_speed", t)].to_numpy()
        g = longest_gap(wide[("wind_speed", t)])
        tr = np.where(training_mask(idx, g.start, 30) & ~np.isnan(y))[0]
        te = np.where((idx >= SHOWN[0]) & (idx <= SHOWN[1]) & ~np.isnan(y))[0]
        shown.append({"id": f"shown_{t}", "target": t, "y": y, "tr": tr, "te": te})
    out = Parallel(n_jobs=8, verbose=5)(delayed(one)(wide, c) for c in exps + shown)
    res = pd.DataFrame([r for rows in out for r in rows])
    res.to_csv("outputs/step2_results.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
