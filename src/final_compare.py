"""Stage F: (1) re-check reference choice with the final features,
(2) compare all contenders on the 84 fake outages,
(3) open the locked 4-9 Sep window ONCE for the final contenders."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core import longest_gap, training_mask
from data import REFERENCES, SHOWN, TARGETS, load_wide
from fe import build_all, select
from models import SectorLinear
from search_lib import experiments, make_model, metrics, paired, summary

OUT = "outputs/search"


def best_params(kind):
    d = pd.read_csv(f"{OUT}/stageD_tune.csv")
    d = d[d.kind == kind]
    return json.loads(d.groupby("params").p_rmse.mean().idxmin())


def ref_rule(rule, wide, tr_idx, y):
    if rule == "all7":
        return REFERENCES
    if rule.startswith("top"):
        k = int(rule[3:])
        ws = wide["wind_speed"][REFERENCES].iloc[tr_idx]
        corr = {r: ws[r].corr(pd.Series(y, index=ws.index)) for r in REFERENCES}
        return sorted(REFERENCES, key=lambda r: -corr[r])[:k]
    return rule.split("+")


def predict(kind, cfg, params, wide, refs, tr, te, y):
    F = build_all(wide, refs, **cfg["build"])
    X = F[select(F, set(cfg["groups"]))].to_numpy()
    m = make_model(kind, params).fit(X[tr], y[tr])
    return np.clip(m.predict(X[te]), 0, None)


def run_all(wide, cases, chosen, params, rules, w_blend):
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=8)(delayed(run_case)(wide, c, chosen, params, rules, w_blend) for c in cases)
    return pd.DataFrame([r for rows in out for r in rows])


def run_case(wide, c, chosen, params, rules, w_blend):
    rows = []
    if True:
        y, tr, te = c["y"], c["tr"], c["te"]
        yt = y[te]
        # step-1 baselines
        p_ws = predict("ridge", {"groups": [], "build": {}}, None, wide, REFERENCES, tr, te, y)
        rows.append({"id": c["id"], "target": c["target"], "model": "baseline: linear, wind speed only", "refs": "all7", **metrics(yt, p_ws)})
        sl = SectorLinear(REFERENCES).fit(wide.iloc[tr], c["target"])
        p_sl = sl.predict(wide.iloc[te]).to_numpy()
        p_sl = np.where(np.isnan(p_sl), np.nanmean(p_sl), p_sl)
        rows.append({"id": c["id"], "target": c["target"], "model": "baseline: direction-aware linear", "refs": "all7", **metrics(yt, p_sl)})
        for rule in rules:
            refs = ref_rule(rule, wide, tr, y[tr])
            pr = predict("ridge", chosen["ridge"], params["ridge"], wide, refs, tr, te, y)
            pl = predict("lgbm", chosen["lgbm"], params["lgbm"], wide, refs, tr, te, y)
            for name, p in [("linear + features", pr), ("lightgbm + features", pl), (f"blend {w_blend:.0%} lightgbm", w_blend * pl + (1 - w_blend) * pr)]:
                rows.append({"id": c["id"], "target": c["target"], "model": name, "refs": rule, **metrics(yt, p)})
    return rows


def table(df):
    return pd.DataFrame({k: summary(g) for k, g in df.groupby(["model", "refs"])}).T.sort_values("p_rmse")


def main():
    wide = load_wide()
    chosen = json.load(open(f"{OUT}/chosen.json"))
    params = {"ridge": best_params("ridge"), "lgbm": best_params("lgbm")}
    blend = pd.read_csv(f"{OUT}/stageE_blend.csv")
    w = float(blend.groupby("w_lgbm").p_rmse.mean().idxmin())
    print("params:", params, "| blend weight on lightgbm:", w)
    stageA_best = ["H1+H5+O1", "H1+H5+O1+O2", "H1+H3+H5+O1"]
    rules = ["all7", "top4", "top5", "top6"] + stageA_best

    exps = experiments(wide)
    val = run_all(wide, exps, chosen, params, rules, w)
    val.to_csv(f"{OUT}/stageF_validation.csv", index=False)
    pd.set_option("display.width", 220)
    T = table(val)
    print("\n=== 84 fake outages ===")
    print(T[["p_rmse", "ws_rmse", "ws_mae", "p_mae", "abs_e_dev_pct", "total_e_dev_pct"]].round(3).to_string())
    best_key = T.index[0]
    best = val[(val.model == best_key[0]) & (val.refs == best_key[1])]
    for other in [("baseline: linear, wind speed only", "all7"), ("baseline: direction-aware linear", "all7"), ("linear + features", "all7"), ("lightgbm + features", "all7")]:
        o = val[(val.model == other[0]) & (val.refs == other[1])]
        pb = paired(best, o)
        print(f"best {best_key} vs {other}: {pb['delta']:.1f} ± {pb['delta_se']:.1f} kW, best wins {pb['win_rate']:.0%} of experiments")

    # locked window, opened once
    idx = wide.index
    cases = []
    for t in TARGETS:
        y = wide[("wind_speed", t)].to_numpy()
        g = longest_gap(wide[("wind_speed", t)])
        tr = np.where(training_mask(idx, g.start, 30) & ~np.isnan(y))[0]
        te = np.where((idx >= SHOWN[0]) & (idx <= SHOWN[1]) & ~np.isnan(y))[0]
        cases.append({"id": f"shown_{t}", "target": t, "y": y, "tr": tr, "te": te})
    shown = run_all(wide, cases, chosen, params, ["all7", "top5"], w)
    shown.to_csv(f"{OUT}/stageF_shown.csv", index=False)
    print("\n=== 4-9 Sep test window (trained on 2-31 Aug), opened once ===")
    print(table(shown)[["p_rmse", "ws_rmse", "ws_mae", "p_mae", "abs_e_dev_pct", "total_e_dev_pct"]].round(3).to_string())


if __name__ == "__main__":
    main()
