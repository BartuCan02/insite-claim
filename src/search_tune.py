"""Stage D+E: model settings, then blending linear + LightGBM.
Reads the chosen feature setup from outputs/search/chosen.json."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import REFERENCES, load_wide
from fe import build_all, select
from search_lib import experiments, metrics, run_config, summary

OUT = "outputs/search"


def features(wide, cfg):
    F = build_all(wide, REFERENCES, **cfg["build"])
    return F[select(F, set(cfg["groups"]))].to_numpy()


def lgbm_grid():
    for leaves, child, (n, lr), col in itertools.product([7, 15, 31], [20, 40, 80], [(400, 0.03), (800, 0.015), (300, 0.06)], [0.5, 0.8]):
        yield dict(num_leaves=leaves, min_child_samples=child, n_estimators=n, learning_rate=lr, colsample_bytree=col)


def job(X, exps, kind, params):
    df, preds = run_config(X, exps, kind, params, return_pred=True)
    df["kind"], df["params"] = kind, json.dumps(params, sort_keys=True)
    return df, preds


def main():
    wide = load_wide()
    exps = experiments(wide)
    chosen = json.load(open(f"{OUT}/chosen.json"))
    X = {k: features(wide, chosen[k]) for k in ["ridge", "lgbm"]}
    jobs = [delayed(job)(X["lgbm"], exps, "lgbm", p) for p in lgbm_grid()]
    jobs += [delayed(job)(X["ridge"], exps, "ridge", {"alpha": a}) for a in [0.01, 0.1, 1, 10, 30, 100, 300]]
    print(f"{len(jobs)} configurations", flush=True)
    out = Parallel(n_jobs=8, verbose=5)(jobs)
    res = pd.concat([o[0] for o in out], ignore_index=True)
    res.to_csv(f"{OUT}/stageD_tune.csv", index=False)

    # best of each kind by power RMSE, then blend their predictions
    best = {}
    for kind in ["ridge", "lgbm"]:
        sub = res[res.kind == kind]
        p = sub.groupby("params").p_rmse.mean().idxmin()
        best[kind] = next(o[1] for o in out if o[0].kind.iloc[0] == kind and o[0].params.iloc[0] == p)
        print(kind, "best params", p, flush=True)
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        for e in exps:
            yp = w * best["lgbm"][e["id"]] + (1 - w) * best["ridge"][e["id"]]
            rows.append({"w_lgbm": w, "id": e["id"], "target": e["target"], **metrics(e["y"][e["te"]], yp)})
    blend = pd.DataFrame(rows)
    blend.to_csv(f"{OUT}/stageE_blend.csv", index=False)
    print(pd.DataFrame({w: summary(g) for w, g in blend.groupby("w_lgbm")}).T.round(3).to_string())


if __name__ == "__main__":
    main()
