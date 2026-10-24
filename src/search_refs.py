"""Stage A: which reference turbines? All 127 subsets + automatic top-k rules."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import REFERENCES, load_wide
from fe import build_all
from search_lib import experiments, make_model, metrics, run_config, summary

OUT = "outputs/search"


def subset_job(subset, F, exps, kind):
    cols = [f"ws__{r}__now" for r in subset]
    df = run_config(F[cols].to_numpy(), exps, kind)
    df["subset"] = "+".join(subset)
    df["kind"] = kind
    return df


def topk_job(k, F, exps, kind, ws):
    """Rank refs by correlation inside each experiment's own training window."""
    rows = []
    for e in exps:
        y = e["y"][e["tr"]]
        corr = {r: np.corrcoef(ws[r].to_numpy()[e["tr"]], y)[0, 1] for r in REFERENCES}
        chosen = sorted(REFERENCES, key=lambda r: -corr[r])[:k]
        X = F[[f"ws__{r}__now" for r in chosen]].to_numpy()
        m = make_model(kind).fit(X[e["tr"]], y)
        yp = np.clip(m.predict(X[e["te"]]), 0, None)
        rows.append({"id": e["id"], "target": e["target"], "chosen": "+".join(chosen), **metrics(e["y"][e["te"]], yp)})
    df = pd.DataFrame(rows)
    df["subset"] = f"auto_top{k}"
    df["kind"] = kind
    return df


def main():
    wide = load_wide()
    F = build_all(wide, REFERENCES)
    ws = F[[f"ws__{r}__now" for r in REFERENCES]].set_axis(REFERENCES, axis=1)
    exps = experiments(wide)
    subsets = [s for n in range(1, 8) for s in itertools.combinations(REFERENCES, n)]
    jobs = [delayed(subset_job)(s, F, exps, kind) for kind in ["ridge", "lgbm"] for s in subsets]
    jobs += [delayed(topk_job)(k, F, exps, kind, ws) for kind in ["ridge", "lgbm"] for k in range(1, 8)]
    print(f"{len(jobs)} configurations x {len(exps)} experiments")
    res = pd.concat(Parallel(n_jobs=8, verbose=0)(jobs), ignore_index=True)
    res.to_csv(f"{OUT}/stageA_refs.csv", index=False)


if __name__ == "__main__":
    main()
