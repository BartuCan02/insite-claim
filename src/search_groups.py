"""Stage B: every combination of the 8 feature groups, for ridge and LightGBM.
Current wind speed of all 7 references is always included."""
from __future__ import annotations

import itertools

import pandas as pd
from joblib import Parallel, delayed

from data import REFERENCES, load_wide
from fe import GROUPS, build_all, select
from search_lib import experiments, run_config

OUT = "outputs/search"


def job(groups, F, exps, kind):
    cols = select(F, set(groups))
    df = run_config(F[cols].to_numpy(), exps, kind)
    df["groups"] = "+".join(groups) if groups else "(ws only)"
    df["n_groups"] = len(groups)
    df["n_feat"] = len(cols)
    df["kind"] = kind
    return df


def main():
    wide = load_wide()
    F = build_all(wide, REFERENCES)
    exps = experiments(wide)
    combos = [c for n in range(0, len(GROUPS) + 1) for c in itertools.combinations(GROUPS, n)]
    jobs = [delayed(job)(c, F, exps, k) for k in ["lgbm", "ridge"] for c in combos]
    print(f"{len(jobs)} configurations", flush=True)
    res = pd.concat(Parallel(n_jobs=8, verbose=5)(jobs), ignore_index=True)
    res.to_csv(f"{OUT}/stageB_groups.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
