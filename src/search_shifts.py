"""Stage C: how far to look back/forward in time, and at what level.
Usage: search_shifts.py <ridge_groups> <lgbm_groups>   (groups joined by '+', or 'none')"""
from __future__ import annotations

import sys

import pandas as pd
from joblib import Parallel, delayed

from data import REFERENCES, load_wide
from fe import build_all, select
from search_lib import experiments, run_config

OUT = "outputs/search"

SHIFT_OPTIONS = {
    "none": dict(shifts=()),
    "park_1": dict(shifts=(1,)),
    "park_1_3": dict(shifts=(1, 3)),
    "park_1_3_6": dict(shifts=(1, 3, 6)),
    "park_3_6_12": dict(shifts=(3, 6, 12)),
    "park_1_3_6_12": dict(shifts=(1, 3, 6, 12)),
    "park_1_3_6_12_18": dict(shifts=(1, 3, 6, 12, 18)),
    "park_1_2_3_4_6_9_12": dict(shifts=(1, 2, 3, 4, 6, 9, 12)),
    "park_earlier_only_1_3_6": dict(shifts=(1, 3, 6), directions=("m",)),
    "park_later_only_1_3_6": dict(shifts=(1, 3, 6), directions=("p",)),
    "turbine_1_3_6": dict(shifts=(1, 3, 6), shift_level="ref"),
    "turbine_1_3": dict(shifts=(1, 3), shift_level="ref"),
}
ROLL_OPTIONS = {
    "roll_3_6_18": (3, 6, 18),
    "roll_6_18_36": (6, 18, 36),
    "roll_3_6_18_36_72": (3, 6, 18, 36, 72),
}


def job(name, opts, rolls_name, groups, wide, exps, kind):
    F = build_all(wide, REFERENCES, rolls=ROLL_OPTIONS[rolls_name], **opts)
    g = set(groups) | ({"shift"} if opts["shifts"] else set())
    cols = select(F, g)
    df = run_config(F[cols].to_numpy(), exps, kind)
    df["shift_opt"], df["roll_opt"], df["kind"], df["n_feat"] = name, rolls_name, kind, len(cols)
    return df


def parse(s):
    return [] if s == "none" else [g for g in s.split("+") if g != "shift"]


def main():
    wide = load_wide()
    exps = experiments(wide)
    groups = {"ridge": parse(sys.argv[1]), "lgbm": parse(sys.argv[2])}
    jobs = []
    for kind in ["lgbm", "ridge"]:
        for name, opts in SHIFT_OPTIONS.items():
            rolls = ["roll_3_6_18"] if "roll" not in groups[kind] else list(ROLL_OPTIONS)
            for r in rolls:
                jobs.append(delayed(job)(name, opts, r, groups[kind], wide, exps, kind))
    print(f"{len(jobs)} configurations", flush=True)
    res = pd.concat(Parallel(n_jobs=8, verbose=5)(jobs), ignore_index=True)
    res.to_csv(f"{OUT}/stageC_shifts.csv", index=False)


if __name__ == "__main__":
    main()
