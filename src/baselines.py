"""Simple reference-based baselines for the hidden outage period."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from data import REFERENCES, load_wide
from features import ReferenceFeatures
from power import PowerCurve
from validate import block_folds, evaluate, summarise

WS_COLS = [f"ws_{r}" for r in REFERENCES]


class FleetRatio:
    """pred = fleet-normalised reference mean * target mean."""

    def fit(self, train, target):
        self.feat = ReferenceFeatures().fit(train)
        X = self.feat.transform(train)
        y = train[("wind_speed", target)]
        m = y.notna() & X.ws_fleet.notna()
        self.scale = float((y[m] / X.ws_fleet[m]).mean())
        return self

    def predict(self, test):
        return self.feat.transform(test).ws_fleet * self.scale


class LinRefs:
    """OLS on the seven gap-filled reference wind speeds."""

    def __init__(self, cols=WS_COLS):
        self.cols = cols

    def fit(self, train, target):
        self.feat = ReferenceFeatures().fit(train)
        X = self.feat.transform(train)[self.cols]
        y = train[("wind_speed", target)]
        m = y.notna() & X.notna().all(axis=1)
        self.model = LinearRegression().fit(X[m], y[m])
        return self

    def predict(self, test):
        X = self.feat.transform(test)[self.cols]
        pred = pd.Series(np.nan, index=test.index)
        m = X.notna().all(axis=1)
        pred[m] = self.model.predict(X[m])
        return pred.clip(lower=0)


class SectorLinRefs:
    """One OLS per reference wind direction sector, global OLS as fallback."""

    def __init__(self, n_sectors=8, cols=WS_COLS, min_rows=200):
        self.n = n_sectors
        self.cols = cols
        self.min_rows = min_rows

    def _sector(self, wd):
        return (np.floor((wd + 180 / self.n) % 360 / (360 / self.n))).astype("Int64")

    def fit(self, train, target):
        self.feat = ReferenceFeatures().fit(train)
        F = self.feat.transform(train)
        y = train[("wind_speed", target)]
        m = y.notna() & F[self.cols].notna().all(axis=1) & F.wd_ref.notna()
        X, y, sec = F[m], y[m], self._sector(F.wd_ref[m])
        self.global_model = Ridge(alpha=1.0).fit(X[self.cols], y)
        self.models = {}
        for s in range(self.n):
            ms = (sec == s).to_numpy()
            if ms.sum() >= self.min_rows:
                self.models[s] = Ridge(alpha=1.0).fit(X.loc[ms, self.cols], y[ms])
        return self

    def predict(self, test):
        F = self.feat.transform(test)
        pred = pd.Series(np.nan, index=test.index)
        m = F[self.cols].notna().all(axis=1)
        pred[m] = self.global_model.predict(F.loc[m, self.cols])
        sec = self._sector(F.wd_ref)
        for s, model in self.models.items():
            ms = (m & (sec == s)).fillna(False).to_numpy()
            if ms.any():
                pred[ms] = model.predict(F.loc[ms, self.cols])
        return pred.clip(lower=0)


MODELS = {
    "fleet_ratio": FleetRatio,
    "lin_refs": LinRefs,
    "lin_O_only": lambda: LinRefs(cols=["ws_O1", "ws_O2"]),
    "lin_H_only": lambda: LinRefs(cols=[f"ws_{r}" for r in ["H1", "H2", "H3", "H4", "H5"]]),
    "sector_lin_refs": SectorLinRefs,
}


def main():
    wide = load_wide()
    pc = PowerCurve("T1")
    folds = block_folds(days=3)
    print(f"{len(folds)} folds ({len(folds)-1} three-day blocks + shown Sep window)\n")
    all_res, summary = {}, {}
    for name, factory in MODELS.items():
        res = evaluate(factory, wide, folds, pc)
        all_res[name] = res
        summary[name] = summarise(res)
    S = pd.DataFrame(summary).T
    pd.set_option("display.width", 200)
    print(S.round(3).to_string())
    best = S.blk_abs_e_err_pct.idxmin()
    print(f"\nPer-target detail for {best}:")
    r = all_res[best]
    blk = r[r.fold != "shown_sep"]
    print(blk.groupby("target")[["ws_rmse", "ws_bias", "e_err_pct"]].agg(["mean", "std"]).round(3).to_string())
    print(f"\nShown Sep window ({best}):")
    print(r[r.fold == "shown_sep"][["target", "n", "ws_rmse", "ws_bias", "e_true_mwh", "e_pred_mwh", "e_err_pct"]].round(3).to_string(index=False))
    out = pd.concat(all_res, names=["model"]).reset_index(level=0)
    out.to_csv("outputs/baseline_folds.csv", index=False)
    S.to_csv("outputs/baseline_summary.csv")


if __name__ == "__main__":
    main()
