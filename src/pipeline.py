"""The chosen recipe as one reusable pipeline (used by validation and the app).

fit on the 30 days before an outage -> predict the target's wind speed in the outage.
Also produces out-of-sample residuals inside the training window (inner backtest),
which step 2 uses to learn how much real wind wobbles around our guesses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core import candidate_references
from fe import build_all, select
from search_lib import make_model

RECIPE = json.loads((Path(__file__).resolve().parents[1] / "outputs/search/final_recipe.json").read_text())


def pick_references(wide: pd.DataFrame, target: str, tr: np.ndarray, k: int = 5) -> list[str]:
    ws = wide["wind_speed"]
    cands = candidate_references(list(ws.columns), target)
    y = ws[target].iloc[tr]
    corr = {r: ws[r].iloc[tr].corr(y) for r in cands}
    return sorted([r for r in cands if pd.notna(corr[r])], key=lambda r: -corr[r])[:k]


@dataclass
class BlendModel:
    target: str
    refs: list[str] = field(default_factory=list)

    def _X(self, wide, kind):
        cfg = RECIPE["linear" if kind == "ridge" else "lgbm"]
        F = build_all(wide, self.refs, **cfg["build"])
        return F[select(F, set(cfg["groups"]))].to_numpy()

    def fit(self, wide: pd.DataFrame, tr: np.ndarray, sample_weight: np.ndarray | None = None):
        """sample_weight: optional weight per row of the full index (e.g. to stress steep power-curve bands)."""
        y = wide[("wind_speed", self.target)].to_numpy()
        tr = tr[~np.isnan(y[tr])]
        if not self.refs:
            self.refs = pick_references(wide, self.target, tr)
        self.Xr, self.Xl = self._X(wide, "ridge"), self._X(wide, "lgbm")
        sw = None if sample_weight is None else sample_weight[tr]
        rkw = {} if sw is None else {"ridge__sample_weight": sw}
        lkw = {} if sw is None else {"sample_weight": sw}
        self.ridge = make_model("ridge", {"alpha": RECIPE["linear"]["alpha"]}).fit(self.Xr[tr], y[tr], **rkw)
        self.lgbm = make_model("lgbm", RECIPE["lgbm"]["params"]).fit(self.Xl[tr], y[tr], **lkw)
        return self

    def predict(self, te: np.ndarray) -> np.ndarray:
        w = RECIPE["blend_weight_lgbm"]
        p = w * self.lgbm.predict(self.Xl[te]) + (1 - w) * self.ridge.predict(self.Xr[te])
        return np.clip(p, 0, None)


def inner_backtest(wide: pd.DataFrame, target: str, tr: np.ndarray, refs: list[str], block_days: int = 3):
    """Hide each 3-day block of the training window in turn, fit on the rest,
    predict it. Returns time-ordered (prediction, truth) pairs never seen in fitting."""
    y = wide[("wind_speed", target)].to_numpy()
    tr = np.sort(tr)
    t0 = wide.index[tr[0]]
    block_id = ((wide.index[tr] - t0) / pd.Timedelta(days=block_days)).astype(int)
    preds, truth, order = [], [], []
    for b in np.unique(block_id):
        hold = tr[block_id == b]
        fit_idx = tr[block_id != b]
        m = BlendModel(target, refs=refs).fit(wide, fit_idx)
        ok = hold[~np.isnan(y[hold])]
        if len(ok) == 0:
            continue
        preds.append(m.predict(ok))
        truth.append(y[ok])
        order.append(ok)
    return np.concatenate(preds), np.concatenate(truth), np.concatenate(order)


def estimate_outage(wide: pd.DataFrame, target: str, outage: tuple, window_days: int = 30, level: str = "90", air_density: bool = False, pc=None, extra_windows: dict | None = None) -> dict:
    """Everything the app needs for one target and one outage period."""
    from core import training_mask
    from energy import Wobble, energy_kwh, expected_power
    from power import PowerCurve

    idx = wide.index
    y = wide[("wind_speed", target)].to_numpy()
    tr = np.where(training_mask(idx, outage[0], window_days) & ~np.isnan(y))[0]
    te = np.where((idx >= outage[0]) & (idx <= outage[1]))[0]
    model = BlendModel(target).fit(wide, tr)
    pred = model.predict(te)
    ip, it, io = inner_backtest(wide, target, tr, model.refs)
    wob = Wobble().fit(ip, it, io)
    pc = pc or PowerCurve(target)
    rho = None
    if air_density:
        from density import estimate_density, wind_factor

        all_refs = [r for r in wide["wind_speed"].columns if r[0] != target[0]]
        rho = estimate_density(wide, target, all_refs, tr, te)
        f = wind_factor(rho)
        wob.global_sd = wob.global_sd * float(f.mean())
        power = expected_power(pc, pred * f, wob, "gauss_global")
    else:
        power = expected_power(pc, pred, wob, "gauss_global")
    e = energy_kwh(power)
    rel = RECIPE["energy"]["range_turbine_rel"][level]
    extra = {}
    for name, (a, b) in (extra_windows or {}).items():
        w_idx = np.where((idx >= a) & (idx <= b))[0]
        if len(w_idx):
            wp = model.predict(w_idx)
            extra[name] = {"timestamps": idx[w_idx], "wind_pred": wp, "power_kw": expected_power(pc, wp, wob, "gauss_global"),
                           "wind_true": y[w_idx]}
    inner = {"wind_pred": ip, "wind_true": it}
    return {
        "extra": extra, "inner": inner, "train_idx": tr,
        "target": target, "refs": model.refs, "timestamps": idx[te], "wind_pred": pred,
        "power_kw": power, "wobble_sd": wob.global_sd, "air_density": rho,
        "energy_kwh": e, "range_kwh": (e * (1 - rel), e * (1 + rel)), "range_level": level,
    }
