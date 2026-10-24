"""Block hold-out validation that mimics the hidden 3-day outage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import SHOWN, TRAIN_END, TARGETS
from power import PowerCurve, energy_metrics


@dataclass
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def block_folds(days: int = 3, first: str = "2026-07-01") -> list[Fold]:
    folds = []
    start = pd.Timestamp(first)
    while start + pd.Timedelta(days=days) - pd.Timedelta(minutes=10) <= TRAIN_END:
        end = start + pd.Timedelta(days=days) - pd.Timedelta(minutes=10)
        folds.append(Fold(f"blk_{start:%m%d}", start, end))
        start = start + pd.Timedelta(days=days)
    folds.append(Fold("shown_sep", SHOWN[0], SHOWN[1]))
    return folds


def split(wide: pd.DataFrame, fold: Fold):
    idx = wide.index
    in_fold = (idx >= fold.start) & (idx <= fold.end)
    labelled = idx <= TRAIN_END  # Sep 4-9 is never used for training
    train = wide[labelled & ~in_fold]
    test = wide[in_fold]
    return train, test


def wind_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    m = y_true.notna() & y_pred.notna()
    err = (y_pred[m] - y_true[m]).to_numpy()
    return {
        "n": int(m.sum()),
        "ws_rmse": float(np.sqrt(np.mean(err**2))),
        "ws_mae": float(np.mean(np.abs(err))),
        "ws_bias": float(np.mean(err)),
    }


def evaluate(model_factory, wide: pd.DataFrame, folds: list[Fold], pc: PowerCurve) -> pd.DataFrame:
    """model_factory() -> object with fit(train_wide, target) and predict(test_wide) -> Series."""
    rows = []
    for fold in folds:
        train, test = split(wide, fold)
        for target in TARGETS:
            model = model_factory()
            model.fit(train, target)
            pred = model.predict(test)
            y = test[("wind_speed", target)]
            row = {"fold": fold.name, "target": target}
            row.update(wind_metrics(y, pred))
            row.update(energy_metrics(pc, y, pred.fillna(pred.mean())))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.Series:
    blk = res[res.fold != "shown_sep"]
    shown = res[res.fold == "shown_sep"]
    tot_true = blk.e_true_mwh.sum()
    return pd.Series(
        {
            "blk_ws_rmse": blk.ws_rmse.mean(),
            "blk_ws_mae": blk.ws_mae.mean(),
            "blk_ws_bias": blk.ws_bias.mean(),
            "blk_abs_e_err_pct": blk.e_err_pct.abs().mean(),
            "blk_e_err_pct_med": blk.e_err_pct.abs().median(),
            "blk_total_e_err_pct": 100 * (blk.e_pred_mwh.sum() - tot_true) / tot_true,
            "shown_ws_rmse": shown.ws_rmse.mean(),
            "shown_abs_e_err_pct": shown.e_err_pct.abs().mean(),
        }
    )
