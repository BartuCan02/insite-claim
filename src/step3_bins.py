"""Wind-bin analysis (organisers' hint) + slope-weighted training test.
Re-runs the final pipeline on the 84 fake outages and keeps every 10-min prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from energy import Wobble, expected_power
from pipeline import BlendModel, inner_backtest
from power import INTERVAL_HOURS, PowerCurve
from data import load_wide
from search_lib import experiments

PC = PowerCurve("T1")


def slope_weight(y):
    """How much power changes per m/s at the true wind speed (steep part = high weight)."""
    g = np.gradient(PC.p, PC.ws)
    w = np.interp(np.nan_to_num(y), PC.ws, g)
    return np.clip(w / g.max(), 0.1, None)  # floor so calm periods are not ignored entirely


def one(wide, e):
    out = []
    y = e["y"]
    for variant, sw in [("plain", None), ("slope_weighted", slope_weight(y))]:
        m = BlendModel(e["target"]).fit(wide, e["tr"], sample_weight=sw)
        pred = m.predict(e["te"])
        ip, it, io = inner_backtest(wide, e["target"], e["tr"], m.refs) if variant == "plain" else (None, None, None)
        if variant == "plain":
            wob = Wobble().fit(ip, it, io)
        pexp = expected_power(PC, pred, wob, "gauss_global")
        out.append(pd.DataFrame({"id": e["id"], "target": e["target"], "variant": variant, "t": wide.index[e["te"]],
                                 "y": y[e["te"]], "pred": pred, "p_true": PC.power_kw(y[e["te"]]), "p_exp": pexp}))
    return pd.concat(out)


def main():
    wide = load_wide()
    exps = experiments(wide)
    df = pd.concat(Parallel(n_jobs=8, verbose=2)(delayed(one)(wide, e) for e in exps))
    df.to_parquet("outputs/step3_preds.parquet")
    print("saved", len(df))


if __name__ == "__main__":
    main()
