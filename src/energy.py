"""Step 2: from wind guesses to expected power, energy, and an uncertainty range.

Everything is learned from the inner backtest inside the 30 training days:
pairs of (our guess, the real wind) that the model never saw while fitting.
"""
from __future__ import annotations

import numpy as np

from power import INTERVAL_HOURS, PowerCurve

BIN_EDGES = np.array([0, 3, 4.5, 6, 7.5, 9, 11, np.inf])
MIN_PER_BIN = 60
Z = np.linspace(-3, 3, 25)
ZW = np.exp(-0.5 * Z**2)
ZW /= ZW.sum()


class Wobble:
    """How far the real wind lands from our guess, depending on how big the guess is."""

    def fit(self, pred: np.ndarray, truth: np.ndarray, order: np.ndarray | None = None):
        self.resid = truth - pred
        self.pred = pred
        self.global_sd = float(np.std(self.resid))
        self.bin_of = lambda p: np.clip(np.searchsorted(BIN_EDGES, p, side="right") - 1, 0, len(BIN_EDGES) - 2)
        b = self.bin_of(pred)
        self.bin_resid, self.bin_sd = {}, {}
        for k in range(len(BIN_EDGES) - 1):
            r = self.resid[b == k]
            if len(r) >= MIN_PER_BIN:
                self.bin_resid[k], self.bin_sd[k] = r, float(np.std(r))
        # standardised residuals in time order, for simulating realistic error paths
        sd_t = self.sd_for(pred)
        srt = np.argsort(order) if order is not None else np.arange(len(pred))
        self.z_path = (self.resid / sd_t)[srt]
        return self

    def sd_for(self, p: np.ndarray) -> np.ndarray:
        b = self.bin_of(p)
        return np.array([self.bin_sd.get(k, self.global_sd) for k in b])

    def resid_for_bin(self, k: int) -> np.ndarray:
        return self.bin_resid.get(k, self.resid)


def expected_power(pc: PowerCurve, pred: np.ndarray, wob: Wobble | None, method: str) -> np.ndarray:
    if method == "none" or wob is None:
        return pc.power_kw(pred)
    if method == "gauss_global":
        return (pc.power_kw(np.clip(pred[:, None] + wob.global_sd * Z[None, :], 0, None)) * ZW).sum(axis=1)
    if method == "gauss_bin":
        sd = wob.sd_for(pred)
        return (pc.power_kw(np.clip(pred[:, None] + sd[:, None] * Z[None, :], 0, None)) * ZW).sum(axis=1)
    if method == "empirical_bin":
        out = np.empty_like(pred)
        b = wob.bin_of(pred)
        for k in np.unique(b):
            m = b == k
            r = wob.resid_for_bin(k)
            out[m] = pc.power_kw(np.clip(pred[m][:, None] + r[None, :], 0, None)).mean(axis=1)
        return out
    raise ValueError(method)


def energy_kwh(power_kw: np.ndarray) -> float:
    return float(np.sum(power_kw) * INTERVAL_HOURS)


def simulate_energy(pc: PowerCurve, pred: np.ndarray, wob: Wobble, chunk: int, n_sims: int = 1000, seed: int = 0) -> np.ndarray:
    """Plausible 'real wind' paths = our guess + error stretches copied from the
    inner backtest (in chunks, so errors that last for hours stay together)."""
    rng = np.random.default_rng(seed)
    n, zp = len(pred), wob.z_path
    sd = wob.sd_for(pred)
    n_chunks = int(np.ceil(n / chunk))
    starts = rng.integers(0, max(1, len(zp) - chunk), size=(n_sims, n_chunks))
    idx = (starts[:, :, None] + np.arange(chunk)[None, None, :]).reshape(n_sims, -1)[:, :n]
    paths = np.clip(pred[None, :] + zp[idx] * sd[None, :], 0, None)
    return pc.power_kw(paths).sum(axis=1) * INTERVAL_HOURS


def energy_interval(center: float, sims: np.ndarray, level: float) -> tuple[float, float]:
    """Range around our central estimate, shaped by the simulated spread."""
    a = (1 - level) / 2
    lo, hi = np.quantile(sims, [a, 1 - a])
    m = sims.mean()
    return center * lo / m, center * hi / m
