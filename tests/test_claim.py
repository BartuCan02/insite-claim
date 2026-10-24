import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from claim import compute_claim, parse_contract, split_clauses  # noqa: E402

SAMPLE = ROOT / "data" / "sample_contract" / "sample_bi_policy_wind_park_T"


def test_sample_pdf_and_txt_give_the_same_terms():
    for ext in ("pdf", "txt"):
        p = SAMPLE.with_suffix(f".{ext}")
        c = parse_contract(p.name, p.read_bytes())
        t = c.terms
        assert c.preamble["policy"] == "AR-BI-2026-0417"
        assert (t["waiting_h"].value, t["waiting_h"].clause) == (12, "4.1")
        assert (t["rate_eur_kwh"].value, t["rate_eur_kwh"].clause) == (0.082, "5.1")
        assert (t["cap_eur"].value, t["cap_eur"].clause) == (250000, "5.2")
        assert (t["notice_days"].value, t["notice_days"].clause) == (14, "7.1")
        assert (t["payment_days"].value, t["payment_days"].clause) == (30, "9.1")
        assert t["method"].clause == "6.1" and t["exclusions"].clause == "8.1" and t["cover"].clause == "3.1"


def test_wrapped_number_is_not_a_new_clause():
    _, cl = split_clauses("4 Waiting period\n4.1 Each event has a waiting period of\n12 consecutive hours.\n5 Indemnity\n5.1 Rate is EUR 0.08 per kWh.")
    assert [c.num for c in cl] == ["4.1", "5.1"] and "12 consecutive hours" in cl[0].text


def test_claim_deducts_waiting_period_and_applies_rate_and_cap():
    ts = pd.date_range("2026-09-01", periods=432, freq="10min")
    results = [{"target": "T1", "timestamps": ts, "power_kw": np.full(432, 600.0), "energy_kwh": 600.0 * 72}]
    p = SAMPLE.with_suffix(".txt")
    c = parse_contract(p.name, p.read_bytes())
    calc = compute_claim(results, (ts[0], ts[-1]), c, {}, park_rel=0.17)
    assert calc["waiting_kwh"] == 600.0 * 12
    assert calc["claimable_kwh"] == 600.0 * 60
    assert abs(calc["amount"] - 600.0 * 60 * 0.082) < 1e-6 and not calc["capped"]
    assert calc["notify_by"] == pd.Timestamp("2026-09-17")
    capped = compute_claim(results, (ts[0], ts[-1]), c, {"rate_eur_kwh": 100.0}, park_rel=0.17)
    assert capped["capped"] and capped["amount"] == 250000


def test_quotes_keep_their_first_letter():
    p = SAMPLE.with_suffix(".txt")
    c = parse_contract(p.name, p.read_bytes())
    assert c.terms["cover"].quote.startswith("The Insurer indemnifies")
    assert c.terms["waiting_h"].quote.startswith("Each interruption event")
    assert c.terms["cap_eur"].quote.startswith("The maximum indemnity")
