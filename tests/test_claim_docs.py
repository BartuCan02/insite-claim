import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app_core as ac  # noqa: E402
import claim_docs as cd  # noqa: E402
from claim import compute_claim, parse_contract  # noqa: E402

RAW = ROOT / "data" / "raw"
CONTRACT = ROOT / "data" / "sample_contract" / "sample_bi_policy_wind_park_T.pdf"
FILES = ["reconstructed_wind_and_power.csv", "reference_measurements.csv", "target_measurements_before_outage.csv",
         "power_curves.csv", "validation.json", "incident.json", "claim_calculation.csv", "summary.json", "README.txt",
         CONTRACT.name]


@pytest.fixture(scope="module")
def real():
    """The example data run through the real analysis once (about 15 s)."""
    long = ac.parse_wind_files([(p.name, p.read_bytes()) for p in sorted(RAW.glob("*_data.xlsx"))])
    wide = ac.to_wide(long)
    curves = ac.parse_power_curves([(p.name, p.read_bytes()) for p in sorted(RAW.glob("*_power_curve.xlsx"))])
    det = ac.detect_outage(wide)
    tw = ac.detect_test_window(wide, ac.test_flags(long), det["targets"], det["end"])
    results = ac.run_analysis(wide, det["targets"], (det["start"], det["end"]), curves, 30, "90", False, tw)
    total, lo, hi = ac.park_range(results, "90")
    contract = parse_contract(CONTRACT.name, CONTRACT.read_bytes())
    calc = compute_claim(results, (det["start"], det["end"]), contract, {}, 0.171)
    ctx = {"start": det["start"], "end": det["end"], "hours": (det["end"] - det["start"]).total_seconds() / 3600 + 1 / 6,
           "total": total, "lo": lo, "hi": hi, "test_window": tw}
    return {"wide": wide, "results": results, "ctx": ctx, "contract": contract, "calc": calc}


def pdf(real, contract=True, park_name="Wind park T"):
    return cd.report_pdf(real["results"], real["ctx"], real["contract"] if contract else None, real["calc"] if contract else None,
                         "90", park_name, "Aurora Renewables Insurance AG", "AR-BI-2026-0417", 30, 70, 30, True)


def pdf_text(data: bytes) -> tuple[str, int]:
    pages = PdfReader(io.BytesIO(data)).pages
    return "\n".join(p.extract_text() for p in pages), len(pages)


@pytest.fixture(scope="module")
def evidence(real):
    summary = {"total_kwh": real["ctx"]["total"], "outage_start": real["ctx"]["start"]}
    data = cd.evidence_zip(real["wide"], real["results"], real["ctx"], real["calc"], CONTRACT.name, CONTRACT.read_bytes(), summary)
    return zipfile.ZipFile(io.BytesIO(data))


def test_sanitizes_text_for_core_fonts():
    s = cd.latin1("a – b — c × ≥ ≤ ± € “q” ‘s’ → \U0001f600 ä")
    assert s == "a - b - c x >= <= +/- EUR \"q\" 's' ->  ä"
    s.encode("latin-1")


def test_report_pdf_has_figures_contract_and_method(real):
    data = pdf(real)
    text, pages = pdf_text(data)
    assert data.startswith(b"%PDF") and len(data) > 20_000
    assert 2 <= pages <= 3
    assert f"{real['ctx']['total']:,.0f}" in text
    assert "Energy yield loss report" in text and "InSite Claim" in text and "Wind park T" in text
    assert "Contract basis" in text and "4.1" in text and f"{real['calc']['amount']:,.2f}" in text
    assert "Accuracy check" in text and "Method" in text and "84 simulated 3-day outages" in text
    assert "Not a confirmation of coverage" in text
    for r in real["results"]:
        assert r["target"] in text and f"{r['energy_kwh']:,.0f}" in text


def test_report_pdf_without_contract_and_with_non_latin1_park_name(real):
    data = pdf(real, contract=False, park_name="Wind park T – north €")
    text, pages = pdf_text(data)
    assert data.startswith(b"%PDF") and 2 <= pages <= 3
    assert "Contract basis" not in text and "Claimed amount" not in text
    assert f"{real['ctx']['total']:,.0f}" in text and "Wind park T - north EUR" in text


def test_report_pdf_fits_many_long_named_turbines(real):
    """Six long turbine names: the KPI note is shortened to fit its tile and names are not broken mid-word."""
    big = []
    for i in range(6):
        r = dict(real["results"][i % 3], target=f"WTG-{i + 1:02d}-Nordsee")
        r["refs"] = list(r["refs"]) + ["REF-ALPHA-LONG", "REF-BRAVO-LONG"]
        big.append(r)
    ctx = dict(real["ctx"], total=sum(r["energy_kwh"] for r in big))
    calc = compute_claim(big, (ctx["start"], ctx["end"]), real["contract"], {"rate_eur_kwh": 0.12345}, 0.171)
    text, pages = pdf_text(cd.report_pdf(big, ctx, real["contract"], calc, "90", "P", "", "", 30, 70, 30, True))
    assert 2 <= pages <= 3
    assert all(r["target"] in text for r in big)
    assert "WTG-01-Nordsee, WTG-02-Nordsee" in text and "..." in text
    assert ", ".join(r["target"] for r in big) not in text
    assert f"EUR 0.12345 = EUR {calc['claimable_kwh'] * 0.12345:,.2f}" in text


def test_clip_and_legend_columns():
    pdf = cd._Report(format="A4", unit="mm")
    pdf.add_page()
    pdf.set_font(cd.FONT, "", 8)
    assert cd._clip(pdf, "T1, T2, T3", 40) == "T1, T2, T3"
    short = cd._clip(pdf, ", ".join(f"WTG-{i:02d}-Nordsee" for i in range(10)), 40)
    assert short.endswith("...") and pdf.get_string_width(short) <= 40
    assert cd._legend_cols(["T1", "T2", "T3"]) == 3
    assert 1 <= cd._legend_cols([f"WTG-{i:02d}-Nordsee" for i in range(6)]) < 6
    assert cd._legend_cols(["A" * 80, "B"]) == 1


def test_evidence_zip_holds_the_real_data(real, evidence):
    results, ctx = real["results"], real["ctx"]
    assert set(FILES) <= set(evidence.namelist())
    rec = pd.read_csv(evidence.open("reconstructed_wind_and_power.csv"))
    assert list(rec.columns) == ["timestamp", "turbine", "rebuilt_wind_speed_ms", "expected_power_kw", "energy_kwh"]
    assert len(rec) == len(results) * 432
    assert abs(rec.energy_kwh.sum() - ctx["total"]) < 1
    ref = pd.read_csv(evidence.open("reference_measurements.csv"), parse_dates=["timestamp"])
    for t in {x for r in results for x in r["refs"]}:
        assert {f"{t}_wind_speed_ms", f"{t}_wind_direction_deg"} <= set(ref.columns)
    t0 = real["wide"].index[min(r["train_idx"].min() for r in results)]
    assert ref.timestamp.min() == t0 and ref.timestamp.max() == ctx["end"]
    tgt = pd.read_csv(evidence.open("target_measurements_before_outage.csv"), parse_dates=["timestamp"])
    assert {f"{r['target']}_wind_speed_ms" for r in results} <= set(tgt.columns)
    assert tgt.timestamp.min() == t0 and tgt.timestamp.max() < ctx["start"]
    pcs = pd.read_csv(evidence.open("power_curves.csv"))
    assert set(pcs.turbine) == {r["target"] for r in results} and list(pcs.columns) == ["turbine", "wind_speed_ms", "power_kw"]
    assert evidence.read(CONTRACT.name) == CONTRACT.read_bytes()
    val = json.loads(evidence.read("validation.json"))
    assert set(val["turbines"]) == {r["target"] for r in results}
    assert all("test_metrics" in v and "inner_metrics" in v and v["wobble_sd"] > 0 for v in val["turbines"].values())
    inc = json.loads(evidence.read("incident.json"))
    assert pd.Timestamp(inc["outage_end"]) == ctx["end"] + pd.Timedelta(minutes=10)
    assert inc["references"] == {r["target"]: r["refs"] for r in results}
    assert json.loads(evidence.read("summary.json"))["total_kwh"] == pytest.approx(ctx["total"])
    assert evidence.read("claim_calculation.csv") == cd.calculation_csv(real["calc"], results)


def test_evidence_zip_without_contract_or_claim(real):
    data = cd.evidence_zip(real["wide"], real["results"], real["ctx"], None, None, None, {"x": 1})
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "claim_calculation.csv" not in names and CONTRACT.name not in names and "README.txt" in names


def test_calculation_csv_contains_amount_and_steps(real):
    calc = real["calc"]
    text = cd.calculation_csv(calc, real["results"]).decode()
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["step", "value", "unit", "basis"] and all(len(r) == 4 and r[1] for r in rows)
    steps = {r[0]: r for r in rows[1:]}
    assert steps["Claim amount"][1] == f"{calc['amount']:.2f}" and steps["Claim amount"][2] == "EUR"
    assert steps["Lost energy total"][1] == f"{calc['total_kwh']:.1f}"
    assert steps["Waiting period"][1] == "12" and steps["Notify insurer by"][1] == "2026-09-17"
    assert all(f"Lost energy {r['target']}" in steps for r in real["results"])


def test_calculation_csv_skips_missing_values(real):
    ctx = real["ctx"]
    calc = compute_claim(real["results"], (ctx["start"], ctx["end"]), None, {}, 0.171)
    steps = [r[0] for r in csv.reader(io.StringIO(cd.calculation_csv(calc, real["results"]).decode()))]
    assert "Claimable energy" in steps
    assert not {"Claim amount", "Compensation rate", "Limit per event", "Capped", "Notify insurer by"} & set(steps)
