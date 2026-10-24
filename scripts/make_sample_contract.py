"""Write the SAMPLE insurance contract used to demo contract-aware claims.

Fictional parties only. Run: uv run python scripts/make_sample_contract.py
Outputs data/sample_contract/sample_bi_policy_wind_park_T.{pdf,txt}
"""
from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_contract"

HEADER = [
    "SAMPLE CONTRACT - FICTIONAL PARTIES - FOR DEMONSTRATION ONLY",
    "",
    "BUSINESS INTERRUPTION INSURANCE POLICY - ONSHORE WIND",
    "",
    "Policy No.: AR-BI-2026-0417",
    "Insurer: Aurora Renewables Insurance AG (fictional)",
    "Policyholder: Windpark T GmbH & Co. KG (fictional)",
    "Insured site: Wind park T, turbines T1, T2 and T3 (3 x 4,500 kW)",
    "Period of insurance: 1 January 2026 to 31 December 2026",
]

CLAUSES = [
    ("1", "Definitions", [
        ("1.1", "Interruption event means any loss of grid feed-in or loss of communication between the insured turbines and the "
                "operator's monitoring system (SCADA) that prevents the insured turbines from delivering electricity to the grid."),
        ("1.2", "Lost energy means the electricity the insured turbines would have generated during an interruption event, "
                "expressed in kWh."),
    ]),
    ("2", "Insured assets", [
        ("2.1", "This policy covers wind turbines T1, T2 and T3 of wind park T, each with a rated power of 4,500 kW, "
                "including the cabling to the grid connection point."),
    ]),
    ("3", "Cover", [
        ("3.1", "The Insurer indemnifies the Policyholder for loss of revenue from lost energy caused by an interruption event, "
                "including cable damage between the wind park and the grid connection station and technical faults that "
                "prevent electricity from being fed into the grid."),
    ]),
    ("4", "Waiting period", [
        ("4.1", "Each interruption event is subject to a waiting period of 12 consecutive hours. Lost energy during the "
                "waiting period is not indemnified."),
    ]),
    ("5", "Indemnity and limits", [
        ("5.1", "Lost energy is compensated at the agreed rate of EUR 0.0820 per kWh."),
        ("5.2", "The maximum indemnity per interruption event is EUR 250,000."),
        ("5.3", "The annual aggregate limit is EUR 1,000,000."),
    ]),
    ("6", "Determination of lost energy", [
        ("6.1", "Where measurements of the insured turbines are unavailable, lost energy shall be determined from wind "
                "measurements of reference turbines in neighbouring wind parks, applied to the certified power curves of "
                "the insured turbines."),
        ("6.2", "The method used and its uncertainty shall be documented and shall have been validated on historical data."),
        ("6.3", "Supporting wind speed and power time series shall be provided at 10-minute resolution."),
    ]),
    ("7", "Claims procedure", [
        ("7.1", "The Policyholder shall notify the Insurer of an interruption event in writing within 14 days of its end."),
        ("7.2", "The claim shall state the start and end of the event, the affected turbines, the lost energy and the claimed "
                "amount, and shall include the supporting documentation under clause 6."),
    ]),
    ("8", "Exclusions", [
        ("8.1", "Lost energy is not indemnified during planned maintenance or during curtailment ordered by the grid operator."),
        ("8.2", "Losses caused by wilful misconduct of the Policyholder are excluded."),
    ]),
    ("9", "Payment", [
        ("9.1", "The Insurer shall pay the indemnity within 30 days of receiving the complete claim documentation."),
    ]),
]


def as_text() -> str:
    lines = list(HEADER) + [""]
    for num, title, subs in CLAUSES:
        lines.append(f"{num} {title}")
        for snum, text in subs:
            lines.append(f"{snum} {text}")
        lines.append("")
    lines.append("Signed for the Insurer and the Policyholder (sample - not a real agreement).")
    return "\n".join(lines)


def as_pdf(path: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(22, 20, 22)
    pdf.set_auto_page_break(True, 20)
    pdf.add_page()
    pdf.set_text_color(178, 80, 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, HEADER[0], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_text_color(29, 29, 31)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, HEADER[2], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for line in HEADER[4:]:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for num, title, subs in CLAUSES:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"{num} {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for snum, text in subs:
            pdf.multi_cell(0, 5.4, f"{snum} {text}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        pdf.ln(1)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 115)
    pdf.cell(0, 6, "Signed for the Insurer and the Policyholder (sample - not a real agreement).", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_bi_policy_wind_park_T.txt").write_text(as_text())
    as_pdf(OUT / "sample_bi_policy_wind_park_T.pdf")
    print("written to", OUT)
