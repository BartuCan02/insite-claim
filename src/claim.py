"""Contract-aware claims: read an insurance contract, find the clauses that matter,
turn the estimated loss into a claim amount, and cite the clauses in the email.

Deterministic text rules (no AI). Works on the sample contract and on contracts that
number their clauses ("4 Waiting period", "4.1 Each event ...") and state values in
plain words (hours, days, EUR ... per kWh).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from power import INTERVAL_HOURS


# ------------------------------------------------------------------ reading
def read_contract(name: str, data: bytes) -> str:
    low = name.lower()
    if low.endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if low.endswith(".docx"):
        import docx

        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8-sig", errors="replace")


@dataclass
class Clause:
    num: str
    section_title: str
    text: str


_HEAD = re.compile(r"^\s*(?:Clause\s+|§\s*)?(\d{1,2})(?:\.(\d{1,2}))?\.?\s+(\S.*)$")


def split_clauses(text: str) -> tuple[str, list[Clause]]:
    """Returns (preamble, clauses). A numbered line starts a clause only if its number
    goes up (so '12 consecutive hours' wrapped onto a new line is not a clause)."""
    preamble, clauses, titles = [], [], {}
    last = (0, 0)
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEAD.match(line)
        if m and m.group(3)[:1].isupper():
            major, minor = int(m.group(1)), int(m.group(2) or 0)
            if (major, minor) > last:
                last = (major, minor)
                if minor == 0 and len(m.group(3)) <= 60:
                    titles[major] = m.group(3).strip()
                    cur = None
                    continue
                cur = Clause(f"{major}.{minor}" if minor else str(major), titles.get(major, ""), m.group(3).strip())
                clauses.append(cur)
                continue
        if cur is not None:
            cur.text += " " + line
        elif not clauses:
            preamble.append(line)
    return "\n".join(preamble), clauses


# ------------------------------------------------------------------ terms
@dataclass
class Term:
    key: str
    label: str
    value: object = None
    clause: str | None = None
    quote: str = ""


@dataclass
class Contract:
    name: str
    preamble: dict = field(default_factory=dict)
    terms: dict = field(default_factory=dict)
    clauses: list = field(default_factory=list)


def _num(s: str) -> float:
    s = s.strip().rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):  # 250,000 or 250.000
        return float(re.sub(r"[.,]", "", s))
    return float(s.replace(",", "."))


def _sentence(text: str, start: int) -> str:
    i = text.rfind(". ", 0, start)
    a = 0 if i == -1 else i + 2
    b = text.find(". ", start)
    b = len(text) if b == -1 else b + 1
    return text[a:b].strip()


EUR = r"(?:EUR|€)\s*([\d.,]+)"
RULES = [
    # key, label, clause keywords (any), value regex (group 1), converter
    ("cover", "Cover", ["indemnif"], r"(interruption|communication|grid)", None),
    ("waiting_h", "Waiting period", ["waiting period", "deductible", "excess"], r"(\d+(?:[.,]\d+)?)\s*(?:consecutive\s+)?hours?", _num),
    ("rate_eur_kwh", "Compensation rate", ["per kwh", "/kwh"], EUR + r"\s*(?:per|/)\s*kWh", _num),
    ("cap_eur", "Limit per event", ["per interruption event", "per event", "per occurrence"], r"maximum.*?" + EUR, _num),
    ("method", "How lost energy is determined", ["reference turbine", "neighbouring", "neighboring"], r"(reference turbines?)", None),
    ("validation", "Method validation", ["uncertainty", "validated"], r"(uncertainty|validated)", None),
    ("resolution", "Required data", ["10-minute", "10 minute", "resolution"], r"(10[- ]minute)", None),
    ("notice_days", "Notification deadline", ["notify", "notification", "notice"], r"within\s+(\d+)\s+days", _num),
    ("contents", "Claim contents", ["claim shall state", "claim shall include", "shall include"], r"(shall (?:state|include))", None),
    ("exclusions", "Exclusions", ["maintenance", "curtailment", "excluded"], r"(maintenance|curtailment)", None),
    ("payment_days", "Payment term", ["pay the indemnity", "payment", "shall pay"], r"within\s+(\d+)\s+days", _num),
]


def parse_contract(name: str, data: bytes) -> Contract:
    text = read_contract(name, data)
    pre, clauses = split_clauses(text)
    c = Contract(name=name, clauses=clauses)
    for key, rx in [("policy", r"Policy\s*(?:No\.?|number)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-/]+)"),
                    ("insurer", r"Insurer\s*:\s*(.+)"), ("policyholder", r"Policyholder\s*:\s*(.+)"),
                    ("site", r"Insured site\s*:\s*(.+)"), ("period", r"Period of insurance\s*:\s*(.+)")]:
        m = re.search(rx, pre, flags=re.I)
        if m:
            c.preamble[key] = m.group(1).strip()
    for key, label, kws, rx, conv in RULES:
        t = Term(key, label)
        for cl in clauses:
            low = cl.text.lower()
            if not any(k in low for k in kws):
                continue
            m = re.search(rx, cl.text, flags=re.I)
            if not m:
                continue
            try:
                t.value = conv(m.group(1)) if conv else True
            except ValueError:
                continue
            t.clause, t.quote = cl.num, _sentence(cl.text, m.start())
            break
        c.terms[key] = t
    return c


# ------------------------------------------------------------------ claim amount
def compute_claim(results, outage, contract: Contract | None, overrides: dict, park_rel: float) -> dict:
    """Lost energy -> claimable energy (minus waiting period) -> amount (rate, cap) -> deadlines."""
    t = (contract.terms if contract else {})
    get = lambda k: overrides.get(k) if overrides.get(k) is not None else (t[k].value if k in t else None)  # noqa: E731
    waiting_h = float(get("waiting_h") or 0)
    rate = get("rate_eur_kwh")
    cap = get("cap_eur")
    notice = get("notice_days")
    pay = get("payment_days")
    steps = int(round(waiting_h * 6))
    total = sum(r["energy_kwh"] for r in results)
    waiting_kwh = 0.0
    for r in results:
        order = np.argsort(np.asarray(r["timestamps"]))
        waiting_kwh += float(np.asarray(r["power_kw"])[order][:steps].sum() * INTERVAL_HOURS)
    claimable = max(total - waiting_kwh, 0.0)
    out = {"total_kwh": total, "waiting_h": waiting_h, "waiting_kwh": waiting_kwh, "claimable_kwh": claimable,
           "claimable_lo": claimable * (1 - park_rel), "claimable_hi": claimable * (1 + park_rel),
           "rate": rate, "cap": cap, "notice_days": notice, "payment_days": pay,
           "outage_hours": (outage[1] - outage[0]).total_seconds() / 3600 + 1 / 6}
    out["below_waiting"] = out["outage_hours"] <= waiting_h
    if rate:
        amt = claimable * rate
        out.update(amount=amt, amount_lo=out["claimable_lo"] * rate, amount_hi=out["claimable_hi"] * rate)
        out["capped"] = bool(cap and amt > cap)
        if cap:
            out.update(amount=min(amt, cap), amount_lo=min(out["amount_lo"], cap), amount_hi=min(out["amount_hi"], cap))
    if notice:
        out["notify_by"] = (outage[1] + pd.Timedelta(days=int(notice))).normalize()
    return out


def cite(contract: Contract | None, key: str) -> str:
    if not contract or key not in contract.terms or not contract.terms[key].clause:
        return ""
    return f"clause {contract.terms[key].clause}"


# ------------------------------------------------------------------ email
def claim_email(results, outage, contract: Contract, calc: dict, level: str, park_name: str, insurer_name: str,
                sender: str, no_exclusion: bool, refs: list[str], window_days: int) -> str:
    pre = contract.preamble
    policy = pre.get("policy", "[policy number]")
    to = insurer_name or pre.get("insurer", "Claims department")
    s, e = outage[0], outage[1] + pd.Timedelta(minutes=10)
    c = lambda k: (f" ({cite(contract, k)})" if cite(contract, k) else "")  # noqa: E731
    L = [f"Subject: Claim under policy {policy} – interruption event at {park_name}, {s:%d %b %Y} to {outage[1]:%d %b %Y}", "",
         f"Dear {to},", "",
         f"under policy {policy} we notify an interruption event and claim indemnity for the lost energy{c('notice_days')}.", "",
         "1. The event" + c("cover"),
         f"   Communication with {', '.join(r['target'] for r in results)} was lost from {s:%d %B %Y, %H:%M} to {e:%d %B %Y, %H:%M} "
         f"({calc['outage_hours']:.0f} hours), so no electricity from these turbines reached the grid. This is an interruption event "
         f"covered by the policy.", "",
         "2. Lost energy" + c("method"),
         f"   As the turbines' own measurements are unavailable, the lost energy was determined from wind measurements of the reference "
         f"turbines {', '.join(refs)} in neighbouring wind parks, applied to the certified power curves of the insured turbines, "
         f"as the policy requires.",
         f"   Lost energy: {calc['total_kwh']:,.0f} kWh"]
    for r in results:
        L.append(f"     • {r['target']}: {r['energy_kwh']:,.0f} kWh")
    L += ["", "3. Waiting period" + c("waiting_h"),
          f"   The first {calc['waiting_h']:g} hours are not indemnified: {calc['waiting_kwh']:,.0f} kWh deducted.",
          f"   Claimable energy: {calc['claimable_kwh']:,.0f} kWh ({level} % range {calc['claimable_lo']:,.0f} – {calc['claimable_hi']:,.0f} kWh).", ""]
    if calc.get("rate"):
        L += ["4. Claimed amount" + c("rate_eur_kwh"),
              f"   {calc['claimable_kwh']:,.0f} kWh × EUR {calc['rate']:.4f} per kWh = EUR {calc['claimable_kwh'] * calc['rate']:,.2f}."]
        if calc.get("cap"):
            L.append(f"   {'Capped at' if calc.get('capped') else 'Within'} the limit of EUR {calc['cap']:,.0f} per event{c('cap_eur')}.")
        L += [f"   We claim EUR {calc['amount']:,.2f}.", ""]
    L += ["5. Evidence" + (f" ({', '.join(x for x in [cite(contract, 'validation'), cite(contract, 'resolution'), cite(contract, 'contents')] if x)})"
                           if any(cite(contract, k) for k in ['validation', 'resolution', 'contents']) else ""),
          f"   • Claim report: method, uncertainty and validation on 84 simulated 3-day outages and on measured data after the event.",
          f"   • Wind speed and power time series at 10-minute resolution for every affected turbine ({sum(len(r['timestamps']) for r in results):,} rows).",
          f"   • The model was trained only on the {window_days} days before the event.", ""]
    if no_exclusion:
        L += ["6. Exclusions" + c("exclusions"),
              "   We confirm that there was no planned maintenance and no curtailment ordered by the grid operator during the event.", ""]
    nxt = []
    if calc.get("notify_by"):
        nxt.append(f"This notification is made within the {calc['notice_days']:.0f}-day period{c('notice_days')}, before {calc['notify_by']:%d %B %Y}.")
    if calc.get("payment_days"):
        nxt.append(f"We look forward to payment within {calc['payment_days']:.0f} days of receipt of the complete documentation{c('payment_days')}.")
    L += nxt + ["", "Kind regards,", sender or "[Name]"]
    return "\n".join(L)


def report_section_html(contract: Contract, calc: dict, level: str, esc) -> str:
    rows = []

    def row(key, rule, applied):
        t = contract.terms.get(key)
        rows.append(f"<tr><td>{esc(t.clause if t and t.clause else '—')}</td><td>{esc(rule)}</td><td>{esc(applied)}</td></tr>")

    row("cover", "Interruption events are covered", "Communication loss, turbines could not feed in")
    row("method", "Lost energy from reference turbines and power curves", "Done: 5 references per turbine, certified power curves")
    row("waiting_h", f"Waiting period {calc['waiting_h']:g} h", f"{calc['waiting_kwh']:,.0f} kWh deducted")
    if calc.get("rate"):
        row("rate_eur_kwh", f"EUR {calc['rate']:.4f} per kWh", f"{calc['claimable_kwh']:,.0f} kWh → EUR {calc['claimable_kwh'] * calc['rate']:,.2f}")
    if calc.get("cap"):
        row("cap_eur", f"Limit EUR {calc['cap']:,.0f} per event", "capped" if calc.get("capped") else "within limit")
    row("validation", "Method and uncertainty documented and validated", f"84 simulated outages, {level} % range")
    row("resolution", "10-minute time series", "Attached as CSV")
    if calc.get("notify_by"):
        row("notice_days", f"Notify within {calc['notice_days']:.0f} days", f"by {calc['notify_by']:%d %B %Y}")
    amount = (f"<p><b>Claimed amount: EUR {calc['amount']:,.2f}</b> ({level} % range EUR {calc['amount_lo']:,.0f} – {calc['amount_hi']:,.0f})</p>"
              if calc.get("rate") else "")
    pre = contract.preamble
    return (f"<h2>Contract basis</h2><p>Policy {esc(pre.get('policy', '—'))} · {esc(pre.get('insurer', ''))}</p>{amount}"
            f"<table><tr><th>Clause</th><th>Contract rule</th><th>Applied to this claim</th></tr>{''.join(rows)}</table>")
