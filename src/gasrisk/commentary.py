"""Stage 18 (optional): LLM risk commentary with every number checked against the source tables.

Usage::

    python -m gasrisk.commentary            # uses the Claude API if the anthropic SDK and credentials are available
    python -m gasrisk.commentary --template # deterministic template, no API call

1. A fact sheet is built from the day's output tables: each figure with its exact printed form
   (``outputs/stage18_fact_sheet.json``).
2. The model (``claude-opus-5-5``) writes a short commentary using only those printed figures.
3. Every number token in the text must match a printed figure exactly (or be a date, a confidence label (95%, 97.5%, 99%) or a small
   integer such as a day count). A commentary with any unverified number is rejected and the deterministic template is published
   instead; the rejected draft and its unverified numbers are kept for review.

The ``anthropic`` package is optional and not in requirements.txt (``pip install anthropic`` to enable the LLM path).
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from gasrisk.config import Config, load_config
from gasrisk.doccheck import NUM, _allowed, strip_non_prose

MODEL = "claude-opus-5-5"
# Methodology labels (confidence levels), not results: allowed without a fact.
LABELS = {"95%", "97.5%", "99%"}
SYSTEM = (
    "You are a market risk analyst writing the daily risk commentary for a Polish gas supplier's management. "
    "Write 120 to 200 words of plain prose, no headings or bullet points, for non-specialists. "
    "Use only figures from the fact sheet, copied exactly as they appear in its 'printed' field, including separators "
    "and units. Do not compute, round or convert any number, and do not introduce any number that is not in the fact "
    "sheet. State that the book is synthetic. Cover market risk against its limit, limit breaches, stress, margin "
    "liquidity and counterparty credit."
)


def fact_sheet(cfg: Config) -> dict[str, dict]:
    out = cfg.path("outputs_dir")
    off = pd.read_csv(out / "stage7_official_method.csv")["method"].iloc[0]
    v = pd.read_csv(out / "stage5_var_today.csv").set_index("method").loc[off]
    lim = pd.read_csv(out / "stage7_limits_today.csv").set_index("limit")["utilisation"]
    st = pd.concat([pd.read_csv(out / "stage8_historical.csv"), pd.read_csv(out / "stage8_hypothetical.csv")], ignore_index=True)
    w = st.loc[st["pnl_total"].idxmin()]
    m = pd.read_csv(out / "stage12_summary.csv").iloc[0]
    ms = pd.read_csv(out / "stage12_stress.csv")
    mw = ms.loc[ms["margin_call_eur"].idxmax()]
    cl = pd.read_csv(out / "stage13_limits.csv")
    br = cl[cl["status"] == "breach"]
    tl = pd.read_csv(out / "stage6_traffic_light.csv").set_index("method").loc[off]
    facts = {
        "as_of": (str(v["as_of"]), str(v["as_of"])),
        "official_var99_pln": (v["var_99"], f"{v['var_99']:,.0f} PLN"),
        "official_es975_pln": (v["es_97.5"], f"{v['es_97.5']:,.0f} PLN"),
        "var_limit_utilisation": (lim["99% VaR (PLN)"], f"{lim['99% VaR (PLN)']:.1%}"),
        "open_gas_limit_utilisation": (lim["open gas exposure (PLN)"], f"{lim['open gas exposure (PLN)']:.1%}"),
        "fx_open_limit_utilisation": (lim["FX net open position (EUR)"], f"{lim['FX net open position (EUR)']:.1%}"),
        "backtest_zone": (tl["latest_zone"], str(tl["latest_zone"])),
        "worst_stress_scenario": (w["scenario"], str(w["scenario"])),
        "worst_stress_loss_pln": (-w["pnl_total"], f"{-w['pnl_total']:,.0f} PLN"),
        "worst_stress_x_var": (w["loss_over_var99"], f"{w['loss_over_var99']:.1f} times"),
        "initial_margin_eur": (m["im_today_fhs_eur"], f"{m['im_today_fhs_eur']:,.0f} EUR"),
        "liquidity_buffer_eur": (m["buffer_10d_q99_eur"], f"{m['buffer_10d_q99_eur']:,.0f} EUR"),
        "buffer_share_of_credit_line": (m["buffer_over_credit_line"], f"{m['buffer_over_credit_line']:.1%}"),
        "worst_margin_call_eur": (mw["margin_call_eur"], f"{mw['margin_call_eur']:,.0f} EUR"),
        "worst_margin_call_share_of_line": (mw["credit_line_utilisation"], f"{mw['credit_line_utilisation']:.1%}"),
        "counterparties_in_breach": (len(br), str(len(br))),
        "worst_counterparty": (br["counterparty"].iloc[0] if len(br) else "none", br["counterparty"].iloc[0] if len(br) else "none"),
        "worst_counterparty_utilisation": (br["utilisation"].iloc[0] if len(br) else 0.0,
                                           f"{br['utilisation'].iloc[0]:.2f} times its limit" if len(br) else "none"),
    }
    return {k: {"value": (float(a) if isinstance(a, (int, float)) else a), "printed": b} for k, (a, b) in facts.items()}


def check_numbers(text: str, facts: dict[str, dict]) -> list[str]:
    """Number tokens in ``text`` that do not appear exactly in a printed fact (dates, integers up to 31 and confidence labels allowed)."""
    printed = " | ".join(f["printed"] for f in facts.values())
    bad = []
    for m in NUM.finditer(strip_non_prose(text)):
        tok = m.group(0).rstrip(",").lstrip("+")
        if _allowed(tok) or tok in LABELS:
            continue
        if not re.search(r"(?<![\d.,])" + re.escape(tok) + r"(?![\d])", printed):
            bad.append(tok)
    return bad


def template(f: dict[str, dict]) -> str:
    p = {k: v["printed"] for k, v in f.items()}
    breach = (f"Counterparty {p['worst_counterparty']} uses {p['worst_counterparty_utilisation']}."
              if f["counterparties_in_breach"]["value"] else "No counterparty is in breach.")
    return (f"On {p['as_of']} the synthetic book's official one-day 99% value at risk is {p['official_var99_pln']} "
            f"(expected shortfall {p['official_es975_pln']}), {p['var_limit_utilisation']} of its limit; the backtest is in the "
            f"{p['backtest_zone']} zone. Rising gas prices keep the open gas exposure at {p['open_gas_limit_utilisation']} and "
            f"the FX open position at {p['fx_open_limit_utilisation']} of their limits. The worst stress scenario, the "
            f"{p['worst_stress_scenario']}, would lose {p['worst_stress_loss_pln']}, {p['worst_stress_x_var']} the value at risk, "
            f"because the Dutch TTF hedge does not follow Polish day-ahead spikes. Exchange initial margin is "
            f"{p['initial_margin_eur']} and the 10-day liquidity buffer {p['liquidity_buffer_eur']} "
            f"({p['buffer_share_of_credit_line']} of the bank line); a sharp TTF fall would call {p['worst_margin_call_eur']}, "
            f"{p['worst_margin_call_share_of_line']} of the line. {breach}")


def llm_commentary(facts: dict[str, dict]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium"},
        system=SYSTEM,
        messages=[{"role": "user", "content": "Fact sheet (JSON):\n" + json.dumps(facts, indent=2)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run(cfg: Config, use_llm: bool = True) -> dict:
    out = cfg.path("outputs_dir")
    facts = fact_sheet(cfg)
    (out / "stage18_fact_sheet.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    source, draft, bad, note = "template", None, [], ""
    if use_llm:
        try:
            draft = llm_commentary(facts)
            bad = check_numbers(draft, facts)
            if bad:
                note = f"LLM draft rejected: unverified numbers {', '.join(bad)}"
                (out / "stage18_rejected_draft.md").write_text(draft + f"\n\nUnverified: {bad}\n", encoding="utf-8")
            else:
                source = f"LLM ({MODEL})"
        except Exception as exc:  # noqa: BLE001 - SDK missing, no credentials, network or refusal
            note = f"LLM not used: {type(exc).__name__}: {str(exc)[:120]}"
    text = draft if source.startswith("LLM") else template(facts)
    final_bad = check_numbers(text, facts)
    if final_bad:
        raise AssertionError(f"published commentary has unverified numbers: {final_bad}")
    doc = (f"# Daily risk commentary\n\nSource: {source}. Every number checked against "
           f"`outputs/stage18_fact_sheet.json` (built from the output tables): 0 unverified.{' ' + note if note else ''}\n\n"
           f"{text}\n")
    (out / "stage18_commentary.md").write_text(doc, encoding="utf-8")
    return {"source": source, "note": note, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    r = run(load_config(), use_llm=not ap.parse_args().template)
    print(r["source"], r["note"])
    print(r["text"])


if __name__ == "__main__":
    main()
