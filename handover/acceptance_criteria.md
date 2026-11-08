# Acceptance criteria

The receiving team accepts a daily run when every **error** criterion passes. `python handover/validate_outputs.py`
evaluates them all and writes `outputs/validation_report.csv` (one row per check, with the detail). Thresholds are
tolerances on numerical identities or regulatory test levels, not tuned to the current results.

## Input criteria (`handover/input_schema.yaml`)

| Id | Criterion | Severity |
|---|---|---|
| IN-file | Every input file exists in `data/clean` | error |
| IN-rows | The panel has at least 1500 rows | error |
| IN-columns | Required columns present, numeric, within their plausible range, null share within limit | error |
| IN-date | Dates valid, unique and increasing | error |
| IN-01 | TGE in EUR equals TGE in PLN divided by EUR/PLN | error |
| IN-02 | The Stage 1 data quality log exists | error |
| IN-03 | No calendar gap above 8 days in the panel | error |
| IN-staleness | The last panel date is within 7 calendar days of the run date | warning |
| CFG | Book sizes, hedge ratios and the amber threshold within their allowed range | error |

## Output criteria

| Id | Area | Criterion | Severity |
|---|---|---|---|
| AC-01 | Completeness | Every daily output table exists | error |
| AC-02 | Freshness | VaR as-of date equals the last panel date | error |
| AC-03 | VaR | VaR and ES positive and finite for every method | error |
| AC-04 | VaR | VaR 99% at least VaR 95%; ES 97.5% at least 0.9 x VaR 99% | error |
| AC-05 | Backtest | The official method passes Christoffersen conditional coverage at 95% and 99% (p above 0.05) | error |
| AC-06 | Backtest | The official method is not in the red Basel traffic-light zone | error |
| AC-07 | Limits | Utilisation equals measure divided by limit | error |
| AC-08 | Attribution | Components sum to total P&L | error |
| AC-09 | Pricing | Put-call parity error below 1e-10 of the futures price | error |
| AC-10 | Pricing | Published reference examples reproduced at their rounding | error |
| AC-11 | Pricing | Analytic Greeks within 1e-4 relative of finite differences | error |
| AC-12 | Hedging | Ineffectiveness variance shares sum to 1 | error |
| AC-13 | Margin | IM and buffer positive; 10-day buffer at least the 1-day 99% call | error |
| AC-14 | Margin | Simulated IM within 10% of the closed-form approximation | error |
| AC-15 | Credit | PFE 97.5% at least PFE 95%, both non-negative; netted EE not above EE without netting | error |
| AC-16 | Credit | Breach status consistent with utilisation | error |
| AC-17 | Reporting | Number of limit and counterparty breaches reported | info |
| AC-18 | Exports | Dashboard CSVs exist and carry the synthetic label | error |

A failed **error** check stops the publication of the daily report (Stage 16 pipeline). A breach of a risk limit is not
a validation failure: it is a valid result that the escalation process in `docs/risk_framework.md` handles.

## Model acceptance (one-off, at handover)

| Model | Acceptance evidence |
|---|---|
| VaR and ES | Stage 6 backtests; official method chosen by rule in Stage 7 |
| Black-76 | Stage 10 validation tables; `tests/test_pricing.py` |
| Hedging | Exact ineffectiveness identity; `tests/test_hedging.py` |
| Margin | Historical buffer check in the Stage 12 report; `tests/test_margin.py` |
| Credit | Netting and CE tests; `tests/test_credit.py` |
| Attribution | Exact reconciliation in code and in `tests/test_attribution.py` |
| Documents | Every number traced to outputs: `tests/test_docs.py` |
