# Input data specification

Machine-readable version: [input_schema.yaml](input_schema.yaml), enforced by `python -m gasrisk.validation --inputs-only`.

## Sources and refresh

| Series | Source | Frequency and timing | Unit | Licence and handling |
|---|---|---|---|---|
| TGEgasDA | Instrat API (data from TGE) | Daily, per delivery day; shifted to the trade date (D-1) | PLN/MWh | CC BY-NC 4.0; never committed |
| TTF second month (ICE Endex) and front month (CME listing) | Investing.com manual export placed in `data/raw/investing/` | Daily settlement, UK trading days | EUR/MWh | Redistribution not permitted; never committed |
| EUR/PLN | ECB reference rate (risk source), NBP and Yahoo (cross-checks) | Daily | PLN per EUR | Public (ECB, NBP); Yahoo not redistributed |
| EUR yield curve | ECB AAA spot curve (Svensson), 3M to 2Y | Daily | percent | Public |
| EUA auction prices | Instrat API (data from EEX) | Auction days | EUR/t | Context series only |

In production the Investing.com export would be replaced by a licensed ICE Endex settlement feed with contract
identifiers; the loader interface (`gasrisk.data.sources`) is the only part that changes.

## Derived inputs used by the models

| File (`data/clean/`) | Built by | Content |
|---|---|---|
| `panel.csv` | Stage 1 | Aligned daily panel: TTF levels and roll-adjusted returns, TGE in PLN and EUR, EUR/PLN by source, basis |
| `garch_conditional_vol.csv`, `garch_std_resid.csv` | Stage 4 | Conditional volatility and standardised residuals per series |
| `var_forecasts.csv` | Stage 5 | Walk-forward VaR and ES per method with realised P&L |

## Validation rules

1. Every file exists; the panel has at least 1500 rows.
2. Dates are valid, unique and increasing; no gap above 8 calendar days (a long holiday plus a weekend).
3. Prices within plausible ranges (TTF 1 to 1000 EUR/MWh, TGE 5 to 5000 PLN/MWh, EUR/PLN 3.5 to 6.0); daily log
   returns within plus or minus 0.8 (TTF) and 0.1 (EUR/PLN); null shares within the limits in the schema (returns may
   miss on the two excluded roll days).
4. TGE in EUR equals TGE in PLN divided by EUR/PLN.
5. The Stage 1 data quality log exists, so every outlier decision, fill and exclusion is traceable.
6. The last panel date is within 7 calendar days of the run date (warning).

Stage 1 applies the detailed cleaning rules (robust z-score 6.0 on a 250-day window, cross-series confirmation, stale
runs of 5 observations, FX carried over at most 3 days) before these checks run.
