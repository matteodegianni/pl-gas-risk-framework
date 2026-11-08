# Handoff - project state

Polish gas supply risk framework (TGE physical, TTF hedge, EUR/PLN). Portfolio project for a Quantitative Analyst
(Analysis & Risk) application. Full brief: `docs/project_brief.md`. Interview preparation: `docs/interview_notes.md`.

## Working rules (from the user)

- Every figure in docs comes from code output; never invent or round numbers by hand (copy them exactly from the
  generated `outputs/stage*_report.md`).
- Synthetic data (positions, customer demand, counterparties) labelled **synthetic** everywhere.
- No raw or cleaned market data committed (Investing.com, TGE via Instrat); `data/` is gitignored except its README.
- Only hyphens in text, never long dashes (check with `grep -rnP "[\x{2013}\x{2014}]"`).
- After each stage: run all tests, regenerate the stage report, update README, commit (check nothing under
  `data/raw` or `data/clean` is staged). Stop only if a test fails and can't be fixed, a result contradicts an
  earlier stage, or a choice would change the brief.
- Before each stage, explain the key concept in 3-5 sentences.
- Push after each stage commit (`git push`); the user authorised this for Phases 2 and 3.

## Environment

- Repo: `~/pl-gas-risk-framework`, branch `master`, Python 3.12 venv in `.venv` (`pip install -r requirements.txt`).
- Investing.com exports in `data/raw/investing/ttf_c1.csv`, `ttf_c2.csv` (user-provided, gitignored).
- Pipeline order (each writes `outputs/stageN_*`):
  `gasrisk.data.build` (add `--offline` to reuse raw files), `gasrisk.diagnostics`, `gasrisk.forecasting`
  (5-10 min), `gasrisk.volatility`, `gasrisk.var` (1 min), `gasrisk.backtest`, `gasrisk.limits`, `gasrisk.stress`,
  `gasrisk.attribution`. Run as `.venv/bin/python -W ignore -m <module>`.
- Tests: `.venv/bin/python -m pytest -W ignore` (154 passing at Stage 18).

## Status

| Stage | Commit | Key outcome |
|---|---|---|
| 1 Data and quality | 94eeb2e | panel 2,321 days; c2 (ICE) held to roll is the TTF risk series |
| 2 Diagnostics | 6a85ceb | ECB chosen as EUR/PLN source by timing analysis; basis = location plus tenor |
| 3 Forecasting | e665018 | nothing beats the random walk after Holm; TGE 1-day gain from same-day TTF/FX |
| 4 Volatility | 0f52f78 | AR/DOW means selected; Student-t everywhere; EWMA unbeaten out of sample |
| 5 VaR and ES | 180cbda | six methods incl. t copula, ES 97.5% |
| 6 Backtesting | a7f1b7d | normal and EWMA fail at 99%; GARCH-t and FHS pass |
| 7 Limits | 5ed2e30 | official method GARCH-t Gaussian copula; TGE drives all VaR breaches |
| 8 Stress | 958f7d7 | worst crisis move driven by TGE spot, hedge ineffective against spikes |
| 9 Attribution | 212e67e | exact reconciliation; hedge offsets most of the purchase price effect |
| Phase 1 wrap-up | see git log | notebook, interview notes, this file, LICENSE |
| 10 Pricing | dd529de | Black-76 futures-style (ICE TTF options), GARCH-t vol as a labelled assumption; Hull and Haug examples match |
| 11 Hedging | aa5d303 | dollar-offset fails (29.1% pass), regression passes; basis carries 99.1% of ineffectiveness variance |
| 12 Margin | ee93c18 | IM 3.68m EUR, 10-day buffer 7.09m EUR; liquidity risk is a TTF fall, not the P&L stress scenarios |
| 13 Credit | 3fbc325 | CP04 breaches PFE limit; wrong-way risk documented and illustrated (EE 4.56x-5.92x in high-price paths) |
| 14 Framework | 899a61f | docs/risk_framework.md; `gasrisk.doccheck` traces every document number to outputs |
| 15 Handover | e7e762a | 9 model cards, input schema, 18 acceptance criteria, handover/validate_outputs.py |
| 16 Pipeline | 6f13b7e, b33bd12 | one-command pipeline, Dockerfile, Airflow DAG, exports/, Power BI PBIP (validated offline, never opened in Desktop) |
| 17 Memo | 45591df | docs/management_memo.md generated from tables; recommends TTF futures at 70% |
| 18 Commentary | 3df9be1 | number-checked LLM commentary; committed run used the template (anthropic SDK not installed) |

## Next steps

1. Public GitHub repository: https://github.com/matteodegianni/pl-gas-risk-framework (Phase 1 pushed; push
   later stages after each commit with `git push`).
2. Phase 2 (Stages 10-13): done.
3. Phase 3 (Stages 14-18): done.
4. Open items for the user: open `powerbi/GasRisk.pbip` in Power BI Desktop and follow the first-open checklist in
   `powerbi/README.md`; optionally `pip install anthropic` and run `python -m gasrisk.commentary` with credentials.

## Approved Phase 2 design

- **10. Pricing.** Black-76 for European options on TTF futures; underlying = TTF second-month settlement; EUR rate
  from the ECB API; ACT/365 to option expiry (verify the ICE TTF option expiry rule); analytic delta, gamma, vega, theta.
  *Change 1:* there is no free TTF implied volatility data, so volatility is the GARCH-t forecast or historical
  volatility, clearly labelled as an assumption (model values, not market prices). Validate the pricer with
  put-call parity and a published reference example (plus finite-difference Greeks).
- **11. Hedging.** Minimum-variance hedge ratio of TGE (PLN) on TTF x EUR/PLN (rolling, daily and monthly); strategies:
  no hedge, futures (optimal ratio and 0.70), TTF collar; compare risk reduction (P&L variance, VaR, ES) and cost.
  Effectiveness by dollar-offset (80-125%) and regression; state it is simplified, not IFRS 9 hedge accounting.
  *Change 2:* report a dollar-offset failure as it is, and quantify how much of the ineffectiveness comes from the
  TGE-TTF location plus tenor basis found in Stage 8.
- **12. Margin.** Variation margin from daily futures mark-to-market (EUR); initial margin approximated by a 2-day 99%
  VaR (filtered historical simulation); margin calls under base and stress (Stage 8) scenarios; liquidity buffer =
  99% quantile of 10-day cumulative margin outflow, in EUR and PLN.
- **13. Credit.** OTC TTF forwards with 6 **synthetic** counterparties; Monte Carlo of TTF (GARCH-t) on a monthly grid;
  CE, EE, PFE 95%/97.5% per counterparty with netting; scorecard from synthetic financial ratios (leverage, interest
  coverage, current ratio, EBITDA margin, size) mapped to an internal rating; rating plus PFE sets limits;
  utilisation and breach reporting. *Change 3:* document wrong-way risk as a limitation (counterparty quality falls
  when gas prices spike, which is when forward exposure peaks); the model treats exposure and default as independent.

## Known limitations to keep in mind

- No TGE forward curve: unhedged future purchases are valued at TGE day-ahead, so basis risk mixes location and tenor.
- Specifications for the GARCH mean are selected on the full sample (first-window check reported).
- Synthetic limits were set as round numbers relative to the synthetic book.
- Option values use a volatility assumption (no free implied volatility); IM is a VaR proxy, not the ICE method.
- Credit model treats exposure and default as independent (wrong-way risk documented, not modelled).
- Power Query has no relative paths: DataFolder defaults to the GitHub exports URL; `powerbi/set_data_folder.py`
  switches to the local clone.
