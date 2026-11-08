# Handoff - project state

Polish gas supply risk framework (TGE physical, TTF hedge, EUR/PLN). Portfolio project for a Quantitative Analyst
(Analysis & Risk) application. Full brief: `docs/project_brief.md`.

## Working rules (from the user)

- Every figure in docs comes from code output; never invent or round numbers by hand.
- Synthetic data (positions, customer demand, counterparties) labelled **synthetic** everywhere.
- No raw or cleaned market data committed (Investing.com, TGE via Instrat); `data/` is gitignored except its README.
- Only hyphens in text, never long dashes.
- After each stage: run all tests, regenerate the stage report, update README, commit (check nothing under
  `data/raw` or `data/clean` is staged). Stop only if a test fails and can't be fixed, a result contradicts an
  earlier stage, or a choice would change the brief.
- Before each stage, explain the key concept in 3-5 sentences. Commit only to the local repo; do not push to
  GitHub until the user confirms.

## Environment

- Repo: `~/pl-gas-risk-framework`, Python 3.12 venv in `.venv` (`pip install -r requirements.txt`).
- Investing.com exports in `data/raw/investing/ttf_c1.csv`, `ttf_c2.csv` (user-provided, gitignored).
- Pipeline order: `python -m gasrisk.data.build` (add `--offline` to reuse raw files), `gasrisk.diagnostics`,
  `gasrisk.forecasting` (5-10 min), `gasrisk.volatility`, `gasrisk.var` (1 min), `gasrisk.backtest`.
- Tests: `.venv/bin/python -m pytest -W ignore`.

## Status

| Stage | Commit | State |
|---|---|---|
| 1 Data and quality | 94eeb2e | done |
| 2 Diagnostics, ECB as FX source | 6a85ceb | done |
| 3 Forecasting (+ leakage, placebo, Holm) | e665018 | done |
| 4 Volatility (mean/dist selection) | 0f52f78 | done |
| 5 VaR and ES (6 methods, t copula, ES 97.5%) | 180cbda | done |
| 6 VaR backtesting | a7f1b7d | done |
| 7 Limits and breach analysis | - | next |
| 8 Stress tests and sensitivities | - | pending |
| 9 P&L attribution | - | pending |

## Next steps

1. Stage 7: official VaR method chosen by rule (passes conditional coverage at both levels, highest share of
   green traffic-light windows); synthetic limits (VaR 99%, open gas exposure value, FX net open position);
   utilisation, amber at 80%, breach episodes and factor attribution (Euler components).
2. Stage 8: historical crisis scenarios, hypothetical shocks, sensitivities per 1 PLN/MWh TGE, 1 EUR/MWh TTF, 0.01 EUR/PLN.
3. Stage 9: P&L attribution (plan margin, volume, price, FX, hedge) with synthetic demand; exact reconciliation test.
4. Phase 1 wrap-up: notebook, `docs/interview_notes.md`, update this file, 3 CV bullets, public release prep (no push).
5. Phase 2: show a short design summary for Stages 10-13 and wait for approval.
