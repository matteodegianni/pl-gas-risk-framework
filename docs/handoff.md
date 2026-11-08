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
- Do **not** push to GitHub until the user confirms.

## Environment

- Repo: `~/pl-gas-risk-framework`, branch `master`, Python 3.12 venv in `.venv` (`pip install -r requirements.txt`).
- Investing.com exports in `data/raw/investing/ttf_c1.csv`, `ttf_c2.csv` (user-provided, gitignored).
- Pipeline order (each writes `outputs/stageN_*`):
  `gasrisk.data.build` (add `--offline` to reuse raw files), `gasrisk.diagnostics`, `gasrisk.forecasting`
  (5-10 min), `gasrisk.volatility`, `gasrisk.var` (1 min), `gasrisk.backtest`, `gasrisk.limits`, `gasrisk.stress`,
  `gasrisk.attribution`. Run as `.venv/bin/python -W ignore -m <module>`.
- Tests: `.venv/bin/python -m pytest -W ignore` (93 passing at Stage 9).

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

## Next steps

1. User to confirm before pushing to a public GitHub repository (README, LICENSE and data notice are ready;
   `git remote` is not set).
2. Phase 2: the user asked for a short design summary of Stages 10-13 (pricing method and inputs, hedge strategies,
   margin method, PFE setup, counterparty scoring inputs) and to wait for approval before building. The summary
   was given in chat at the end of Phase 1; build only after the user's OK.
3. Phase 3 (Stages 14-18) after Phase 2.

## Known limitations to keep in mind

- No TGE forward curve: unhedged future purchases are valued at TGE day-ahead, so basis risk mixes location and tenor.
- Specifications for the GARCH mean are selected on the full sample (first-window check reported).
- Synthetic limits were set as round numbers relative to the synthetic book.
