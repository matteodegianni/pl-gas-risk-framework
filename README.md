# Polish Gas Supply Risk Framework

Market, margin and credit risk framework for a Polish gas supplier that **buys physical gas at the TGE day-ahead index (PLN)**, **sells to B2B customers at fixed PLN prices** and **hedges with ICE Endex Dutch TTF futures and options (EUR)**. The book is exposed to gas price risk, TGE-TTF basis risk, EUR/PLN FX risk, margin liquidity risk and counterparty credit risk.

Full scope and stage plan: [docs/project_brief.md](docs/project_brief.md).

> Status: **Phases 1 and 2 (Stages 1-13) complete.** Phase 3 (framework and delivery) in progress.

## Repository layout

```text
config/config.yaml        all parameters: paths, sources, thresholds, calendars
src/gasrisk/
  config.py               config loading, repo-relative paths
  data/
    sources.py            loaders and parsers (Instrat, NBP, ECB, Yahoo, Investing.com)
    calendars.py          UK and Polish business days, TTF expiry and roll schedule
    rolls.py              roll-adjusted TTF returns from the c1/c2 pair
    quality.py            issue log and checks (gaps, OHLC, stale runs, outliers)
    build.py              Stage 1 pipeline: load, check, align, write outputs
    report.py             generated Stage 1 report
  diagnostics.py          Stage 2: FX source timing, stationarity, serial dependence, distribution
  diagnostics_report.py   generated Stage 2 report
  forecasting.py          Stage 3: walk-forward forecasts, Diebold-Mariano tests, ablation
  forecasting_report.py   generated Stage 3 report and figure
  volatility.py           Stage 4: GARCH(1,1) normal vs t, EWMA and rolling benchmarks, forecast evaluation
  volatility_report.py    generated Stage 4 report and figure
  var.py                  Stage 5: VaR and ES of the synthetic book (six methods)
  var_report.py           generated Stage 5 report and figure
  backtest.py             Stage 6: Kupiec, Christoffersen, Basel traffic light, ES check
  backtest_report.py      generated Stage 6 report and figure
  limits.py               Stage 7: official VaR method, synthetic limits, utilisation, breach attribution
  limits_report.py        generated Stage 7 report and figure
  stress.py               Stage 8: sensitivities, historical and hypothetical scenarios, reverse stress
  stress_report.py        generated Stage 8 report
  attribution.py          Stage 9: daily P&L attribution of a synthetic supply contract
  attribution_report.py   generated Stage 9 report and figure
  pricing.py              Stage 10: Black-76 options on TTF futures, Greeks, ECB rate, validation
  pricing_report.py       generated Stage 10 report
  hedging.py              Stage 11: minimum-variance hedge ratio, strategies, dollar-offset and regression tests
  hedging_report.py       generated Stage 11 report
  ttf_sim.py              shared TTF GARCH refit and path simulation (Stages 10, 12, 13)
  margin.py               Stage 12: variation and initial margin, base and stress calls, liquidity buffer
  margin_report.py        generated Stage 12 report
  credit.py               Stage 13: CE, EE, PFE by Monte Carlo, synthetic scorecard, credit limits
  credit_report.py        generated Stage 13 report
  doccheck.py             checks that every number in a document appears in the generated outputs or config
  validation.py           Stage 15: input validation (schema) and post-deployment output validation
tests/                    pytest suite (offline, no network)
outputs/                  committed results: data quality log and Stage 1 tables
data/                     raw and clean data (gitignored, see data/README.md)
docs/                     brief, interview notes, handoff; later: risk framework, model cards, memo
notebooks/                01_core_market_risk.ipynb: Phase 1 story from the committed result tables
handover/                 Stage 15: input data spec and schema, acceptance criteria, output validation script
airflow/                  later stages
```

## How to run

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Place the two Investing.com exports (English site, daily, see [data/README.md](data/README.md)) in `data/raw/investing/`, then:

```bash
.venv/bin/python -m gasrisk.data.build
```

This downloads TGE gas and EUA auction prices from the Instrat API, EUR/PLN from NBP, ECB and Yahoo, runs every quality check and writes `outputs/`. Use `--offline` to rebuild from the stored raw files. Then run the Stage 2 diagnostics:

```bash
.venv/bin/python -m gasrisk.diagnostics
.venv/bin/python -m gasrisk.forecasting   # Stage 3, about 5-10 minutes on 4 cores
.venv/bin/python -m gasrisk.volatility    # Stage 4
.venv/bin/python -m gasrisk.var           # Stage 5, about 1 minute
.venv/bin/python -m gasrisk.backtest      # Stage 6
.venv/bin/python -m gasrisk.limits        # Stage 7
.venv/bin/python -m gasrisk.stress        # Stage 8
.venv/bin/python -m gasrisk.attribution   # Stage 9
.venv/bin/python -m gasrisk.pricing       # Stage 10 (fetches the ECB yield curve; --offline reuses it)
.venv/bin/python -m gasrisk.hedging       # Stage 11
.venv/bin/python -m gasrisk.margin        # Stage 12
.venv/bin/python -m gasrisk.credit        # Stage 13
```

```bash
.venv/bin/python -m pytest
```

## Stage 1 - data and data quality

All results are generated by the build: see [outputs/stage1_report.md](outputs/stage1_report.md) and the full issue log [outputs/data_quality_log.csv](outputs/data_quality_log.csv) (one row per event or contiguous run, with the action taken and the reason).

Methodology in brief:

- **Calendars.** TTF on ICE Endex trading days, NBP on Polish business days, TGE on its 7-day trading calendar. The aligned panel uses TTF trading days.
- **TGE timing.** TGEgasDA is published per delivery day but is the volume weighted price of trades on the previous day, so it is shifted to the trade date before it is compared with TTF settlements or FX fixings.
- **TTF rolls.** Expiry is two UK business days before the delivery month; the roll day is the first trading day after expiry. On roll days the return is computed within one contract, ln(c1_t / c2_t-1), which removes the calendar spread. Because c1 (CME listing) and c2 (ICE Endex) come from different venues, adjusted roll returns beyond 4 robust standard deviations are excluded as probable misalignments.
- **TTF risk series.** The ICE second-month contract held to roll is the TTF risk factor: c1 is a thin CME listing with many settlement-only days, c2 is the liquid ICE contract. c1 is used for roll-day returns and as a cross-check.
- **Outliers.** Robust z-scores on log returns against a trailing median and MAD (no look-ahead). A move is presumed real; a price is removed only if a second series contradicts the move and it reverses the next day.
- **EUR/PLN.** All three sources (NBP, ECB, Yahoo) are cleaned and carried into the panel, each over its own holidays for at most 3 days. Gaps between sources are logged. The risk source is chosen in Stage 2.

## Stage 2 - diagnostics

Generated results: [outputs/stage2_report.md](outputs/stage2_report.md).

- **EUR/PLN source timing.** Daily returns of series sampled at different times of day are asynchronous, which biases same-day correlations toward zero. Lead-lag correlations with TTF returns and a moving-block bootstrap of the share of correlation captured at lag 0 select the source sampled closest to the TTF settlement (ECB); NBP and Yahoo remain cross-checks. The analysis also detects that Yahoo's daily EUR/PLN values behave like quotes taken around the previous evening.
- **Stationarity.** ADF and KPSS read jointly on log levels (TTF as a roll-adjusted index) and on returns.
- **Location plus tenor basis.** TGE day-ahead spot (in EUR) minus TTF second month; diagnostics include which leg drives its tail days.
- **Serial dependence.** Ljung-Box on returns and squared returns, a heteroskedasticity-robust portmanteau test for the mean (standard Ljung-Box over-rejects under volatility clustering) and ARCH-LM.
- **Distribution.** Skewness, excess kurtosis, Jarque-Bera, tail frequencies and quantiles against the normal, full sample and pre-crisis, crisis and post-crisis sub-periods.

## Stage 3 - forecasting

Generated results: [outputs/stage3_report.md](outputs/stage3_report.md).

- **Targets and horizons.** TTF (roll-adjusted), EUR/PLN (ECB), TGE and the location plus tenor basis at 1, 5 and 21 business days; log-price loss for prices, EUR/MWh for the basis.
- **Walk-forward.** Rolling 750-day window, parameters re-estimated every 21 trading days, state updated daily; direct h-step models train only on targets observed by the fit date (tested).
- **Models.** Random walk benchmark, ARIMA and ETS chosen by AIC, gradient boosting, and a linear direct model for TGE and the basis with a step-by-step ablation of its feature groups.
- **Evaluation.** RMSE, MAE, directional accuracy with a binomial test, Diebold-Mariano against the random walk (HAC variance with h-1 lags, Harvey-Leybourne-Newbold correction), Holm adjustment across all 42 model-horizon-series comparisons. A random-walk win is reported as a valid finding.
- **Robustness.** Leakage tests (no feature or forecast at t changes when data after t change), a placebo with TTF and FX lagged one more day, and a Newey-West sensitivity for the DM variance.

Key findings (all figures from the generated report):

- **TTF and EUR/PLN:** the random walk is not beaten at any horizon; gradient boosting is significantly worse at every horizon.
- **TGE, 1 day:** the linear model has an RMSE ratio of 0.916 against the random walk (raw DM p 0.0022, directional accuracy 0.670). The gain comes from same-day TTF and FX moves: with them lagged one more day, 5.9% of the gain survives. It **does not survive Holm adjustment** (adjusted p 0.0928 across all 42 comparisons; 0.0530 within the 24 TGE and basis comparisons).
- **Basis, 1 day:** the lowest RMSE ratio is 0.911 (ETS) and directional accuracy ranges from 0.627 to 0.654 across models, but the RMSE gain is not significant (smallest raw DM p 0.0729); the linear model's gain comes from the basis's own reversal and survives the placebo.
- **Overall:** after multiple-testing correction, no model beats the random walk for any series or horizon.

## Stage 4 - volatility

Generated results: [outputs/stage4_report.md](outputs/stage4_report.md).

- **Mean equation.** Constant, AR(p) and day-of-week candidates fitted with GARCH(1,1)-t; the constant mean is kept only if its standardised residuals pass Ljung-Box at lags 5, 10 and 20, otherwise the lowest-BIC candidate that passes is used (Stage 5 filtered historical simulation resamples these residuals). The selection is repeated on the first training window as a look-ahead check.
- **Innovations.** Normal and Student-t for all series, plus skewed-t for the basis; chosen by BIC and two-sided 1% tail calibration.
- **Out-of-sample.** Rolling 750-day window, monthly re-estimation, daily filter updates; cumulative variance forecasts at 1, 5 and 21 days (accounting for the AR mean) against EWMA (RiskMetrics, lambda 0.94) and a 63-day rolling variance; MSE and QLIKE losses, DM tests with Holm adjustment, Mincer-Zarnowitz regressions.

Key findings (all figures from the generated report):

- **AR(1) is not enough.** It leaves significant residual autocorrelation for TTF, TGE and the basis. Selected means: TTF AR(5), TGE AR(10) with day-of-week, basis AR(5) with day-of-week, EUR/PLN constant. Ljung-Box p at lag 10 moves from 0.0039 to 0.5927 (TTF), 0.0000 to 0.8344 (TGE) and 0.0000 to 0.5709 (basis). The first-window check gives the same choice except for EUR/PLN (AR(1) on the first window).
- **Student-t is selected for all four series** on both BIC and tail calibration. For the basis, skewed-t improves the left 1% tail (0.0134 against 0.0147) but worsens the right tail (0.0134 against 0.0099) and has a higher BIC (7898.8 against 7896.0), so Student-t is kept.
- **Volatility is near-integrated** for TTF (persistence 0.9992) and the basis, which is also why EWMA, an integrated model, is hard to beat.
- **Out of sample, EWMA is not beaten after Holm adjustment** for any series or horizon; GARCH lowers 1-day QLIKE for TGE and the basis at the raw level only (raw DM p 0.0051 to 0.0307, Holm p 0.1990 or higher).

## Stage 5 - VaR and Expected Shortfall

Generated results: [outputs/stage5_report.md](outputs/stage5_report.md). All positions are **synthetic**.

- **Book (synthetic).** Short floating TGE purchases behind fixed-price PLN sales (500,000 MWh open), long TTF futures (0.70 x open volume, EUR), long EUR/PLN forward (EUR notional 0.70 x open volume x TGE in EUR). Risk factors: TGE in EUR, TTF (held to roll), EUR/PLN (ECB); every scenario fully revalued.
- **Methods.** Historical simulation (500 days), multivariate normal (250 days), GARCH(1,1)-t marginals from Stage 4 with a Gaussian copula and with a Student-t copula, filtered historical simulation, and EWMA (RiskMetrics, lambda 0.94). VaR at 95% and 99%; ES at 95%, 97.5% (Basel FRTB) and 99%. Walk-forward from the same test start as Stages 3-4.
- **Tests.** Sign conventions (TGE rise loses, TTF rise gains, EUR/PLN rise gains on the forward and raises the PLN cost of TGE), legs sum to the total, copula tail dependence, no look-ahead.

## Stage 6 - VaR backtesting

Generated results: [outputs/stage6_report.md](outputs/stage6_report.md).

- Kupiec proportion-of-failures, Christoffersen independence and conditional coverage at 95% and 99% for all six Stage 5 methods; Basel traffic light on 99% VaR over 250-day windows; descriptive ES check on exception days.
- Key findings (from the generated report): over 1,568 days, the normal and EWMA methods fail at 99% (30 exceptions against 15.68 expected each; conditional coverage p 0.0004 and 0.0048), consistent with the Stage 4 finding that normal tails are too thin. GARCH-t with both copulas and FHS pass at 99% (12, 11 and 10 exceptions), and FHS has every rolling 250-day window in the green zone. Historical simulation and GARCH-t with the Gaussian copula pass conditional coverage at both levels.

## Stage 7 - limits and breach analysis

Generated results: [outputs/stage7_report.md](outputs/stage7_report.md). Limits are **synthetic** management numbers.

- **Official VaR method** chosen by rule from Stage 6: passes conditional coverage at both levels, highest share of green traffic-light windows (GARCH-t with Gaussian copula).
- **Limits:** 99% VaR (PLN), open gas exposure value (unhedged volume x TGE in PLN), FX net open position (EUR); amber from 80%, breach above 100%; daily utilisation and breach episodes.
- **Breach attribution:** Euler decomposition of the official VaR by risk factor (main contributor and largest increase since 20 days before the episode); exposure breaches split exactly into TGE price and EUR/PLN contributions.
- Key findings (from the generated report): TGE is the main contributor in all 33 VaR breach episodes and has a 0.9108 share of today's 99% VaR; today the synthetic book's VaR limit utilisation is 0.7329 and breaches both value-based exposure limits (utilisation 1.0152 and 1.1556) because TGE prices have risen.

## Stage 8 - stress testing and sensitivity

Generated results: [outputs/stage8_report.md](outputs/stage8_report.md). Today's **synthetic** positions, full revaluation.

- Sensitivities per +1 PLN/MWh TGE, +1 EUR/MWh TTF and +0.01 EUR/PLN, plus a grid of larger moves; worst 1/5/10/21-day joint moves in the 2021-2022 crisis and named episodes; hypothetical shocks (TTF +50% alone, gas complex +/-50%, EUR/PLN +5%, combined, basis widening); reverse stress against the VaR limit.
- Key findings (from the generated report): P&L is -500,000 PLN per +1 PLN/MWh TGE and +1,537,375 PLN per +1 EUR/MWh TTF. The worst scenario is the worst 21-day crisis move (November 2022: TGE +355.0%, TTF +18.6%), a loss of 689,596,369 PLN or 37.64x today's 99% VaR; the TGE leg accounts for 103.3% of it because the month-ahead TTF hedge does not follow TGE day-ahead spikes. A relative TGE rise of 0.1231 alone would lose the 25,000,000 PLN VaR limit.

## Stage 9 - P&L attribution

Generated results: [outputs/stage9_report.md](outputs/stage9_report.md). Contract, demand and hedges are **synthetic**.

- One-year fixed-price PLN supply contract with a winter-peak contracted profile and synthetic AR(1) demand deviations; purchases at TGE, TTF futures and EUR forward hedges sized on the remaining volume.
- Daily P&L split sequentially and exactly into plan margin, volume effect, price effect (TGE purchases and TTF hedge at plan FX) and FX effect (purchase cost, hedge translation, forward); reconciliation enforced in code and by tests.
- Key findings (from the generated report): total P&L 1,565,811 PLN against a plan margin of 10,000,000 PLN; the price effect is -7,936,717 PLN, with purchases at -38,096,776 PLN partly offset by the TTF hedge at +30,160,059 PLN; maximum daily reconciliation residual 9.313e-10 PLN.

## Stage 10 - Black-76 option pricing

Generated results: [outputs/stage10_report.md](outputs/stage10_report.md).

- European options on the TTF second-month future with analytic delta, gamma, vega, theta and rho; expiry by the ICE Endex rule (5 calendar days before the delivery month, prior UK business day), ACT/365; EUR rate from the ECB AAA spot curve. ICE Endex TTF options are futures-style margined, so the exchange value is undiscounted; the premium-paid value is shown for OTC options.
- **Volatility is an assumption**, not market data: there is no free TTF implied volatility, so the input is the Stage 4 GARCH-t forecast to expiry, with 63-day historical and EWMA volatility as alternatives. Values are model values, reported as a share of the futures price.
- Validation: two published textbook examples (Hull; Haug), put-call parity on a strike grid, finite-difference Greeks and an implied-volatility round trip.
- Key findings (from the generated report): the Jan 2027 option expires on 2026-12-24 (T 0.131507); GARCH-t volatility 0.5947; the ATM call is worth 8.5864% of F with GARCH-t, 7.6939% with historical and 9.1661% with EWMA volatility, so the volatility choice moves the value by 1.4722% of F. All reference examples match; maximum finite-difference relative error 3.529e-08.

## Stage 11 - hedging and hedge effectiveness

Generated results: [outputs/stage11_report.md](outputs/stage11_report.md). Book is **synthetic**; effectiveness tests are a simplified illustration, **not IFRS 9 hedge accounting**.

- Minimum-variance hedge ratio of TGE (PLN) on TTF futures converted to PLN, daily (rolling 250 days) and on non-overlapping 21-day changes (full sample, sub-periods, rolling 36 blocks), with and without an EUR forward.
- Strategies on today's book over 21-day historical windows with full revaluation: no hedge, futures at h*, futures at 0.70, TTF collar (long 1.10F call, short 0.90F put), each with and without the EUR forward; P&L variance, 99% VaR, 97.5% ES and cost.
- Dollar-offset (80-125%) and regression (R2 >= 0.80, slope 0.80-1.25) tests per 21-day period, and an exact split of the ineffectiveness into the TGE-TTF basis term and the FX term.
- Key findings (from the generated report): h* is 0.7840 on daily changes and 1.1143 on 21-day changes (latest rolling 36 blocks 0.8999). Futures at h* cut 21-day P&L variance by 52.8% and 99% VaR by 44.0%, but 99% VaR by only 12.5% in crisis windows. The **dollar-offset test fails** (29.1% of 110 periods pass) while the regression test passes (R2 0.8439); the TGE-TTF location plus tenor basis carries 99.1% of the variance of the ineffectiveness and 93.4% of it in failing periods. The collar costs a net 0.7711% of F at the volatility assumption.

## Stage 12 - margin forecasting

Generated results: [outputs/stage12_report.md](outputs/stage12_report.md). Position and bank credit line are **synthetic**.

- Variation margin from daily mark-to-market of the long TTF futures hedge (EUR); initial margin as a 2-day 99% VaR by filtered historical simulation on the refitted Stage 4 GARCH-t model (driftless futures price).
- Base forecast: 20,000 FHS-GARCH paths over 10 days with IM recomputed along each path; stress: Stage 8 scenarios plus the worst crisis TTF falls, with IM after the scenario, and a volatility-only stress.
- Liquidity buffer: 99% quantile of the 10-day cumulative margin outflow in EUR and PLN, against a synthetic credit line, with a historical check.
- Key findings (from the generated report): IM today 3,676,450 EUR (12.30% of notional); 10-day buffer 7,085,393 EUR (31,122,589 PLN, 28.3% of the credit line). The largest stress call is the worst 10-day TTF fall of the crisis, 17,864,382 EUR (71.5% of the line). The P&L stress scenarios (TGE spikes) bring cash in on the hedge: liquidity risk sits in the opposite scenario, a TTF collapse.

## Stage 13 - counterparty credit risk

Generated results: [outputs/stage13_report.md](outputs/stage13_report.md). Counterparties, ratios, trades and limits are **synthetic**.

- OTC TTF forward strips with six synthetic counterparties; Monte Carlo of TTF with the refitted GARCH(1,1)-t (5,000 paths, daily steps, monthly grid, flat curve); CE, EE, PFE 95% and 97.5% per counterparty with and without netting, discounted on the ECB curve.
- Scorecard from five synthetic financial ratios (leverage, interest coverage, current ratio, EBITDA margin, size) mapped to an internal grade; the grade sets the limit on peak 95% PFE; utilisation and breach status as in Stage 7.
- **Wrong-way risk is a documented limitation**: the model treats exposure and default as independent; the report shows how much higher exposure is in high-price paths.
- Key findings (from the generated report): CP04 (grade 5) breaches its PFE limit (utilisation 14.35) and CP05 is amber (0.96); netting lowers CP05's peak EE by 27.5%; for counterparties selling gas to the supplier, peak-month EE in the 10% highest-price paths is 4.56x to 5.92x the average, which is where wrong-way risk would bite.

## Stage 14 - risk framework

[docs/risk_framework.md](docs/risk_framework.md): risk taxonomy, methodologies, limit structure and thresholds, escalation, model validation, current risk position, assumptions and limitations, based only on what Stages 1-13 build. Every number in it, and in the README, memo, model cards and interview notes, is checked against the generated outputs by `python -m gasrisk.doccheck` and `tests/test_docs.py`.

## Stage 15 - model handover package

- **Model cards** ([docs/model_cards/](docs/model_cards/README.md)): one per model (forecasting, volatility, VaR and ES, stress, attribution, Black-76, hedging, margin, credit) with purpose, inputs, method, validation evidence, limitations and ownership.
- **Input data specification** ([handover/input_data_spec.md](handover/input_data_spec.md)) and its machine-readable schema ([handover/input_schema.yaml](handover/input_schema.yaml)): sources, refresh, units, licences, validation rules.
- **Acceptance criteria** ([handover/acceptance_criteria.md](handover/acceptance_criteria.md)): input criteria and 18 output criteria (completeness, freshness, VaR sanity, backtest, reconciliation, pricing identities, margin and credit consistency, export labelling).
- **Output validation script**: `python handover/validate_outputs.py` runs every criterion, writes `outputs/validation_report.csv` and exits with code 1 on any error. On the current run all checks pass, with 2 exposure limits and 1 counterparty in breach reported as information.

## Licence

Code: MIT (see `LICENSE`). Market data are not included and remain subject to the providers' terms below.

## Data sources and credits

| Data | Source | Licence |
|---|---|---|
| TGEgasDA gas day-ahead index | [energy.instrat.pl](https://energy.instrat.pl/en/prices/gas-dam/) (Instrat), data originating from TGE (Towarowa Gielda Energii S.A., [tge.pl](https://tge.pl)) | CC BY-NC 4.0 (Instrat); TGE terms of use also apply |
| EU ETS primary auction prices | [energy.instrat.pl](https://energy.instrat.pl/en/prices/eu-ets/) (Instrat), data from EEX | CC BY 4.0 unless stated otherwise (Instrat) |
| Dutch TTF futures, first and second month | Investing.com, manual export | Redistribution not permitted |
| EUR/PLN mid fixing | [NBP](https://api.nbp.pl) | Public data |
| EUR/PLN reference rate | [ECB](https://data.ecb.europa.eu) | Public data |
| EUR/PLN quote | Yahoo Finance via yfinance | Not redistributed |

Gas price data: Instrat, energy.instrat.pl, CC BY-NC 4.0, original source TGE. This is a non-commercial portfolio project. No raw or cleaned market data are committed; the code fetches API data at runtime and reads the Investing.com exports from a local, gitignored folder. Committed outputs contain counts, dates, returns-based statistics and issue metadata only.

## Limitations

- TTF prices come from manual exports of continuation series; contract identities on each day are inferred from the expiry rule, not supplied by the source.
- The front-month export (CME listing) is thin, with long settlement-only periods.
- **The TGE-TTF basis is a location plus tenor basis**, not a pure location basis: TGE is a Polish day-ahead spot index, the hedge leg is the Dutch TTF second-month future. Its fat tails come mainly from TGE spot spikes (the TGE leg is the larger leg on most basis tail days, see the Stage 2 report), so basis risk measured here overstates what a same-tenor hedge would leave. A TTF day-ahead or TGE forward series would be needed to separate the two components; neither is available from the free sources used.
- EUA prices are auction clearing prices held every few days, not continuous futures settlements.
- GARCH mean equations are selected on the full sample (a first-window check is reported), and near-unit persistence may partly reflect unmodelled structural breaks such as the 2021-2022 crisis.
- TGE 1-day forecastability comes from same-day TTF and FX moves (Stage 3 ablation and placebo) and does not survive a Holm correction for multiple testing. TGEgasDA averages trades over the day while TTF settles at the end of it, so the effect partly reflects index construction: usable as a risk forecast input, not evidence of a tradable edge.
