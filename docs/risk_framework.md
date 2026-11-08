# Risk framework - Polish gas supply book

Scope: the risk framework of a Polish gas supplier that buys at the TGE day-ahead index (PLN), sells at fixed PLN
prices to B2B customers, hedges with ICE Endex TTF futures and options (EUR) and trades OTC TTF forwards. It documents
only what Stages 1-13 of this repository build and measure. **Positions, demand, counterparties, limits and the bank
credit line are synthetic**; market data are real (TGE via Instrat, TTF from Investing.com, EUR/PLN from ECB, NBP and
Yahoo). Every figure is copied from the generated reports in `outputs/` or from `config/config.yaml`, and
`python -m gasrisk.doccheck docs/risk_framework.md` verifies it.

## 1. Risk taxonomy

| Risk | Source in the business | Primary measure | Stage |
|---|---|---|---|
| Gas price | Fixed PLN sales against floating TGE purchases | 1-day 99% VaR and 97.5% ES (PLN) | 5-7 |
| Basis (location plus tenor) | TGE day-ahead (PLN) hedged with TTF second month (EUR) | Stress scenarios, hedge effectiveness, ineffectiveness split | 2, 8, 11 |
| EUR/PLN | Hedge P&L, premia and margin in EUR; TGE embeds EUR/PLN | VaR component, FX net open position (EUR) | 5, 7, 8 |
| Margin liquidity | Daily variation margin and initial margin on TTF futures | IM, 10-day 99% liquidity buffer, stress margin calls | 12 |
| Counterparty credit | Replacement cost of OTC TTF forwards | CE, EE, PFE 95% and 97.5% per netting set | 13 |
| Model | Every model above | Backtests, benchmarks, reconciliation, validation tests | 3-13 |

## 2. Methodologies

| Area | Method in production | Key parameters | Output |
|---|---|---|---|
| Data | Daily refresh, calendar alignment (UK trading days for TTF, TGE trade-date shift), TTF roll handling with the c1/c2 pair, outlier rule with cross-series confirmation, cross-source FX check | robust z 6.0, FX source gap 0.005 | `outputs/data_quality_log.csv` |
| Volatility | GARCH(1,1) with Student-t innovations, mean equation chosen by Ljung-Box and BIC (Stage 4) | window 750, monthly refit | conditional volatility per factor |
| VaR and ES | **Official: GARCH-t marginals with a Gaussian copula** (`garch_t_gauss`), chosen by rule in Stage 7; historical simulation, FHS, t copula, normal and EWMA kept as challengers | 20000 simulations, VaR 95% and 99%, ES 97.5% | daily VaR, ES and Euler components by factor |
| Backtesting | Kupiec, Christoffersen independence and conditional coverage, Basel traffic light on 250-day windows | 95% and 99% | pass or fail per method, zone |
| Limits | VaR, open gas exposure and FX net open limits with amber and breach states; breach attribution by Euler contribution | amber from 0.80 | utilisation, breach episodes and drivers |
| Stress | Worst 1/5/10/21-day joint moves in the 2021-2022 crisis, named episodes, hypothetical shocks, reverse stress, sensitivities | crisis window 2021-07-01 to 2022-12-31 | loss in PLN and as a multiple of VaR |
| P&L attribution | Sequential exact split: plan margin, volume, price, FX | exact reconciliation enforced in code (run fails otherwise) | daily and monthly attribution |
| Option pricing | Black-76, futures-style (undiscounted) for ICE Endex TTF options; ECB AAA curve; volatility is a labelled assumption (GARCH-t) | expiry 5 calendar days before delivery | values and Greeks as a share of F |
| Hedging | Minimum-variance ratio (daily and 21-day), strategy comparison, dollar-offset and regression effectiveness | dollar-offset band 0.80-1.25, regression R2 0.80 | hedge ratio, risk reduction, effectiveness |
| Margin | VM from mark-to-market; IM as 2-day 99% FHS VaR; 10-day buffer from FHS-GARCH paths | 20000 paths | IM, buffer in EUR and PLN, stress calls |
| Credit | Monte Carlo GARCH-t TTF on a monthly grid, netting sets, scorecard grade mapped to a PFE limit | 5000 paths, 24 months | CE, EE, PFE, utilisation |

## 3. Limit structure and thresholds

All limits are **synthetic** management numbers set relative to the synthetic book (Stage 7 note: they are round
numbers, not fitted to the backtest).

| Limit | Measure | Limit | Amber | Breach | Owner of the measure |
|---|---|---|---|---|---|
| Market risk | Official 1-day 99% VaR (PLN) | 25000000 PLN | 0.80 of limit | above 1.00 | Risk function |
| Open gas exposure | Unhedged volume x TGE price (PLN) | 60000000 PLN | 0.80 | above 1.00 | Risk function |
| FX net open position | EUR value of TGE purchases not covered by the EUR forward | 12000000 EUR | 0.80 | above 1.00 | Treasury |
| Margin liquidity | 10-day 99% buffer and stress margin calls against the committed bank line | 25000000 EUR line | stress call above the buffer | stress call above the line | Treasury |
| Counterparty credit | Peak 95% PFE per netting set against the grade limit (grade 1: 20000000 EUR down to grade 6: 0) | per grade | 0.80 | above 1.00 | Credit risk |
| Hedge programme | TTF futures volume as a share of open volume | 0.70 in the reference book | effectiveness test failures reported monthly | - | Risk committee |

## 4. Escalation

| State | Trigger | Action | Timing |
|---|---|---|---|
| Green | Utilisation below 0.80 | Daily report only | Daily |
| Amber | Utilisation from 0.80 to 1.00, a yellow traffic-light zone, or a stress margin call above the 10-day buffer | Head of Risk informed; desk proposes how to reduce or justifies the position; driver attribution attached | Same day |
| Breach | Utilisation above 1.00, a red traffic-light zone, or a stress margin call above the bank line | CFO informed; position reduced or a temporary limit approved in writing; root cause from the breach attribution (Stage 7) | Same day, plan within 2 business days |
| Repeated breach | Breach episode longer than 5 business days or recurring within 20 days | Risk committee and Management Board; limit or hedge programme reviewed | Next committee |
| Model alert | Backtest fails conditional coverage, attribution residual above tolerance, data quality error | Model owner investigates; challenger method used for reporting until resolved | Within 5 business days |

## 5. Model validation

| Model | Validation in this repository | Current result (from the reports) |
|---|---|---|
| VaR (official `garch_t_gauss`) | Kupiec, Christoffersen, traffic light; challengers | Passes conditional coverage at both levels; 0.9431 of 250-day windows green |
| VaR challengers | Same tests | Normal and EWMA fail at 99% (30 exceptions each against 15.68 expected) |
| Volatility | Out-of-sample QLIKE and MSE against EWMA and rolling variance, Diebold-Mariano with Holm | EWMA not beaten after Holm adjustment |
| Forecasting | Walk-forward against the random walk, leakage tests, placebo | No model beats the random walk after Holm adjustment |
| Black-76 | Published examples, put-call parity, finite-difference Greeks, implied-volatility round trip | Hull and Haug examples match; maximum finite-difference relative error 3.529e-08 |
| Hedge effectiveness | Dollar-offset and regression on 21-day periods | Dollar-offset fails (29.1% of periods pass); regression passes (R2 0.8439) |
| Margin | Historical check of the buffer | 1.86% of historical 10-day moves exceed the buffer (target 1%), explained by crisis volatility |
| Attribution | Exact reconciliation enforced in code and tests | Maximum daily residual 9.313e-10 PLN |
| Credit | Unit tests for netting, CE and expiry; price fan reviewed | Wrong-way risk not modelled (limitation) |

Validation frequency: backtests and the attribution check run daily in the pipeline (Stage 16); model re-estimation is
monthly (refit every 21 trading days); a full revalidation of every model card (Stage 15) is annual or after a
material change in data, positions or methodology.

## 6. Current risk position (synthetic book)

- Official 99% VaR 18,323,060 PLN, utilisation 0.7329 (green); TGE carries a 0.9108 share of it.
- Both value-based exposure limits are in breach (utilisation 1.0152 for open gas exposure and 1.1556 for FX net
  open) because TGE prices have risen, which scales exposure in PLN.
- Worst stress loss: 689,596,369 PLN, 37.64x today's 99% VaR (worst 21-day crisis move, driven by TGE spot).
- Margin: IM 3,676,450 EUR; 10-day buffer 7,085,393 EUR (28.3% of the credit line); largest stress call
  17,864,382 EUR (71.5% of the line).
- Credit: CP04 in breach (utilisation 14.35), CP05 amber (0.96).

## 7. Assumptions and limitations

- **Basis.** No TGE forward curve and no TTF day-ahead series in the free data: unhedged future purchases are valued at
  TGE day-ahead, so the measured basis mixes location and tenor, and VaR and stress overstate what a same-tenor hedge
  would leave.
- **Volatility for options.** No free TTF implied volatility: option values use the GARCH-t forecast and are model
  values; there is no smile.
- **GARCH specifications** are selected on the full sample (first-window check reported) and near-unit persistence may
  partly reflect structural breaks.
- **Positions are static** in MWh over the VaR horizon; the hedge ratio of the reference book is fixed at 0.70.
- **Hedge effectiveness** is a simplified test, not IFRS 9 hedge accounting.
- **Initial margin** is a VaR proxy, not the ICE methodology.
- **Credit** uses a single-factor flat-curve price model, no collateral, and treats exposure and default as
  independent (**wrong-way risk not modelled**).
- **Synthetic inputs.** Limits, the bank line, positions, demand and counterparties are synthetic; the framework's
  logic, not its calibration, is the deliverable.
