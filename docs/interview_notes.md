# Interview notes - Stages 1-18

All figures are copied from the generated reports in `outputs/`. Book, limits, demand and contract are synthetic.

## Stage 1 - Data and data quality

**Decisions and why**
- TTF history from Investing.com exports (Yahoo `TTF=F` returned one daily row); TGE from the Instrat API (CC BY-NC 4.0, data from TGE); EUR/PLN from NBP, ECB and Yahoo.
- Risk series: ICE second-month TTF (c2) held to roll, because the CME front month (c1) is thin with long settlement-only periods.
- Roll days from the expiry rule (two UK business days before delivery); roll-day return ln(c1_t / c2_t-1) removes the calendar spread.
- TGE shifted from delivery date to trade date before aligning with TTF and FX.
- Outliers kept unless contradicted by a second series and reversed the next day; every action logged.

**Results**
- Panel of 2,321 trading days, 2017-10-23 to 2026-11-06.
- Roll validation: slope of raw c1 return on the prior c2/c1 spread 0.6917 on roll days vs 0.0204 on other days; 107 roll days adjusted, 2 excluded as misaligned.

**Likely questions**
1. *Why not use the front month?* Its export has long settlement-only periods (logged as `settlement_only_row` in `outputs/data_quality_log.csv`) and it is a thin CME listing; c2 is the liquid ICE contract and can be held to roll with a clean return.
2. *How do you handle a futures roll?* On the first trading day after expiry the front contract is yesterday's second month, so the same-contract return is ln(c1_t / c2_t-1); raw returns that day include the calendar spread.
3. *Why shift TGE by a day?* TGEgasDA for delivery day D is the volume weighted price of trades on D-1; aligning on delivery date would misalign it with TTF settlements and FX fixings.

## Stage 2 - Diagnostics

**Decisions and why**
- EUR/PLN source chosen by timing: the source whose correlation with TTF sits at lag 0 (capture ratio) is sampled closest to the TTF settlement.
- Heteroskedasticity-robust portmanteau test for the mean, because standard Ljung-Box over-rejects under volatility clustering.
- Basis labelled a location plus tenor basis (TGE day-ahead spot vs TTF second month).

**Results**
- Capture ratio: ECB 0.6713, NBP 0.5166, Yahoo 0.4816; ECB best in 86.4% of bootstrap draws (not significant at 95% vs NBP; consistent across robustness checks). Yahoo's daily values behave like quotes taken the previous evening.
- Prices: unit root; returns: stationary; basis level: stationary. ARCH effects and fat tails in every series.
- Basis tails come mainly from TGE: the TGE leg is larger on 34 of 52 tail days.

**Likely questions**
1. *Why ECB rather than NBP, the accounting rate?* For risk the FX factor must be sampled close to the TTF settlement, otherwise correlations are biased toward zero; NBP remains available for accounting conversion.
2. *Why not trust standard Ljung-Box here?* It assumes i.i.d. returns; with ARCH effects it rejects too often. The robust test showed TTF and EUR/PLN autocorrelation is weak.
3. *What does a stationary basis mean for hedging?* Basis shocks mean-revert (AR(1) half-life 6.5 days in Stage 3), so TTF is a usable hedge on longer horizons, but day-to-day TGE spikes remain.

## Stage 3 - Forecasting

**Decisions and why**
- Walk-forward, rolling 750-day window, monthly re-estimation, daily state updates; direct h-step ML models train only on targets observed by the fit date (leakage tests at feature and engine level).
- Diebold-Mariano with HAC variance (h-1 lags) and Harvey-Leybourne-Newbold correction; Holm adjustment across all 42 comparisons.

**Results**
- TTF and EUR/PLN: random walk not beaten at any horizon.
- TGE 1-day: linear model RMSE ratio 0.916, raw DM p 0.0022, but Holm p 0.0928 (does not survive); placebo with TTF and FX lagged one more day leaves 5.9% of the gain, so it comes from same-day TTF information.
- After Holm adjustment, no model beats the random walk anywhere.

**Likely questions**
1. *Is a random-walk win a failure?* No; for liquid futures and FX it is the expected result and a valid finding.
2. *How do you know there is no look-ahead?* Tests replace all data after t and check that every feature and every model's forecast at t is unchanged; the placebo confirms the gain depends on same-day (known) TTF moves.
3. *Why Holm?* With 42 model-horizon-series tests, a raw 5% threshold makes false wins likely; Holm controls the family-wise error rate.

## Stage 4 - Volatility

**Decisions and why**
- Mean equation chosen so standardised residuals pass Ljung-Box at lags 5, 10, 20 (needed for filtered historical simulation); AR(1) was not enough.
- Distribution chosen by BIC and two-sided 1% tail calibration; skewed-t tested for the basis.

**Results**
- Selected: TTF AR(5), TGE AR(10) with day-of-week, basis AR(5) with day-of-week, EUR/PLN constant; Student-t for all.
- Basis skewed-t: left 1% tail 0.0134 vs 0.0147 for t, but right tail 0.0134 vs 0.0099 and higher BIC; t kept.
- Volatility near-integrated for TTF (persistence 0.9992) and the basis; EWMA not beaten out of sample after Holm.

**Likely questions**
1. *Why Student-t?* Normal innovations put more than 1% of residuals below the fitted 1% quantile in every series; t fixes the calibration and has lower BIC.
2. *Why is EWMA hard to beat?* EWMA is an integrated GARCH without a constant, and the fitted persistence is close to one.
3. *How do you evaluate a volatility forecast without true variance?* Squared returns are a noisy but unbiased proxy; MSE and QLIKE rank forecasts consistently despite that noise (Patton 2011).

## Stage 5 - VaR and Expected Shortfall

**Decisions and why**
- Synthetic book: short floating TGE (500,000 MWh), long TTF futures (0.70), long EUR forward (0.70); TGE valued as TGE in EUR x EUR/PLN so an EUR/PLN rise raises the PLN cost.
- Six methods: historical simulation, normal, GARCH-t with Gaussian copula, GARCH-t with Student-t copula, FHS, EWMA; ES at 97.5% for FRTB.

**Results**
- Today (2026-11-06) official GARCH-t Gaussian-copula 99% VaR 18,323,060 PLN (see Stage 5 report for all methods).
- The t copula adds little: mean ratio to Gaussian 1.0075 for 99% VaR and 1.0092 for 97.5% ES, because the hedged book partly offsets joint moves.

**Likely questions**
1. *Why a t copula?* The Gaussian copula has no tail dependence, so it understates joint extremes; the test shows the t copula produces more joint tail events.
2. *Why ES at 97.5%?* FRTB replaced 99% VaR with 97.5% ES because ES captures tail severity and is subadditive.
3. *Why full revaluation?* The P&L is non-linear in prices and FX (TGE x FX product); sign-convention and leg-reconciliation tests enforce correctness.

## Stage 6 - VaR backtesting

**Results**
- 1,568 days. At 99%: normal and EWMA 30 exceptions vs 15.68 expected (conditional coverage p 0.0004 and 0.0048, fail); GARCH-t Gaussian 12, t copula 11, FHS 10 (pass); historical simulation 18 (pass).
- FHS has every rolling 250-day window in the green zone.

**Likely questions**
1. *What does Kupiec test, and what does it miss?* The exception count; it ignores clustering, which Christoffersen's independence test adds.
2. *EWMA was the best variance forecaster but fails at 99%. Contradiction?* No: variance accuracy and tail-quantile accuracy are different; EWMA uses normal tails.
3. *Why not adjust for multiple testing here?* Standard practice tests each model separately; with 12 tests the chance of one failing by luck is higher than 5%, which the report notes.

## Stage 7 - Limits

**Results**
- Official method by rule: GARCH-t Gaussian copula (passes conditional coverage at both levels, green share 0.9431).
- Today: VaR utilisation 0.7329; open gas exposure 1.0152 and FX net open 1.1556 (breach, driven by TGE price).
- TGE is the main contributor in all 33 VaR breach episodes; share of today's VaR 0.9108.

**Likely questions**
1. *How do you attribute VaR to risk factors?* Euler allocation: components s_k (Sigma s)_k / sigma_p sum to the total; rescaled to the fully revalued VaR.
2. *Why value-based exposure limits?* They bind when prices rise even if volumes do not change, which VaR alone can miss in calm periods.
3. *What happens at a breach?* Amber from 80% for early warning; breach above 100% escalates (Phase 3 framework document).

## Stage 8 - Stress testing

**Results**
- Sensitivities: -500,000 PLN per +1 PLN/MWh TGE; +1,537,375 PLN per +1 EUR/MWh TTF.
- Worst scenario: worst 21-day crisis move (TGE +355.0%, TTF +18.6%), loss 689,596,369 PLN, 37.64x today's 99% VaR; TGE leg 103.3% of the loss.
- Reverse stress: a relative TGE rise of 0.1231 alone loses the 25,000,000 PLN VaR limit.

**Likely questions**
1. *Why did the hedge not work in November 2022?* TGE day-ahead spiked while the TTF second month barely moved: tenor and location mismatch.
2. *Why reverse stress?* It turns a limit into an intuitive market move the board can judge.
3. *Why apply historical moves to today's book?* To ask what the past crisis would do to current positions, independent of the positions held then.

## Stage 9 - P&L attribution

**Results**
- Total P&L 1,565,811 PLN vs plan margin 10,000,000 PLN; price effect -7,936,717 PLN (purchases -38,096,776 PLN, TTF hedge +30,160,059 PLN); maximum daily residual 9.313e-10 PLN.

**Likely questions**
1. *How do you make attribution reconcile exactly?* Sequential decomposition: volume at plan prices, price at actual volume and plan FX, FX on actual volume and price; the interaction terms are assigned explicitly.
2. *Why does the hedge not fully offset the price effect?* 70% hedge ratio, TGE vs TTF basis and daily rebalancing to the remaining volume.
3. *How does this differ from VaR?* Attribution explains realised P&L after the fact; VaR forecasts potential loss.

## Stage 10 - Black-76 pricing

**Decisions and why**
- Black-76 because the underlying is a futures price; ICE Endex TTF options are futures-style margined, so the exchange value is undiscounted (the premium-paid value is kept for OTC options).
- Expiry by the ICE rule (5 calendar days before the delivery month, prior UK business day); rate from the ECB AAA spot curve.
- No free implied volatility: GARCH-t forecast as a labelled assumption, historical and EWMA as alternatives; values published as a share of F so no price level is committed.

**Results**
- GARCH-t volatility 0.5947; ATM call 8.5864% of F (7.6939% with historical, 9.1661% with EWMA volatility): the volatility choice moves the value by 1.4722% of F.
- Hull and Haug examples reproduced; maximum finite-difference relative error 3.529e-08.

**Likely questions**
1. *Why not discount the premium?* Futures-style margining settles the option value daily, so there is no upfront premium to discount.
2. *What is the biggest model risk?* The volatility input: no smile, and a time-series forecast instead of a market implied volatility.
3. *How did you validate the pricer?* Published examples, put-call parity, finite-difference Greeks and an implied-volatility round trip.

## Stage 11 - Hedging and effectiveness

**Results**
- h* 0.7840 on daily changes, 1.1143 on 21-day changes, latest rolling 0.8999.
- Futures at h* cut 21-day variance by 52.8% and 99% VaR by 44.0%, but crisis 99% VaR by only 12.5%; the collar removes less (41.0%) and costs 0.7711% of F.
- Dollar-offset fails (29.1% of 110 periods pass) while regression passes (R2 0.8439); the basis carries 99.1% of the ineffectiveness variance.

**Likely questions**
1. *Why is the daily hedge ratio lower than the monthly one?* Asynchronous day-ahead vs month-ahead prices and a mean-reverting basis: more of the TTF move reaches TGE over a month.
2. *Why do dollar-offset and regression disagree?* Dollar-offset judges each period alone and breaks when the TGE change is small (60.3% of failures); regression judges the relationship over many periods.
3. *Is this IFRS 9?* No: a simplified illustration without designation documentation or a hypothetical derivative.

## Stage 12 - Margin

**Results**
- IM 3,676,450 EUR (12.30% of notional); 10-day 99% buffer 7,085,393 EUR, 28.3% of the synthetic credit line.
- Largest stress call: the worst 10-day TTF fall of the crisis, 17,864,382 EUR (71.5% of the line).
- Historical check: 1.86% of 10-day moves exceed the buffer (target 1%), because history contains volatility far above today's.

**Likely questions**
1. *Why do the P&L stress scenarios bring cash in?* The hedge is long TTF; P&L risk is a TGE spike, liquidity risk is a TTF collapse.
2. *Why a driftless simulated price?* A futures price is a martingale under the pricing measure; drift would bias margin forecasts.
3. *How would you size the bank line?* On stressed price and volatility levels, not today's notional: at crisis levels the same volume needed a 10-day call of 1.69x the line.

## Stage 13 - Counterparty credit

**Results**
- CP04 (grade 5) breaches its PFE limit (utilisation 14.35); CP05 amber (0.96); netting lowers CP05's peak EE by 27.5%.
- Wrong-way risk documented: EE in the 10% highest-price paths is 4.56x to 5.92x the average for counterparties selling to the supplier.

**Likely questions**
1. *CE vs EE vs PFE?* CE is today's replacement cost, EE the mean future exposure, PFE a high quantile of it; limits use peak PFE.
2. *Why is wrong-way risk a problem here?* A gas seller loses on its forward exactly when prices spike, which is when the supplier's exposure to it peaks.
3. *What would you add first?* Collateral (CSA thresholds) and a price-dependent PD.

## Stage 14 - Risk framework

- [docs/risk_framework.md](risk_framework.md): taxonomy, methodologies, limit structure (VaR 25000000 PLN, open gas 60000000 PLN, FX 12000000 EUR, credit PFE by grade, bank line 25000000 EUR), amber from 0.80, escalation matrix, model validation table, limitations.
- *Likely question: who owns what?* Risk function owns market risk measures, treasury the FX and liquidity limits, credit risk the counterparty limits; repeated breaches go to the risk committee and Management Board.

## Stage 15 - Model handover

- Nine model cards, an input data specification with a machine-readable schema, 18 output acceptance criteria and `handover/validate_outputs.py` (exit code 1 on any error).
- *Likely question: what does a failed check do?* It stops publication of the daily report; a limit breach is a valid result handled by escalation, not a validation failure.

## Stage 16 - Pipeline and dashboard

- `python -m gasrisk.pipeline` runs 14 steps from refresh to report and stops at the first failure; Docker image; Airflow DAG with retries on API steps (cron documented as the simpler alternative).
- Star-schema exports with a synthetic label and no raw price levels; Power BI Project (TMDL model with 53 measures, PBIR report with 4 pages) validated without Desktop against Microsoft's JSON schemas and every DAX, visual and relationship reference.
- *Likely question: how do you know the dashboard works without opening it?* Structural validation plus deliberate error injection; DAX semantics and rendering still need the first-open checklist in the README.

## Stage 17 - Management memo

- One page, generated from the tables: TTF futures at 70% with the EUR forward (rule: highest 99% VaR reduction, lower ratio within 1 point), 45.2% reduction against 44.2% at the minimum-variance ratio; buffer funding about 184,269 EUR a year; stress call 71% of the line.
- *Likely question: why not the optimal ratio?* It adds no tail protection here and raises margin needs by 29%.

## Stage 18 - Checked LLM commentary

- Fact sheet of printed figures, Claude writes the commentary, every number must match a printed figure exactly, otherwise the deterministic template is published. The committed run used the template (SDK not installed).
- *Likely question: how do you stop an LLM inventing numbers?* Do not trust it: give it only printed figures and reject any number that is not one of them.

