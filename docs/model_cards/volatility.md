# Model card - GARCH(1,1)-t volatility (Stage 4)

- **Purpose.** Conditional volatility of TTF, TGE, EUR/PLN and the basis; marginals for VaR (Stage 5), the option
  volatility assumption (Stage 10), initial margin (Stage 12) and credit exposure (Stage 13).
- **Inputs.** Daily log returns from `panel.csv` (TTF roll-adjusted, held to roll).
- **Method.** GARCH(1,1) with normal and Student-t innovations (skewed-t also tested for the basis). The constant mean
  is kept only if standardised residuals pass Ljung-Box at lags 5, 10 and 20; otherwise the lowest-BIC passing
  candidate. Selected: TTF AR(5), TGE AR(10) with day-of-week, basis AR(5) with day-of-week, EUR/PLN constant; Student-t
  for all four.
- **Outputs.** `outputs/stage4_*`, `data/clean/garch_conditional_vol.csv`, `garch_std_resid.csv`.
- **Validation.** Out-of-sample 1, 5 and 21-day variance forecasts against EWMA and a 63-day rolling variance (MSE,
  QLIKE, Diebold-Mariano with Holm, Mincer-Zarnowitz). Ljung-Box p at lag 10 moves from 0.0039 to 0.5927 for TTF with
  the selected mean.
- **Result.** EWMA is not beaten after Holm adjustment; TTF persistence 0.9992 (near-integrated).
- **Limitations.** Specification chosen on the full sample (first-window check reported; EUR/PLN differs); persistence
  may reflect structural breaks.
- **Monitoring.** Monthly refit; residual Ljung-Box and the VaR backtest are the ongoing checks.
- **Owner.** Quantitative analyst.
