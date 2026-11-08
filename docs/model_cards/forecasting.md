# Model card - Price and FX forecasts (Stage 3)

- **Purpose.** Point forecasts of TTF, TGE, EUR/PLN and the TGE-TTF basis at 1, 5 and 21 business days, as a research
  benchmark for the risk models. Not used to set limits or hedges.
- **Inputs.** `panel.csv` (log prices; basis in EUR/MWh).
- **Method.** Random walk benchmark, ARIMA and ETS chosen by AIC, gradient boosting, and a linear direct model for TGE
  and the basis. Rolling 750-day window, refit every 21 trading days, daily state updates.
- **Outputs.** `outputs/stage3_*` (metrics, Diebold-Mariano tests, ablation, placebo).
- **Validation.** Walk-forward only; leakage tests (no feature or forecast at t changes when data after t change);
  Diebold-Mariano with Harvey-Leybourne-Newbold correction and Holm adjustment over 42 comparisons.
- **Result.** After Holm adjustment no model beats the random walk. The best raw result, TGE at 1 day (RMSE ratio
  0.916, raw DM p 0.0022), has Holm p 0.0928 and comes from same-day TTF and FX moves (5.9% of the gain survives the
  placebo).
- **Limitations.** Daily data only; the TGE effect partly reflects index construction (volume-weighted day-ahead vs
  end-of-day settlement).
- **Decision.** Keep the random walk as the forecast assumption in the risk models.
- **Owner.** Quantitative analyst (model owner); independent review by the risk function.
