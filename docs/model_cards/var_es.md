# Model card - VaR and Expected Shortfall (Stages 5-7)

- **Purpose.** Daily 1-day VaR (95%, 99%) and ES (95%, 97.5%, 99%) of the **synthetic** book in PLN; the official 99%
  VaR is the market risk limit measure.
- **Inputs.** TGE in EUR, TTF second month (roll-adjusted) and EUR/PLN (ECB); synthetic positions from `config.yaml`
  (500000 MWh open, TTF hedge 0.70, EUR forward 0.70).
- **Method.** Full revaluation of three legs. Official method **GARCH-t marginals with a Gaussian copula**
  (`garch_t_gauss`), 20000 simulations, refit every 21 days on 750 observations. Challengers: historical simulation,
  multivariate normal, EWMA, GARCH-t with a Student-t copula, filtered historical simulation.
- **Official method rule (Stage 7).** Passes conditional coverage at 95% and 99%, then the highest share of green
  250-day traffic-light windows.
- **Outputs.** `outputs/stage5_*`, `stage6_*`, `stage7_*`; Euler contributions by factor.
- **Validation.** Over 1,568 days: the official method passes conditional coverage at both levels and 0.9431 of
  250-day windows are green; normal and EWMA fail at 99% (30 exceptions against 15.68 expected).
- **Current result.** 99% VaR 18,323,060 PLN; utilisation 0.7329 of the synthetic limit; TGE share 0.9108.
- **Limitations.** Static MWh positions over the horizon; unhedged purchases valued at TGE day-ahead (location plus
  tenor basis); few 99% exceptions, so tests have limited power.
- **Monitoring.** Daily backtest update in the pipeline; validation criteria AC-02 to AC-06.
- **Owner.** Risk function (model owner); quantitative analyst (development).
