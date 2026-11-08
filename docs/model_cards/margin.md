# Model card - Margin forecast and liquidity buffer (Stage 12)

- **Purpose.** Forecast variation and initial margin on the TTF futures hedge and size the EUR liquidity buffer.
- **Method.** VM = position x daily TTF change. IM = 2-day 99% loss quantile by filtered historical simulation
  (refitted GARCH-t, bootstrapped standardised residuals, driftless price). Base forecast: 20000 paths over 10 days with
  IM recomputed along each path. Buffer = 99% quantile of 10-day cumulative outflow. Stress: Stage 8 scenarios, worst
  crisis TTF falls and a volatility-only stress.
- **Result.** IM 3,676,450 EUR (12.30% of notional); buffer 7,085,393 EUR (31,122,589 PLN), 28.3% of the synthetic
  credit line; largest stress call 17,864,382 EUR (71.5% of the line).
- **Validation.** Simulated IM against a closed-form approximation (criterion AC-14); historical check: 1.86% of
  10-day moves exceed the buffer against a 1% target, explained by crisis volatility above today's.
- **Limitations.** VaR proxy, not the ICE margin methodology; single contract; options excluded.
- **Owner.** Treasury (user); risk function (model owner).
