# Model card - Black-76 option pricer (Stage 10)

- **Purpose.** Value and Greeks of European options on TTF futures (hedge collars, Stage 11).
- **Inputs.** TTF second-month future, ECB AAA spot curve (interpolated to expiry), volatility **assumption**.
- **Method.** Black-76; ICE Endex TTF options are futures-style margined, so the exchange value is undiscounted (the
  premium-paid value is available for OTC options). Expiry 5 calendar days before the delivery month (prior UK business
  day), ACT/365. Analytic delta, gamma, vega, theta, rho.
- **Volatility assumption.** No free TTF implied volatility: the input is the GARCH-t forecast to expiry (0.5947 on the
  as-of date), with 63-day historical and EWMA volatility as alternatives. Values are model values, not market prices.
  The volatility choice moves the ATM value by 1.4722% of F.
- **Validation.** Hull and Haug published examples reproduced; put-call parity; finite-difference Greeks (maximum
  relative difference 3.529e-08); implied-volatility round trip.
- **Limitations.** No smile or skew, so out-of-the-money wings are likely mispriced in stressed markets; single futures
  price, no curve.
- **Production change needed.** Replace the volatility assumption with ICE settlement volatilities by strike.
- **Owner.** Quantitative analyst.
