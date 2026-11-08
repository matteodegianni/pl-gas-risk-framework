# Model card - Hedge ratio and hedge effectiveness (Stage 11)

- **Purpose.** Size the TTF hedge of TGE purchases and monitor its effectiveness (simplified, **not IFRS 9 hedge
  accounting**).
- **Method.** Minimum-variance ratio of TGE (PLN) changes on TTF futures P&L in PLN, daily (rolling 250 days) and on
  non-overlapping 21-day blocks (rolling 36); strategy comparison on 21-day historical windows; dollar-offset
  (80-125%) and regression (R2 at least 0.80, slope 0.80-1.25); exact split of the ineffectiveness into basis and FX
  terms.
- **Result.** h* 0.7840 daily, 1.1143 on 21-day changes, latest rolling 0.8999. Futures at h* cut 21-day variance by
  52.8% but crisis 99% VaR by only 12.5%. Dollar-offset fails (29.1% of periods pass); regression passes (R2 0.8439);
  the basis carries 99.1% of the ineffectiveness variance.
- **Validation.** Identity tests; a zero-basis synthetic market passes dollar-offset with the EUR forward (test).
- **Limitations.** Daily TGE vs month-ahead TTF timing; collar priced with the volatility assumption; no transaction
  costs.
- **Owner.** Risk committee (policy); quantitative analyst (model).
