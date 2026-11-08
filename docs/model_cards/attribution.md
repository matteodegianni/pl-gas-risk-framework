# Model card - P&L attribution (Stage 9)

- **Purpose.** Explain daily P&L of a **synthetic** one-year fixed-price supply contract by driver.
- **Method.** Sequential exact split: plan margin, volume effect (synthetic AR(1) demand deviations), price effect
  (TGE purchases and TTF hedge at plan FX), FX effect (purchase cost, hedge translation, EUR forward).
- **Outputs.** `outputs/stage9_*`, `data/clean/attribution_daily.csv`.
- **Validation.** Reconciliation enforced in code (the run fails otherwise) and by tests; maximum daily residual
  9.313e-10 PLN.
- **Result.** Total P&L 1,565,811 PLN against a plan margin of 10,000,000 PLN; price effect -7,936,717 PLN (purchases
  -38,096,776 PLN, TTF hedge +30,160,059 PLN).
- **Limitations.** Order-dependent sequential split (price before FX); synthetic demand.
- **Owner.** Finance controlling (user); risk function (model owner).
