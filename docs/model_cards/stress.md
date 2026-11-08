# Model card - Stress testing and sensitivities (Stage 8)

- **Purpose.** Loss of today's **synthetic** book under historical and hypothetical scenarios, sensitivities and
  reverse stress against the VaR limit.
- **Method.** Full revaluation; worst 1, 5, 10 and 21-day joint moves inside the 2021-2022 crisis window, three named
  episodes, hypothetical shocks (TTF +50% alone, gas complex plus or minus 50%, EUR/PLN +5%, combined, basis widening
  of 10 and 25 EUR/MWh), reverse stress by root finding.
- **Outputs.** `outputs/stage8_*`.
- **Validation.** Sensitivities equal the exact linear values (tests); the worst window is the minimum P&L window (test);
  reverse stress loses exactly the target (test).
- **Result.** -500,000 PLN per +1 PLN/MWh TGE; worst scenario the worst 21-day crisis move, 689,596,369 PLN (37.64x
  VaR), with the TGE leg at 103.3% of the loss; a TGE rise of 0.1231 alone loses the VaR limit.
- **Limitations.** Replays factor moves on today's levels; no liquidity or management actions; margin effects are in
  the margin model card.
- **Owner.** Risk function.
