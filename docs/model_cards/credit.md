# Model card - Counterparty exposure and limits (Stage 13)

- **Purpose.** CE, EE and PFE of OTC TTF forwards per **synthetic** counterparty, a scorecard grade and PFE limits.
- **Method.** Monte Carlo of TTF with the refitted GARCH(1,1)-t (5000 paths, daily steps, monthly grid over 24 months,
  flat curve, driftless); netting sets; discounting on the ECB curve. Scorecard: five synthetic ratios scored 1-5,
  weighted, mapped to six grades; the grade sets the limit on peak 95% PFE (amber from 0.80).
- **Result.** CP04 (grade 5) in breach with utilisation 14.35; CP05 amber (0.96); netting lowers CP05's peak EE by
  27.5%.
- **Validation.** Tests for scoring, trade value signs and expiry, CE, and netting never increasing exposure; criteria
  AC-15 and AC-16.
- **Wrong-way risk (limitation).** Exposure and default are independent in the model. For counterparties selling gas to
  the supplier, peak-month EE in the 10% highest-price paths is 4.56x to 5.92x the average: exactly when such
  counterparties are stressed. Mitigants: collateral, shorter tenors for weak grades, a price-dependent PD.
- **Other limitations.** No collateral or CSA; single factor; parameters from the last 750 days.
- **Owner.** Credit risk.
