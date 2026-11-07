# Project Brief - Polish Gas Supply Risk Framework

Portfolio project supporting an application for a Quantitative Analyst (Analysis & Risk) role at a Polish natural gas trader. It covers FX and energy price forecasting, market risk (VaR, limits, stress testing), margin forecasting, hedge effectiveness and valuation, counterparty credit risk, P&L attribution, a documented risk framework, model handover and management reporting.

## Business scenario (revised)

A Polish gas supplier:

- **Buys physical gas at the TGE day-ahead index (TGEgasDA, PLN/MWh)** on the Polish Power Exchange.
- **Sells to Polish B2B customers at fixed PLN prices** under contracts with an agreed volume profile. Actual customer demand deviates from contracted volumes (simulated, labelled **synthetic**).
- **Hedges with ICE Endex Dutch TTF futures and options, priced and margined in EUR**, because TTF is the liquid regional benchmark and TGE has no comparably liquid futures and options market.
- **Trades OTC forwards with counterparties** (synthetic counterparty set).

### Risk factors this creates

| Risk | Source in the scenario |
|---|---|
| Gas price risk | Fixed PLN sales against floating TGEgasDA purchases |
| Basis risk | TGE (PLN) and TTF (EUR) do not move one for one, so a TTF hedge leaves residual TGE exposure |
| EUR/PLN FX risk | Hedge P&L, option premia and margin calls are in EUR while the business reports in PLN; the TGE price itself embeds EUR/PLN through import parity |
| Margin liquidity risk | Daily variation margin and initial margin on TTF futures, funded in EUR |
| Counterparty credit risk | Replacement cost of OTC forwards if a counterparty defaults |

The original brief (TTF physical, TTF hedge) had no basis risk. The revised scenario is closer to how a Polish supplier is actually exposed and adds a TGE-TTF basis analysis.

## Data sources

| Series | Source | Access | Licence and handling |
|---|---|---|---|
| TGEgasDA daily, PLN/MWh, 2016 onward | Instrat API (data originate from TGE) | Fetched at runtime | CC BY-NC 4.0 (Instrat); TGE terms also apply. Never committed. |
| EUA primary auction clearing price, EUR/t, 2012 onward | Instrat API (data from EEX) | Fetched at runtime | Instrat states CC BY 4.0 unless stated otherwise; upstream EEX terms not reviewed. Never committed. |
| TTF front month (TFAc1, CME listing), EUR/MWh, 2017-10-23 onward | Investing.com manual export | User places CSV in `data/raw/investing/` | Investing.com terms prohibit redistribution. Never committed. |
| TTF second month (TFMBMc2, ICE Endex), EUR/MWh, 2010-03-15 onward | Investing.com manual export | Same | Same |
| EUR/PLN NBP mid fixing (primary) | NBP API | Fetched at runtime | Official public data |
| EUR/PLN ECB reference rate (cross-check) | ECB Data API | Fetched at runtime | Official public data |
| EUR/PLN Yahoo close (cross-check) | yfinance | Fetched at runtime | Not committed |

Committed outputs contain summary statistics, returns-based diagnostics and issue metadata, never raw price levels.

## Phases and stages

### Phase 1 - Core market risk
1. **Data and data quality**: loaders, calendar alignment (UK trading days for TTF, Polish business days for NBP, 7-day trading for TGE), TGE delivery-to-trade date shift, TTF roll handling with the c1/c2 pair, outliers with cross-series confirmation, cross-source FX checks, `data_quality_log.csv`.
2. **Diagnostics**: log returns, ADF and KPSS, Ljung-Box on returns and squared returns, ARCH-LM, skew, kurtosis, Jarque-Bera for TGE, TTF, basis and EUR/PLN.
3. **Forecasting**: TGE, TTF and EUR/PLN at 1-day, 1-week and 1-month horizons; random walk, ARIMA, ETS and gradient boosting; walk-forward backtest; RMSE, MAE, directional accuracy, Diebold-Mariano.
4. **Volatility**: GARCH(1,1) normal vs Student-t, evaluation against realised volatility.
5. **VaR and ES** for the book in PLN: historical simulation, parametric normal, GARCH-t, filtered historical simulation.
6. **VaR backtesting**: Kupiec and Christoffersen.
7. **Limits**: VaR and exposure limits, utilisation, breach detection and breach attribution by risk factor.
8. **Stress and sensitivity**: 2021-2022 gas crisis replay, hypothetical shocks (TTF +50%, TGE-TTF basis widening, EUR/PLN +5%, combined), sensitivities per 1 PLN/MWh TGE, per 1 EUR/MWh TTF and per 0.01 EUR/PLN.
9. **P&L attribution**: TGE price, TTF hedge, basis, volume (synthetic demand deviations) and FX effects, reconciling exactly to total P&L with an enforcing test.

### Phase 2 - Derivatives, margin and credit
10. Black-76 valuation of TTF futures options with Greeks, validated against reference values.
11. Hedging: minimum-variance hedge ratio of TGE (in PLN) on TTF (in EUR) plus EUR/PLN, strategy comparison (no hedge, futures, collar, with and without FX hedge), dollar-offset and regression effectiveness. Simplified test, not IFRS 9 hedge accounting.
12. Margin forecasting: variation margin from mark-to-market, VaR-based initial margin over a margin period of risk, base and stress margin calls, liquidity buffer in EUR and PLN.
13. Counterparty credit risk: CE, EE and PFE for OTC forwards by Monte Carlo, synthetic counterparty scoring, score and PFE mapped to limits, utilisation and breach reporting.

### Phase 3 - Framework, delivery and communication
14. `docs/risk_framework.md`.
15. Model handover package: model cards, input data specification, acceptance criteria, output validation script.
16. Daily risk pipeline in one command, Docker, Airflow DAG or a justified simpler scheduler, Power BI ready CSVs.
17. `docs/management_memo.md`, one page for the CFO and Management Board.
18. Optional LLM risk commentary with every number checked against source tables.

## Working rules

- Every figure in documentation comes from code output; no invented or rounded numbers.
- Synthetic data (counterparties, demand, positions) is labelled **synthetic** wherever it appears.
- Where data cannot support a step, the limitation is stated and an alternative proposed.
- Only hyphens are used in text, never long dashes.
