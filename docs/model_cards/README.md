# Model cards

One card per model in the daily framework. Each card states purpose, inputs, method, outputs, validation evidence,
assumptions and limitations, monitoring and ownership. Every number is copied from the generated reports and checked by
`tests/test_docs.py`. Positions, limits, demand and counterparties are **synthetic**.

| Card | Model | Stage | Status |
|---|---|---|---|
| [forecasting.md](forecasting.md) | Price and FX forecasts | 3 | Research only, not used for limits |
| [volatility.md](volatility.md) | GARCH(1,1)-t volatility | 4 | Production input |
| [var_es.md](var_es.md) | VaR and ES (official GARCH-t Gaussian copula and challengers) | 5-7 | Production |
| [stress.md](stress.md) | Stress testing and sensitivities | 8 | Production |
| [attribution.md](attribution.md) | P&L attribution | 9 | Production |
| [black76.md](black76.md) | Black-76 option pricer | 10 | Production (volatility is an assumption) |
| [hedging.md](hedging.md) | Hedge ratio and effectiveness | 11 | Monthly monitoring |
| [margin.md](margin.md) | Margin forecast and liquidity buffer | 12 | Production |
| [credit.md](credit.md) | Counterparty exposure and limits | 13 | Production |
