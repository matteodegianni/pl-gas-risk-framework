# Data folder

Nothing in `data/raw/` or `data/clean/` is committed: the TGE (via Instrat) and Investing.com data may not be redistributed.

## Downloaded automatically by `python -m gasrisk.data.build`

| Raw file | Source |
|---|---|
| `raw/instrat/tge_gas_da.json` | Instrat API, TGEgasDA (data from TGE), CC BY-NC 4.0 |
| `raw/instrat/eua_auctions.json` | Instrat API, EU ETS primary auctions (data from EEX) |
| `raw/nbp/eurpln_table_a.json` | NBP API, table A EUR mid rate |
| `raw/ecb/eurpln_exr.csv` | ECB Data API, EXR D.PLN.EUR.SP00.A |
| `raw/yahoo/eurpln.csv` | Yahoo Finance EURPLN=X |

`raw/manifest.json` records the URL, parameters, download time, row count and SHA-256 of every raw file.

## Manual exports (place in `raw/investing/`)

From the English Investing.com site, daily frequency, maximum date range, "Download":

| File name | Page | Contract |
|---|---|---|
| `ttf_c1.csv` | Dutch TTF Natural Gas Futures historical data | TFAc1, front month, CME listing |
| `ttf_c2.csv` | ICE Dutch TTF second-month historical data | TFMBMc2, second month, ICE Endex |

Expected columns: `Date, Price, Open, High, Low, Vol., Change %` with dates as MM/DD/YYYY.

## Clean outputs (`clean/`)

`panel.csv` is the aligned business-day panel used from Stage 2 onward; `ttf.csv`, `tge_gas_da.csv`, `eurpln.csv` and `eua_auctions.csv` are the cleaned source series.
