# Power BI dashboard (PBIP)

Power BI Project with a TMDL semantic model and a PBIR report, built on the CSVs in `../exports/`. **All figures are a
synthetic book on real market data**; every page shows this label.

The project was generated and validated on macOS without Power BI Desktop (see "What was validated"), so the first
open in Desktop is also its first real test: use the checklist below.

## Contents

```text
GasRisk.pbip                       open this file in Power BI Desktop
GasRisk.SemanticModel/             TMDL: 17 tables, 15 relationships, 53 DAX measures
  definition/model.tmdl            model settings and table list
  definition/expressions.tmdl      DataFolder parameter and fnLoadCsv loader
  definition/relationships.tmdl    star schema (many-to-one, single direction)
  definition/tables/*.tmdl         one file per table: columns, measures, Power Query partition
GasRisk.Report/                    PBIR: 4 pages, 24 visuals, custom theme
build_pbip.py                      regenerates both folders from exports/*.csv
validate_pbip.py                   validation without Power BI
set_data_folder.py                 switches the data source between GitHub and the local clone
```

## Star schema

| Dimension | Key | Facts |
|---|---|---|
| DimDate | date | FactVaR, FactLimitUtilisation, FactRiskContribution, FactAttribution, FactMarginForecast |
| DimMethod | method_key | FactVaR, FactBacktest |
| DimLimit | limit_key | FactLimitUtilisation |
| DimRiskFactor | factor_key | FactRiskContribution, FactStress |
| DimScenario | scenario_key | FactStress, FactMarginStress |
| DimDriver | driver_key | FactAttribution |
| DimCounterparty | counterparty | FactCredit, FactCreditProfile |

Measure groups: VaR and ES by method (and the official method), VaR limit, exceptions and expected exceptions, limit
utilisation, breach and amber days, VaR contribution by risk factor, stress loss by scenario and factor, margin forecast
quantiles, initial margin, liquidity buffer and credit-line use, stress margin calls, P&L attribution by driver and
cumulative P&L, counterparty exposure, PFE and limit use.

## Pages

1. **Risk overview**: official 99% VaR today, VaR limit, limits in breach; VaR against the limit over time;
   utilisation today and over time by limit; breach and amber days.
2. **Backtesting**: observed and expected 99% exceptions by method; Kupiec and Christoffersen p-values; realised P&L
   against the official VaR.
3. **Stress and margin**: stress loss by scenario split by risk factor; 10-day margin forecast (mean, 95%, 99%);
   initial margin, liquidity buffer and its share of the bank credit line; stress margin calls.
4. **P&L attribution**: contract P&L; P&L by driver; cumulative P&L; monthly P&L by driver group.

## How to open

Requires Power BI Desktop (Windows) with the PBIP and TMDL preview features turned on (File > Options and settings >
Options > Preview features: "Power BI Project (.pbip) save option", "Store semantic model using TMDL format", "Store
reports using enhanced metadata format (PBIR)"). Recent releases have these on by default.

1. Clone the repository.
2. Choose the data source:
   - **Default, no edits:** the `DataFolder` parameter points at `exports/` in the public GitHub repository, so the
     project opens straight after a clone (internet access needed).
   - **Local files:** run `python powerbi/set_data_folder.py` from the repository root. It writes the absolute path of
     this clone's `exports/` folder into the parameter. Power Query cannot resolve paths relative to the `.pbip` file,
     so this one-off step stands in for a relative path. `--github` switches back.
3. Open `powerbi/GasRisk.pbip`, then click **Refresh**. If Desktop asks about privacy levels for the web source, set
   it to Public (the data are public, synthetic risk tables).

To refresh the data after a pipeline run: `python -m gasrisk.pipeline` rewrites `exports/`; run
`python powerbi/build_pbip.py` only if export columns changed, then refresh in Desktop.

## What was validated without Power BI

`python powerbi/validate_pbip.py` (also run by `tests/test_powerbi.py`):

- All 34 JSON files (`.pbip`, `.pbir`, `.pbism`, report, pages, visuals) validate against the Microsoft schemas named
  in their `$schema`, with nested references resolved (from github.com/microsoft/json-schemas, cached locally).
- TMDL: tab indentation, a file for every table in `model.tmdl`, a partition for every table that loads an existing
  CSV, and every `sourceColumn` and type transform matching the CSV header.
- Names: measure names are unique across the model and match no column name in any table, compared
  case-insensitively as Power BI does. Desktop refused the first version of this project because the measure
  `Utilisation` clashed with the `utilisation` columns of FactLimitUtilisation and FactCredit; it is now
  `Limit Utilisation %`, and `tests/test_powerbi.py` re-inserts the old name to prove the validator rejects it.
- DAX: every `Table[Column]` and `[Measure]` reference resolves.
- Relationships: both columns exist, the one side is unique in the data, and every many-side key exists on it.
- Report: page folders match `pages.json`; every field used by a visual exists as a column or measure; every page has
  the synthetic-data label.
- The validator was tested by injecting errors (unknown column, unknown measure, invalid page property, extra visual
  property, broken relationship, a measure named like a column, a duplicate measure name); it reported all of them.

It cannot check DAX semantics, Power Query evaluation, or how the visuals render.

## What to check on first open

1. **The project opens** without a "cannot load" error. A first attempt failed on a measure named like a column
   (`Utilisation`); that is fixed and now checked. If Desktop reports another model error, add the rule to
   `validate_pbip.py` before fixing it, so it cannot come back. If the report fails on the theme, delete `themeCollection` and
   `resourcePackages` from `GasRisk.Report/definition/report.json` (Desktop then uses its default theme).
2. **Refresh succeeds** for all 17 tables (Home > Refresh). If types fail on a non-English locale, check that each
   partition's `Table.TransformColumnTypes` keeps the `"en-US"` culture argument.
3. **Model view** shows 15 relationships, all many-to-one from facts to dimensions, single direction, with no
   ambiguity warning. DimDate is marked as a date table (`date` as the key).
4. **Spot-check numbers** against `outputs/daily_risk_report.md`: Risk overview "Official 99% VaR today" equals the
   report's official 99% VaR; "Limits in breach today" equals the number of limits with status breach; the P&L page's
   contract P&L equals the report's total.
5. **Backtesting page**: observed 99% exceptions per method match `outputs/stage6_backtest.csv` (same sample).
6. **Visual formatting**: titles show and number formats look right (thousands separators, percentages).
7. Save once in Desktop. It may rewrite files to its own schema versions and add `.pbi/` settings (gitignored). Commit
   the rewrite if it is clean.
