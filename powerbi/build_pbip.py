"""Generate the Power BI Project (PBIP): TMDL semantic model and PBIR report, from the exported CSVs.

Usage (from the repository root)::

    python powerbi/build_pbip.py          # regenerate powerbi/GasRisk.* from exports/*.csv
    python powerbi/validate_pbip.py       # validate structure, schemas and every reference

The model is a star schema on ``exports/``: dimensions DimDate, DimMethod, DimRiskFactor, DimLimit, DimDriver,
DimScenario, DimCounterparty and one fact table per exported fact. Column names and types come from the CSV headers,
so the model cannot drift from the export. All data are a **synthetic** book on real market data.

Data location: Power Query has no relative file paths, so the model reads every CSV through one parameter,
``DataFolder``. Its default is the ``exports/`` folder of the public GitHub repository (relative to the repository root),
so the project opens after a clone with no edits; ``python powerbi/set_data_folder.py`` switches it to the local
``exports/`` folder of the clone.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PBI = ROOT / "powerbi"
EXPORTS = ROOT / "exports"
NAME = "GasRisk"
REPO_RAW = "https://raw.githubusercontent.com/matteodegianni/pl-gas-risk-framework/master/exports/"
NS = uuid.UUID("6f1d3a52-6d55-4b8e-9d0e-6a7c2f3b9a10")
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"

TABLES = {  # model table -> export file
    "DimDate": "dim_date", "DimMethod": "dim_method", "DimRiskFactor": "dim_risk_factor", "DimLimit": "dim_limit",
    "DimDriver": "dim_driver", "DimScenario": "dim_scenario", "DimCounterparty": "dim_counterparty",
    "FactVaR": "fact_var", "FactBacktest": "fact_backtest", "FactLimitUtilisation": "fact_limit_utilisation",
    "FactRiskContribution": "fact_risk_contribution", "FactStress": "fact_stress", "FactMarginForecast": "fact_margin_forecast",
    "FactMarginStress": "fact_margin_stress", "FactAttribution": "fact_attribution", "FactCredit": "fact_credit",
    "FactCreditProfile": "fact_credit_profile",
}
DATE_COLUMNS = {"date", "target"}
RELATIONSHIPS = [  # (from table, from column) many-to-one (to table, to column)
    ("FactVaR", "date", "DimDate", "date"), ("FactVaR", "method_key", "DimMethod", "method_key"),
    ("FactBacktest", "method_key", "DimMethod", "method_key"),
    ("FactLimitUtilisation", "date", "DimDate", "date"), ("FactLimitUtilisation", "limit_key", "DimLimit", "limit_key"),
    ("FactRiskContribution", "date", "DimDate", "date"), ("FactRiskContribution", "factor_key", "DimRiskFactor", "factor_key"),
    ("FactStress", "scenario_key", "DimScenario", "scenario_key"), ("FactStress", "factor_key", "DimRiskFactor", "factor_key"),
    ("FactMarginStress", "scenario_key", "DimScenario", "scenario_key"),
    ("FactMarginForecast", "date", "DimDate", "date"),
    ("FactAttribution", "date", "DimDate", "date"), ("FactAttribution", "driver_key", "DimDriver", "driver_key"),
    ("FactCredit", "counterparty", "DimCounterparty", "counterparty"),
    ("FactCreditProfile", "counterparty", "DimCounterparty", "counterparty"),
]

# DAX measures: table -> list of (name, expression, format string). Every column reference is checked by the validator.
MEASURES: dict[str, list[tuple[str, str, str]]] = {
    "FactVaR": [
        ("VaR 95", "AVERAGE ( FactVaR[var_95] )", "#,0"),
        ("VaR 99", "AVERAGE ( FactVaR[var_99] )", "#,0"),
        ("ES 97.5", "AVERAGE ( FactVaR[es_97_5] )", "#,0"),
        ("Official VaR 99", "CALCULATE ( [VaR 99], DimMethod[is_official] = 1 )", "#,0"),
        ("Official ES 97.5", "CALCULATE ( [ES 97.5], DimMethod[is_official] = 1 )", "#,0"),
        ("Latest Official VaR 99",
         "VAR d = CALCULATE ( MAX ( FactVaR[date] ), ALL ( FactVaR ), ALL ( DimDate ) )\n"
         "RETURN CALCULATE ( [Official VaR 99], ALL ( DimDate ), FactVaR[date] = d )", "#,0"),
        ("VaR Limit", "CALCULATE ( MAX ( DimLimit[limit_value] ), DimLimit[limit_key] = \"var_99\" )", "#,0"),
        ("Realised P&L", "SUM ( FactVaR[realised_pnl_next_day] )", "#,0"),
        ("Exceptions 95", "SUM ( FactVaR[exception_95] )", "0"),
        ("Exceptions 99", "SUM ( FactVaR[exception_99] )", "0"),
        ("Expected Exceptions 99", "COUNT ( FactVaR[realised_pnl_next_day] ) * 0.01", "0.0"),
        ("Exception Rate 99", "DIVIDE ( [Exceptions 99], COUNT ( FactVaR[realised_pnl_next_day] ) )", "0.00%"),
        ("Data Label", "\"Synthetic book on real market data\"", ""),
    ],
    "FactBacktest": [
        ("Observed Exceptions", "SUM ( FactBacktest[observed] )", "0"),
        ("Expected Exceptions", "SUM ( FactBacktest[expected] )", "0.0"),
        ("Kupiec p", "AVERAGE ( FactBacktest[p_pof] )", "0.0000"),
        ("Conditional Coverage p", "AVERAGE ( FactBacktest[p_cc] )", "0.0000"),
        ("Tests Passed", "COUNTROWS ( FILTER ( FactBacktest, FactBacktest[pass_cc] = TRUE () ) )", "0"),
    ],
    "FactLimitUtilisation": [
        ("Limit Utilisation %", "AVERAGE ( FactLimitUtilisation[utilisation] )", "0.00%"),
        ("Latest Utilisation",
         "VAR d = CALCULATE ( MAX ( FactLimitUtilisation[date] ), ALL ( FactLimitUtilisation ), ALL ( DimDate ) )\n"
         "RETURN CALCULATE ( [Limit Utilisation %], ALL ( DimDate ), FactLimitUtilisation[date] = d )", "0.00%"),
        ("Breach Days", "SUM ( FactLimitUtilisation[breach] )", "0"),
        ("Amber Days", "COUNTROWS ( FILTER ( FactLimitUtilisation, FactLimitUtilisation[status] = \"amber\" ) )", "0"),
        ("Limits In Breach Today",
         "VAR d = CALCULATE ( MAX ( FactLimitUtilisation[date] ), ALL ( FactLimitUtilisation ), ALL ( DimDate ) )\n"
         "RETURN CALCULATE ( SUM ( FactLimitUtilisation[breach] ), ALL ( DimDate ), FactLimitUtilisation[date] = d )", "0"),
    ],
    "FactRiskContribution": [
        ("VaR Contribution", "SUM ( FactRiskContribution[var99_contribution_pln] )", "#,0"),
        ("VaR Contribution Share", "DIVIDE ( [VaR Contribution], CALCULATE ( [VaR Contribution], ALL ( DimRiskFactor ) ) )", "0.0%"),
    ],
    "FactStress": [
        ("Stress P&L", "SUM ( FactStress[pnl_pln] )", "#,0"),
        ("Stress Loss", "- [Stress P&L]", "#,0"),
        ("Stress Loss Over VaR", "MAX ( FactStress[loss_over_var99] )", "0.00"),
    ],
    "FactMarginForecast": [
        ("Margin Outflow Mean", "SUM ( FactMarginForecast[mean_cum_outflow_eur] )", "#,0"),
        ("Margin Outflow 95", "SUM ( FactMarginForecast[q95_cum_outflow_eur] )", "#,0"),
        ("Margin Outflow 99", "SUM ( FactMarginForecast[q99_cum_outflow_eur] )", "#,0"),
        ("Liquidity Buffer", "MAX ( FactMarginForecast[buffer_10d_q99_eur] )", "#,0"),
        ("Initial Margin Today", "MAX ( FactMarginForecast[im_today_eur] )", "#,0"),
        ("Bank Credit Line", "MAX ( FactMarginForecast[bank_credit_line_eur] )", "#,0"),
        ("Buffer Over Credit Line", "DIVIDE ( [Liquidity Buffer], [Bank Credit Line] )", "0.0%"),
    ],
    "FactMarginStress": [
        ("Stress Margin Call", "SUM ( FactMarginStress[margin_call_eur] )", "#,0"),
        ("Stress Margin Call PLN", "SUM ( FactMarginStress[margin_call_pln] )", "#,0"),
        ("Credit Line Utilisation", "MAX ( FactMarginStress[credit_line_utilisation] )", "0.0%"),
    ],
    "FactAttribution": [
        ("Attributed P&L", "SUM ( FactAttribution[pnl_pln] )", "#,0"),
        ("Total P&L", "CALCULATE ( [Attributed P&L], ALL ( DimDriver ) )", "#,0"),
        ("Price Effect", "CALCULATE ( [Attributed P&L], DimDriver[driver_group] = \"price\" )", "#,0"),
        ("FX Effect", "CALCULATE ( [Attributed P&L], DimDriver[driver_group] = \"fx\" )", "#,0"),
        ("Volume Effect", "CALCULATE ( [Attributed P&L], DimDriver[driver_group] = \"volume\" )", "#,0"),
        ("Plan Margin", "CALCULATE ( [Attributed P&L], DimDriver[driver_group] = \"plan\" )", "#,0"),
        ("Cumulative P&L",
         "VAR d = MAX ( DimDate[date] )\n"
         "RETURN CALCULATE ( [Attributed P&L], FILTER ( ALL ( DimDate[date] ), DimDate[date] <= d ) )", "#,0"),
    ],
    "FactCredit": [
        ("Current Exposure", "SUM ( FactCredit[ce_eur] )", "#,0"),
        ("Peak PFE 95", "SUM ( FactCredit[peak_pfe_95_eur] )", "#,0"),
        ("PFE Limit", "SUM ( FactCredit[pfe_limit_eur] )", "#,0"),
        ("Credit Limit Utilisation", "MAX ( FactCredit[utilisation] )", "0.00"),
        ("Counterparties In Breach", "COUNTROWS ( FILTER ( FactCredit, FactCredit[status] = \"breach\" ) )", "0"),
    ],
    "FactCreditProfile": [
        ("Expected Exposure", "SUM ( FactCreditProfile[ee_eur] )", "#,0"),
        ("PFE 95", "SUM ( FactCreditProfile[pfe_95_eur] )", "#,0"),
        ("PFE 97.5", "SUM ( FactCreditProfile[pfe_97_5_eur] )", "#,0"),
    ],
}


def gid(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


def q(name: str) -> str:
    """Quote a TMDL object name if needed."""
    return name if name.replace("_", "").isalnum() else "'" + name.replace("'", "''") + "'"


def column_types(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """TMDL dataType and Power Query type per column."""
    out = {}
    for c in df.columns:
        if c in DATE_COLUMNS:
            out[c] = ("dateTime", "type date")
        elif pd.api.types.is_bool_dtype(df[c]):
            out[c] = ("boolean", "type logical")
        elif pd.api.types.is_integer_dtype(df[c]):
            out[c] = ("int64", "Int64.Type")
        elif pd.api.types.is_float_dtype(df[c]):
            out[c] = ("double", "type number")
        else:
            out[c] = ("string", "type text")
    return out


def table_tmdl(table: str, file: str, df: pd.DataFrame) -> str:
    types = column_types(df)
    lines = [f"table {table}", f"\tlineageTag: {gid(table)}"]
    if table == "DimDate":
        lines.append("\tdataCategory: Time")
    lines.append("")
    for name, expr, fmt in MEASURES.get(table, []):
        body = expr.split("\n")
        if len(body) == 1:
            lines.append(f"\tmeasure {q(name)} = {body[0]}")
        else:
            lines.append(f"\tmeasure {q(name)} =")
            lines += [f"\t\t\t{b}" for b in body]
        if fmt:
            lines.append(f"\t\tformatString: {fmt}")
        lines += [f"\t\tlineageTag: {gid(table, 'measure', name)}", ""]
    for c, (dt, _) in types.items():
        lines.append(f"\tcolumn {q(c)}")
        lines.append(f"\t\tdataType: {dt}")
        if table == "DimDate" and c == "date":
            lines.append("\t\tisKey")
        if dt == "dateTime":
            lines.append("\t\tformatString: yyyy-mm-dd")
        lines.append(f"\t\tlineageTag: {gid(table, c)}")
        lines.append("\t\tsummarizeBy: none" if dt in ("string", "dateTime", "boolean") or c.endswith("_key") or c in
                     ("year", "month", "grade", "day", "is_official", "is_weekend") else "\t\tsummarizeBy: sum")
        lines.append(f"\t\tsourceColumn: {c}")
        lines += ["", "\t\tannotation SummarizationSetBy = Automatic", ""]
    type_list = ", ".join(f'{{"{c}", {pq}}}' for c, (_, pq) in types.items())
    lines += [
        f"\tpartition {table} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\t\tlet",
        f'\t\t\t\t    Source = fnLoadCsv("{file}"),',
        f'\t\t\t\t    Typed = Table.TransformColumnTypes(Source, {{{type_list}}}, "en-US")',
        "\t\t\t\tin",
        "\t\t\t\t    Typed",
        "",
        "\tannotation PBI_ResultType = Table",
        "",
    ]
    return "\n".join(lines)


def semantic_model(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    files = {}
    files["definition/database.tmdl"] = "database\n\tcompatibilityLevel: 1567\n\n"
    order = json.dumps(["DataFolder", "fnLoadCsv", *TABLES])
    files["definition/model.tmdl"] = "\n".join([
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US",
        "\tdataAccessOptions",
        "\t\tlegacyRedirects",
        "\t\treturnErrorValuesAsNull",
        "",
        f"annotation PBI_QueryOrder = {order}",
        "",
        "annotation __PBI_TimeIntelligenceEnabled = 0",
        "",
        *[f"ref table {t}" for t in TABLES],
        "",
    ])
    files["definition/expressions.tmdl"] = "\n".join([
        f'expression DataFolder = "{REPO_RAW}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
        f"\tlineageTag: {gid('DataFolder')}",
        "",
        "\tannotation PBI_ResultType = Text",
        "",
        "expression fnLoadCsv =",
        "\t\t(name as text) as table =>",
        "\t\tlet",
        '\t\t    Path = DataFolder & name & ".csv",',
        '\t\t    Raw = if Text.StartsWith(DataFolder, "http") then Web.Contents(Path) else File.Contents(Path),',
        '\t\t    Csv = Csv.Document(Raw, [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),',
        "\t\t    Promoted = Table.PromoteHeaders(Csv, [PromoteAllScalars = true])",
        "\t\tin",
        "\t\t    Promoted",
        f"\tlineageTag: {gid('fnLoadCsv')}",
        "",
        "\tannotation PBI_ResultType = Function",
        "",
    ])
    rel = []
    for ft, fc, tt, tc in RELATIONSHIPS:
        rel += [f"relationship {gid('rel', ft, fc, tt, tc)}", f"\tfromColumn: {ft}.{q(fc)}", f"\ttoColumn: {tt}.{q(tc)}", ""]
    files["definition/relationships.tmdl"] = "\n".join(rel)
    for t, f in TABLES.items():
        files[f"definition/tables/{t}.tmdl"] = table_tmdl(t, f, frames[t])
    files["definition.pbism"] = json.dumps({"$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json",
                                            "version": "4.2", "settings": {}}, indent=2) + "\n"
    return files


# --------------------------------------------------------------------------- report (PBIR)


def col(table: str, column: str) -> dict:
    return {"field": {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": column}},
            "queryRef": f"{table}.{column}", "nativeQueryRef": column}


def mea(table: str, measure: str) -> dict:
    return {"field": {"Measure": {"Expression": {"SourceRef": {"Entity": table}}, "Property": measure}},
            "queryRef": f"{table}.{measure}", "nativeQueryRef": measure}


def visual(name: str, vtype: str, x: int, y: int, w: int, h: int, roles: dict[str, list[dict]], title: str, z: int) -> dict:
    return {
        "$schema": f"{SCHEMA}/item/report/definition/visualContainer/1.0.0/schema.json",
        "name": name,
        "position": {"x": x, "y": y, "z": z, "width": w, "height": h, "tabOrder": z},
        "visual": {
            "visualType": vtype,
            "query": {"queryState": {role: {"projections": p} for role, p in roles.items()}},
            "visualContainerObjects": {"title": [{"properties": {
                "show": {"expr": {"Literal": {"Value": "true"}}},
                "text": {"expr": {"Literal": {"Value": "'" + title.replace("'", "''") + "'"}}}}}]},
            "drillFilterOtherVisuals": True,
        },
    }


def label_card(page: str, z: int) -> dict:
    return visual(f"{page}_label", "card", 900, 10, 360, 70, {"Values": [mea("FactVaR", "Data Label")]},
                  "Data: synthetic book on real market data", z)


PAGES = [
    ("risk_overview", "Risk overview", lambda p: [
        label_card(p, 0),
        visual(f"{p}_var_card", "card", 20, 10, 270, 110, {"Values": [mea("FactVaR", "Latest Official VaR 99")]},
               "Official 99% VaR today (PLN)", 1),
        visual(f"{p}_limit_card", "card", 300, 10, 270, 110, {"Values": [mea("FactVaR", "VaR Limit")]}, "VaR limit (PLN)", 2),
        visual(f"{p}_breach_card", "card", 580, 10, 300, 110, {"Values": [mea("FactLimitUtilisation", "Limits In Breach Today")]},
               "Limits in breach today", 3),
        visual(f"{p}_var_line", "lineChart", 20, 130, 820, 330,
               {"Category": [col("DimDate", "date")], "Y": [mea("FactVaR", "Official VaR 99"), mea("FactVaR", "VaR Limit")]},
               "Official 99% VaR against the limit (PLN)", 4),
        visual(f"{p}_util_bar", "clusteredBarChart", 860, 130, 400, 330,
               {"Category": [col("DimLimit", "limit_name")], "Y": [mea("FactLimitUtilisation", "Latest Utilisation")]},
               "Limit utilisation today", 5),
        visual(f"{p}_util_line", "lineChart", 20, 470, 820, 240,
               {"Category": [col("DimDate", "date")], "Y": [mea("FactLimitUtilisation", "Limit Utilisation %")],
                "Series": [col("DimLimit", "limit_name")]}, "Utilisation history by limit", 6),
        visual(f"{p}_breach_table", "tableEx", 860, 470, 400, 240,
               {"Values": [col("DimLimit", "limit_name"), mea("FactLimitUtilisation", "Breach Days"),
                           mea("FactLimitUtilisation", "Amber Days")]}, "Breach and amber days", 7),
    ]),
    ("backtesting", "Backtesting", lambda p: [
        label_card(p, 0),
        visual(f"{p}_exc_col", "clusteredColumnChart", 20, 90, 820, 300,
               {"Category": [col("DimMethod", "method_name")],
                "Y": [mea("FactVaR", "Exceptions 99"), mea("FactVaR", "Expected Exceptions 99")]},
               "99% VaR exceptions by method: observed and expected", 1),
        visual(f"{p}_tests", "tableEx", 860, 90, 400, 300,
               {"Values": [col("DimMethod", "method_name"), col("FactBacktest", "level"),
                           mea("FactBacktest", "Observed Exceptions"), mea("FactBacktest", "Expected Exceptions"),
                           mea("FactBacktest", "Kupiec p"), mea("FactBacktest", "Conditional Coverage p")]},
               "Kupiec and Christoffersen tests", 2),
        visual(f"{p}_pnl_line", "lineChart", 20, 400, 1240, 310,
               {"Category": [col("DimDate", "date")], "Y": [mea("FactVaR", "Realised P&L"), mea("FactVaR", "Official VaR 99")]},
               "Realised next-day P&L and official 99% VaR (PLN)", 3),
    ]),
    ("stress_margin", "Stress and margin", lambda p: [
        label_card(p, 0),
        visual(f"{p}_stress_bar", "barChart", 20, 90, 620, 330,
               {"Category": [col("DimScenario", "scenario")], "Y": [mea("FactStress", "Stress Loss")],
                "Series": [col("DimRiskFactor", "factor_name")]}, "Stress loss by scenario and risk factor (PLN)", 1),
        visual(f"{p}_margin_col", "lineChart", 660, 90, 600, 330,
               {"Category": [col("FactMarginForecast", "day")],
                "Y": [mea("FactMarginForecast", "Margin Outflow Mean"), mea("FactMarginForecast", "Margin Outflow 95"),
                      mea("FactMarginForecast", "Margin Outflow 99")]},
               "Margin forecast: cumulative outflow over 10 days (EUR)", 2),
        visual(f"{p}_im_card", "card", 660, 430, 290, 100, {"Values": [mea("FactMarginForecast", "Initial Margin Today")]},
               "Initial margin today (EUR)", 3),
        visual(f"{p}_buffer_card", "card", 970, 430, 290, 100, {"Values": [mea("FactMarginForecast", "Liquidity Buffer")]},
               "10-day 99% liquidity buffer (EUR)", 4),
        visual(f"{p}_mstress_bar", "clusteredBarChart", 20, 430, 620, 280,
               {"Category": [col("DimScenario", "scenario")], "Y": [mea("FactMarginStress", "Stress Margin Call")]},
               "Stress margin calls (EUR)", 5),
        visual(f"{p}_line_card", "card", 660, 540, 600, 100, {"Values": [mea("FactMarginForecast", "Buffer Over Credit Line")]},
               "Buffer as a share of the bank credit line", 6),
    ]),
    ("pnl_attribution", "P&L attribution", lambda p: [
        label_card(p, 0),
        visual(f"{p}_total_card", "card", 20, 10, 270, 110, {"Values": [mea("FactAttribution", "Total P&L")]},
               "Contract P&L since signing (PLN)", 1),
        visual(f"{p}_driver_bar", "clusteredBarChart", 20, 130, 560, 300,
               {"Category": [col("DimDriver", "driver_name")], "Y": [mea("FactAttribution", "Attributed P&L")]},
               "P&L by driver (PLN)", 2),
        visual(f"{p}_cum_line", "lineChart", 600, 130, 660, 300,
               {"Category": [col("DimDate", "date")], "Y": [mea("FactAttribution", "Cumulative P&L")]},
               "Cumulative contract P&L (PLN)", 3),
        visual(f"{p}_monthly", "columnChart", 20, 440, 1240, 270,
               {"Category": [col("DimDate", "year_month")], "Y": [mea("FactAttribution", "Attributed P&L")],
                "Series": [col("DimDriver", "driver_group")]}, "Monthly P&L by driver group (PLN)", 4),
    ]),
]


def report() -> dict[str, str]:
    files = {}
    files["definition.pbir"] = json.dumps({"$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
                                           "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}},
                                          indent=2) + "\n"
    files["definition/version.json"] = json.dumps({"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
                                                   "version": "1.0.0"}, indent=2) + "\n"
    files["definition/report.json"] = json.dumps({
        "$schema": f"{SCHEMA}/item/report/definition/report/1.0.0/schema.json",
        "themeCollection": {"customTheme": {"name": "GasRiskTheme.json", "reportVersionAtImport": "5.55", "type": "RegisteredResources"}},
        "layoutOptimization": "None",
        "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources",
                              "items": [{"name": "GasRiskTheme.json", "path": "GasRiskTheme.json", "type": "CustomTheme"}]}],
    }, indent=2) + "\n"
    files["StaticResources/RegisteredResources/GasRiskTheme.json"] = json.dumps({
        "name": "GasRiskTheme",
        "dataColors": ["#1F4E79", "#C55A11", "#548235", "#7F6000", "#7030A0", "#2E75B6", "#BF9000", "#A5A5A5"],
        "background": "#FFFFFF", "foreground": "#252423", "tableAccent": "#1F4E79",
    }, indent=2) + "\n"
    files["definition/pages/pages.json"] = json.dumps({"$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
                                                       "pageOrder": [p[0] for p in PAGES], "activePageName": PAGES[0][0]}, indent=2) + "\n"
    for pname, display, build in PAGES:
        files[f"definition/pages/{pname}/page.json"] = json.dumps({
            "$schema": f"{SCHEMA}/item/report/definition/page/1.0.0/schema.json",
            "name": pname, "displayName": display, "displayOption": "FitToPage", "height": 720, "width": 1280}, indent=2) + "\n"
        for v in build(pname):
            files[f"definition/pages/{pname}/visuals/{v['name']}/visual.json"] = json.dumps(v, indent=2) + "\n"
    return files


def write(base: Path, files: dict[str, str]) -> None:
    if base.exists():
        shutil.rmtree(base)
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def main() -> None:
    frames = {t: pd.read_csv(EXPORTS / f"{f}.csv") for t, f in TABLES.items()}
    write(PBI / f"{NAME}.SemanticModel", semantic_model(frames))
    write(PBI / f"{NAME}.Report", report())
    (PBI / f"{NAME}.pbip").write_text(json.dumps({
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}}, indent=2) + "\n", encoding="utf-8")
    (PBI / ".gitignore").write_text("**/.pbi/localSettings.json\n**/.pbi/cache.abf\n.schema_cache/\n", encoding="utf-8")
    n_meas = sum(len(v) for v in MEASURES.values())
    print(f"{len(TABLES)} tables, {len(RELATIONSHIPS)} relationships, {n_meas} measures, {len(PAGES)} pages")


if __name__ == "__main__":
    main()
