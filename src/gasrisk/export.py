"""Stage 16: export daily risk tables as CSVs for the Power BI dashboard (star schema).

Usage::

    python -m gasrisk.export

Writes ``exports/`` (committed, so the dashboard opens after a clone). No raw market price level is exported: only
risk measures, P&L, utilisation and scenario results of the **synthetic** book. Every table carries a ``data_label``
column marking the content as synthetic.

Dimensions: dim_date, dim_risk_factor, dim_method, dim_limit, dim_driver, dim_scenario, dim_counterparty.
Facts: fact_var, fact_backtest, fact_limit_utilisation, fact_risk_contribution, fact_stress, fact_margin_forecast,
fact_margin_stress, fact_attribution, fact_credit, fact_credit_profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config

LABEL = "synthetic book on real market data"
METHOD_NAMES = {"hs": "Historical simulation", "normal": "Parametric normal", "garch_t_gauss": "GARCH-t, Gaussian copula",
                "garch_t_tcop": "GARCH-t, Student-t copula", "fhs": "Filtered historical simulation", "ewma": "EWMA normal"}
FACTOR_ROWS = [("tge", "TGE price", "EUR/MWh (PLN via EUR/PLN)", "TGE via Instrat"),
               ("ttf", "TTF price", "EUR/MWh", "ICE Endex second month"),
               ("fx", "EUR/PLN", "PLN per EUR", "ECB reference rate")]
DRIVERS = {"plan_margin": ("Plan margin", "plan"), "volume_effect": ("Volume", "volume"),
           "price_effect_purchase": ("TGE purchase price", "price"), "price_effect_hedge": ("TTF hedge", "price"),
           "fx_effect_purchase": ("FX on purchases", "fx"), "fx_effect_hedge_translation": ("FX on hedge translation", "fx"),
           "fx_effect_forward": ("EUR forward", "fx")}


SHORT_SCENARIO_MAX = 21  # characters: fits the category axis of the dashboard's stress charts
SHORT_PATTERNS = [  # (regex on the full scenario name, short label template)
    (r"^worst (\d+)-day crisis move$", "Crisis worst {0}d"),
    (r"^worst (\d+)-day TTF fall in crisis$", "TTF fall worst {0}d"),
    (r"^basis widening: TGE \+(\d+) EUR/MWh, TTF unchanged$", "Basis TGE +{0} EUR/MWh"),
]
SHORT_NAMES = {
    "Russian invasion of Ukraine": "Ukraine invasion",
    "August 2022 price peak run-up": "Aug 2022 peak run-up",
    "TTF +50% only (TGE unchanged)": "TTF +50% only",
    "gas complex +50% (TGE and TTF)": "Gas +50%",
    "gas complex -50% (TGE and TTF)": "Gas -50%",
    "combined: gas complex +50% and EUR/PLN +5%": "Gas +50%, EUR/PLN +5%",
    "volatility stress: crisis 99th percentile volatility, price unchanged": "Vol stress p99",
}


def short_scenario(name: str) -> str:
    """Dashboard label for a scenario: a pattern, an explicit short name, the name itself if short, else truncated."""
    import re

    for pat, tpl in SHORT_PATTERNS:
        m = re.match(pat, name)
        if m:
            return tpl.format(*m.groups())
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    return name if len(name) <= SHORT_SCENARIO_MAX else name[: SHORT_SCENARIO_MAX - 1].rstrip() + "."


def _label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(data_label=LABEL)


def build_tables(cfg: Config) -> dict[str, pd.DataFrame]:
    out, clean = cfg.path("outputs_dir"), cfg.path("clean_dir")
    official = pd.read_csv(out / "stage7_official_method.csv")["method"].iloc[0]
    lim_cfg = cfg["limits"]

    # VaR facts (no level columns)
    fc = pd.read_csv(clean / "var_forecasts.csv", parse_dates=["origin", "target"])
    var = fc[["origin", "target", "method", "var_95", "var_99", "es_95", "es_97.5", "es_99", "pnl", "pnl_tge", "pnl_ttf", "pnl_fx"]].copy()
    var = var.rename(columns={"origin": "date", "method": "method_key", "es_97.5": "es_97_5", "pnl": "realised_pnl_next_day"})
    var["exception_95"] = (-var["realised_pnl_next_day"] > var["var_95"]).astype(int).where(var["realised_pnl_next_day"].notna())
    var["exception_99"] = (-var["realised_pnl_next_day"] > var["var_99"]).astype(int).where(var["realised_pnl_next_day"].notna())

    bt = pd.read_csv(out / "stage6_backtest.csv").rename(columns={"method": "method_key"})
    bt = bt[["method_key", "level", "n", "expected", "observed", "p_pof", "p_ind", "p_cc", "pass_pof", "pass_ind", "pass_cc"]]

    lu = pd.read_csv(clean / "limit_utilisation.csv", parse_dates=["origin"])
    amber = float(lim_cfg["amber_threshold"])
    pieces = []
    for key, meas, util in (("var_99", "var_99", "util_var_99"), ("open_gas", "open_gas_exposure_pln", "util_open_gas"),
                            ("fx_net_open", "fx_net_open_eur", "util_fx_net_open")):
        p = pd.DataFrame({"date": lu["origin"], "limit_key": key, "measure": lu[meas], "utilisation": lu[util]})
        pieces.append(p)
    util = pd.concat(pieces, ignore_index=True)
    util["status"] = np.select([util["utilisation"] > 1, util["utilisation"] >= amber], ["breach", "amber"], "green")
    util["breach"] = (util["status"] == "breach").astype(int)

    comp = lu[["origin", "comp_var99_tge_eur", "comp_var99_ttf", "comp_var99_fx"]].melt(id_vars="origin", var_name="factor_key", value_name="var99_contribution_pln")
    comp["factor_key"] = comp["factor_key"].map({"comp_var99_tge_eur": "tge", "comp_var99_ttf": "ttf", "comp_var99_fx": "fx"})
    comp = comp.rename(columns={"origin": "date"})

    st_rows = []
    for kind, f in (("historical", "stage8_historical.csv"), ("hypothetical", "stage8_hypothetical.csv")):
        s = pd.read_csv(out / f)
        for _, r in s.iterrows():
            for fk, col in (("tge", "pnl_tge_leg"), ("ttf", "pnl_ttf_leg"), ("fx", "pnl_fx_leg")):
                st_rows.append({"scenario": r["scenario"], "scenario_type": kind, "factor_key": fk, "pnl_pln": r[col],
                                "loss_over_var99": r["loss_over_var99"]})
    stress = pd.DataFrame(st_rows)
    scen = stress[["scenario", "scenario_type"]].drop_duplicates()
    ms = pd.read_csv(out / "stage12_stress.csv")
    scen = pd.concat([scen, ms.loc[~ms["scenario"].isin(scen["scenario"]), ["scenario", "source"]].rename(columns={"source": "scenario_type"})], ignore_index=True)
    scen.insert(0, "scenario_key", range(1, len(scen) + 1))
    scen.insert(2, "scenario_short", scen["scenario"].map(short_scenario))
    key = scen.set_index("scenario")["scenario_key"]
    stress["scenario_key"] = stress["scenario"].map(key)
    margin_stress = ms[["scenario", "ttf_move", "vm_eur", "im_after_eur", "margin_call_eur", "margin_call_pln", "credit_line_utilisation"]].copy()
    margin_stress["scenario_key"] = margin_stress["scenario"].map(key)

    as_of = pd.Timestamp(pd.read_csv(out / "stage12_summary.csv")["as_of"].iloc[0])
    mf = pd.read_csv(out / "stage12_forecast.csv")
    mf.insert(0, "date", pd.bdate_range(as_of + pd.offsets.BDay(1), periods=len(mf)))
    msum = pd.read_csv(out / "stage12_summary.csv").iloc[0]
    mf["buffer_10d_q99_eur"] = msum["buffer_10d_q99_eur"]
    mf["bank_credit_line_eur"] = msum["bank_credit_line_eur"]
    mf["im_today_eur"] = msum["im_today_fhs_eur"]

    att = pd.read_csv(clean / "attribution_daily.csv", parse_dates=["date"])
    attr = att[["date", *DRIVERS]].melt(id_vars="date", var_name="driver_key", value_name="pnl_pln")

    cl = pd.read_csv(out / "stage13_limits.csv")
    sc = pd.read_csv(out / "stage13_scorecard.csv")
    cps = sc[["counterparty", "type", "weighted_score", "grade", "grade_label", "pfe_limit_eur"]]
    prof = pd.read_csv(out / "stage13_exposure_profile.csv")[["counterparty", "month", "ee_eur", "ee_no_netting_eur", "pfe_95_eur", "pfe_97.5_eur"]]
    prof = prof.rename(columns={"pfe_97.5_eur": "pfe_97_5_eur"})
    credit = cl.rename(columns={"peak_pfe_97.5_eur": "peak_pfe_97_5_eur"}).drop(columns=["grade"])

    dates = pd.concat([var["date"], util["date"], attr["date"], mf["date"]]).drop_duplicates().sort_values()
    alld = pd.date_range(dates.min(), dates.max(), freq="D")
    dim_date = pd.DataFrame({"date": alld, "year": alld.year, "quarter": "Q" + alld.quarter.astype(str),
                             "month": alld.month, "month_name": alld.strftime("%b"), "year_month": alld.strftime("%Y-%m"),
                             "weekday": alld.strftime("%a"), "is_weekend": (alld.dayofweek >= 5).astype(int)})
    dim_method = pd.DataFrame([{"method_key": k, "method_name": v, "is_official": int(k == official)} for k, v in METHOD_NAMES.items()])
    dim_factor = pd.DataFrame(FACTOR_ROWS, columns=["factor_key", "factor_name", "unit", "source"])
    dim_limit = pd.DataFrame([
        {"limit_key": "var_99", "limit_name": "99% VaR (official)", "limit_short": "VaR 99%",
         "limit_value": float(lim_cfg["var99_limit_pln"]), "unit": "PLN"},
        {"limit_key": "open_gas", "limit_name": "Open gas exposure", "limit_short": "Gas exposure",
         "limit_value": float(lim_cfg["open_gas_exposure_limit_pln"]), "unit": "PLN"},
        {"limit_key": "fx_net_open", "limit_name": "FX net open position", "limit_short": "FX open position",
         "limit_value": float(lim_cfg["fx_net_open_limit_eur"]), "unit": "EUR"},
    ]).assign(amber_threshold=amber)
    util = util.merge(dim_limit[["limit_key", "limit_value"]], on="limit_key")
    dim_driver = pd.DataFrame([{"driver_key": k, "driver_name": v[0], "driver_group": v[1]} for k, v in DRIVERS.items()])

    tables = {"dim_date": dim_date, "dim_method": dim_method, "dim_risk_factor": dim_factor, "dim_limit": dim_limit,
              "dim_driver": dim_driver, "dim_scenario": scen, "dim_counterparty": cps,
              "fact_var": var, "fact_backtest": bt, "fact_limit_utilisation": util, "fact_risk_contribution": comp,
              "fact_stress": stress.drop(columns=["scenario", "scenario_type"]), "fact_margin_forecast": mf,
              "fact_margin_stress": margin_stress.drop(columns=["scenario"]), "fact_attribution": attr,
              "fact_credit": credit, "fact_credit_profile": prof}
    for k, t in tables.items():
        for c in t.columns:
            if pd.api.types.is_datetime64_any_dtype(t[c]):
                t[c] = t[c].dt.strftime("%Y-%m-%d")
        tables[k] = _label(t)
    return tables


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    tables = build_tables(cfg)
    d = cfg.root / "exports"
    d.mkdir(exist_ok=True)
    for k, t in tables.items():
        t.to_csv(d / f"{k}.csv", index=False)
    return tables


def main() -> None:
    for k, t in run(load_config()).items():
        print(f"{k}: {len(t)} rows, {len(t.columns)} columns")


if __name__ == "__main__":
    main()
