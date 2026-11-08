"""Stage 9: daily P&L attribution of a SYNTHETIC one-year supply contract.

Usage::

    python -m gasrisk.attribution

Contract (synthetic): signed one year before the last panel date. Contracted daily volumes C_d follow a winter
peak profile summing to the open volume of the book; actual volumes V_d = C_d (1 + e_d) with a synthetic AR(1)
demand deviation. The supplier sells at a fixed PLN price S = P0 x F0 + margin (P0 = TGE in EUR, F0 = EUR/PLN at
signing), buys V_d at TGE (G_d x F_d, trade-date price) and hedges with TTF futures and an EUR forward sized on
the remaining contracted volume.

Daily P&L:  V_d (S - G_d F_d) + N_ttf (TTF_d - TTF_d-1) F_d + N_fx (F_d - F_d-1)

Sequential attribution (exact, no residual):

* plan margin      C_d (S - P0 F0)
* volume effect    (V_d - C_d)(S - P0 F0)
* price effect     -V_d (G_d - P0) F0 + N_ttf dTTF F0                 (gas prices at plan FX)
* FX effect        -V_d G_d (F_d - F0) + N_ttf dTTF (F_d - F0) + N_fx dF
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config

COMPONENTS = ["plan_margin", "volume_effect", "price_effect", "fx_effect"]


def demand_profile(index: pd.DatetimeIndex, total: float, amplitude: float, phi: float, sd: float, seed: int) -> pd.DataFrame:
    """SYNTHETIC contracted and actual volumes per day (MWh)."""
    doy = index.day_of_year.to_numpy()
    shape = 1 + amplitude * np.cos(2 * np.pi * (doy - 15) / 365.25)
    contracted = total * shape / shape.sum()
    rng = np.random.default_rng(seed)
    e = np.zeros(len(index))
    innov = rng.normal(0, sd * np.sqrt(1 - phi**2), len(index))
    for t in range(len(index)):
        e[t] = (phi * e[t - 1] if t else 0.0) + innov[t]
    return pd.DataFrame({"contracted_mwh": contracted, "actual_mwh": contracted * (1 + e)}, index=index)


def attribute(df: pd.DataFrame, S: float, P0: float, F0: float) -> pd.DataFrame:
    """Daily P&L and its exact decomposition.

    ``df`` columns: contracted_mwh, actual_mwh, tge_eur (G), fx (F), d_ttf (TTF change in EUR/MWh),
    n_ttf (MWh held over the day), n_fx (EUR held over the day), d_fx (EUR/PLN change).
    """
    C, V, G, F = df["contracted_mwh"], df["actual_mwh"], df["tge_eur"], df["fx"]
    hedge_ttf = df["n_ttf"] * df["d_ttf"] * F
    hedge_fx = df["n_fx"] * df["d_fx"]
    out = pd.DataFrame(index=df.index)
    out["total_pnl"] = V * (S - G * F) + hedge_ttf + hedge_fx
    out["plan_margin"] = C * (S - P0 * F0)
    out["volume_effect"] = (V - C) * (S - P0 * F0)
    out["price_effect_purchase"] = -V * (G - P0) * F0
    out["price_effect_hedge"] = df["n_ttf"] * df["d_ttf"] * F0
    out["fx_effect_purchase"] = -V * G * (F - F0)
    out["fx_effect_hedge_translation"] = df["n_ttf"] * df["d_ttf"] * (F - F0)
    out["fx_effect_forward"] = hedge_fx
    out["price_effect"] = out["price_effect_purchase"] + out["price_effect_hedge"]
    out["fx_effect"] = out["fx_effect_purchase"] + out["fx_effect_hedge_translation"] + out["fx_effect_forward"]
    out["residual"] = out["total_pnl"] - out[COMPONENTS].sum(axis=1)
    return out


def build_inputs(panel: pd.DataFrame, a: dict, total_mwh: float, seed: int) -> tuple[pd.DataFrame, float, float, float]:
    end = panel.index[-1]
    start = panel.index[panel.index >= end - pd.Timedelta(days=int(a["contract_days"]))][0]
    p = panel.loc[start:end]
    P0, F0 = float(p["tge_eur"].iloc[0]), float(p["eurpln"].iloc[0])
    S = P0 * F0 + float(a["sale_margin_pln_mwh"])
    days = p.index[1:]  # deliveries from the day after signing
    dem = demand_profile(days, total_mwh, float(a["seasonal_amplitude"]), float(a["demand_ar1"]), float(a["demand_sd"]), seed)
    remaining_before = total_mwh - dem["contracted_mwh"].cumsum().shift(1, fill_value=0.0)
    r_ttf = p["ttf_risk_ret"].fillna(0.0)  # the two excluded roll days count as no move
    ttf = p["ttf_c2"]
    df = dem.assign(tge_eur=p["tge_eur"].loc[days], fx=p["eurpln"].loc[days],
                    d_ttf=(ttf.shift(1) * np.expm1(r_ttf)).loc[days], d_fx=p["eurpln"].diff().loc[days],
                    n_ttf=float(a["ttf_hedge_ratio"]) * remaining_before,
                    n_fx=float(a["fx_hedge_ratio"]) * remaining_before * P0)
    return df, S, P0, F0


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    a = dict(cfg["attribution"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    df, S, P0, F0 = build_inputs(panel, a, float(cfg["book"]["q_open_mwh"]), cfg.seed)
    att = attribute(df, S, P0, F0)
    max_res = float(att["residual"].abs().max())
    if max_res > 1e-6:
        raise AssertionError(f"attribution does not reconcile: max residual {max_res}")
    att.join(df).to_csv(cfg.path("clean_dir") / "attribution_daily.csv")
    detail = ["price_effect_purchase", "price_effect_hedge", "fx_effect_purchase", "fx_effect_hedge_translation", "fx_effect_forward"]
    total = att[["total_pnl", *COMPONENTS, *detail]].sum()
    monthly = att.groupby(att.index.to_period("M"))[["total_pnl", *COMPONENTS]].sum().reset_index(names="month")
    monthly["month"] = monthly["month"].astype(str)
    monthly["reconciles"] = (monthly["total_pnl"] - monthly[COMPONENTS].sum(axis=1)).abs() < 1e-3
    vol = pd.DataFrame([{"contract_start": df.index[0].date(), "contract_end": df.index[-1].date(), "delivery_days": len(df),
                         "contracted_mwh": df["contracted_mwh"].sum(), "actual_mwh": df["actual_mwh"].sum(),
                         "volume_deviation": df["actual_mwh"].sum() / df["contracted_mwh"].sum() - 1,
                         "sale_margin_pln_mwh": a["sale_margin_pln_mwh"], "max_abs_daily_residual": max_res}])
    tables = {"total": total.rename("pln").rename_axis("component").reset_index(), "monthly": monthly, "contract": vol}
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage9_{k}.csv", index=False)
    from gasrisk.attribution_report import write_stage9_report

    write_stage9_report(cfg, tables, att)
    return tables


def main() -> None:
    t = run(load_config())
    print(t["contract"].to_string(index=False))
    print(t["total"].to_string(index=False))


if __name__ == "__main__":
    main()
