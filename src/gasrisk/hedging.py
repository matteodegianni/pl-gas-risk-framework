"""Stage 11: minimum-variance hedge ratio, strategy comparison and hedge effectiveness for the SYNTHETIC book.

Usage::

    python -m gasrisk.hedging [--offline]

Hedged item: floating TGE purchases in PLN per MWh (S = TGE_PLN). Hedge: TTF second-month futures in EUR, whose
P&L per MWh in PLN is dF = dTTF x FX_t (roll-adjusted TTF change), optionally with an EUR/PLN forward.

* Minimum-variance hedge ratio h* = Cov(dS, dF) / Var(dF) on daily changes (rolling 250 days) and on non-overlapping
  21-day changes (full sample and rolling 36 months); a two-instrument version adds the EUR forward.
* Strategy comparison on today's book over 21-day horizons (overlapping historical windows applied to today's levels,
  full revaluation; collar repriced with Black-76 at the Stage 10 volatility assumption): no hedge, futures at h*,
  futures at 0.70, TTF collar (long call, short put) on 0.70 of the volume; each without and with the EUR forward.
* Effectiveness (simplified, **not IFRS 9 hedge accounting**): dollar-offset per 21-day period (80-125%) and regression
  (R2 >= 0.80, slope 0.80-1.25), with a 1:1 designated volume.
* Ineffectiveness decomposition (exact): dS - dF = dB_adj x FX_t + TGE_EUR_t-1 x dFX, where dB_adj = dTGE_EUR - dTTF is
  the change in the TGE-TTF location plus tenor basis (roll-adjusted). The covariance share of each term in the
  variance of the ineffectiveness says how much of it the basis explains.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from gasrisk.config import Config, load_config
from gasrisk.pricing import black76, market_inputs
from gasrisk.var import var_es


def hedge_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily changes per MWh: dS (TGE PLN), dF (TTF hedge P&L in PLN), basis and FX terms of the difference."""
    r_ttf = panel["ttf_risk_ret"]
    d_ttf = panel["ttf_c2"].shift(1) * np.expm1(r_ttf)
    fx, g = panel["eurpln"], panel["tge_eur"]
    df = pd.DataFrame({"dS": panel["tge_pln"].diff(), "dF": d_ttf * fx, "d_ttf": d_ttf, "dFX": fx.diff(),
                       "dG": g.diff(), "fx": fx, "g_prev": g.shift(1)}, index=panel.index)
    df["dFXfwd"] = df["g_prev"] * df["dFX"]           # P&L per MWh of an EUR forward sized at TGE_EUR_t-1 EUR
    df["basis_term"] = (df["dG"] - df["d_ttf"]) * df["fx"]
    df["fx_term"] = df["dFXfwd"]
    return df.dropna()


def mv_ratio(dS: np.ndarray, dF: np.ndarray) -> float:
    return float(np.cov(dS, dF)[0, 1] / np.var(dF, ddof=1))


def blocks(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sums over non-overlapping ``n``-day blocks ending on the last observation."""
    k = np.arange(len(df))[::-1] // n
    g = df.assign(block=k).groupby("block")
    out = g[["dS", "dF", "dFXfwd", "basis_term", "fx_term"]].sum()
    out["end"] = g.apply(lambda x: x.index[-1], include_groups=False)
    out["n"] = g.size()
    return out[out["n"] == n].set_index("end").sort_index()


def ratio_table(d: pd.DataFrame, m: pd.DataFrame, hcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample, sub-period and latest rolling hedge ratios; rolling daily series."""
    w, wm = int(hcfg["rolling_window_days"]), int(hcfg["rolling_window_months"])
    roll = (d["dS"].rolling(w).cov(d["dF"]) / d["dF"].rolling(w).var()).dropna().rename("h_daily_rolling")
    roll_m = (m["dS"].rolling(wm).cov(m["dF"]) / m["dF"].rolling(wm).var()).dropna()
    rows = []

    def add(name, freq, x):
        X = sm.add_constant(x[["dF"]])
        fit = sm.OLS(x["dS"], X).fit()
        X2 = sm.add_constant(x[["dF", "dFXfwd"]])
        fit2 = sm.OLS(x["dS"], X2).fit()
        rows.append({"sample": name, "frequency": freq, "n": len(x), "h_star": mv_ratio(x["dS"], x["dF"]),
                     "r2": fit.rsquared, "h_ttf_with_fx": fit2.params["dF"], "h_fx_with_fx": fit2.params["dFXfwd"],
                     "r2_with_fx": fit2.rsquared})

    add("full", "daily", d)
    add("full", "21-day", m)
    c0, c1 = hcfg["crisis_window"]
    add("pre-crisis", "21-day", m.loc[:c0])
    add("crisis", "21-day", m.loc[c0:c1])
    add("post-crisis", "21-day", m.loc[c1:])
    rows.append({"sample": f"latest rolling {w} days", "frequency": "daily", "n": w, "h_star": float(roll.iloc[-1])})
    rows.append({"sample": f"latest rolling {wm} blocks", "frequency": "21-day", "n": wm, "h_star": float(roll_m.iloc[-1])})
    summary = pd.DataFrame(rows)
    rs = roll.describe(percentiles=[0.05, 0.5, 0.95]).rename("h_daily_rolling").to_frame().T
    rs = pd.concat([rs, roll_m.describe(percentiles=[0.05, 0.5, 0.95]).rename("h_21day_rolling").to_frame().T])
    return summary, rs.reset_index(names="series")


def strategy_pnl(lv: dict, R: np.ndarray, q: float, n_fut: float, n_collar: float, fx_fwd: bool, mi: dict,
                 hcfg: dict, horizon: int) -> np.ndarray:
    """21-day P&L (PLN) of today's book for joint cumulative log returns R (n x 3: tge_eur, ttf, fx)."""
    g, t, f = lv["tge_eur"], lv["ttf"], lv["fx"]
    g1, t1, f1 = g * np.exp(R[:, 0]), t * np.exp(R[:, 1]), f * np.exp(R[:, 2])
    pnl = -q * (g1 * f1 - g * f) + n_fut * (t1 - t) * f1
    if n_collar:
        T0, s = mi["T"], mi["sigma"]
        T1 = max(T0 - horizon * 7 / 5 / 365.0, 1e-6)  # calendar time over ``horizon`` trading days
        kc, kp = hcfg["collar_call_moneyness"] * t, hcfg["collar_put_moneyness"] * t
        v0 = black76(t, kc, T0, s, 0.0, "call", False) - black76(t, kp, T0, s, 0.0, "put", False)
        v1 = black76(t1, kc, T1, s, 0.0, "call", False) - black76(t1, kp, T1, s, 0.0, "put", False)
        pnl = pnl + n_collar * (v1 - v0) * f1
    if fx_fwd:
        pnl = pnl + float(hcfg["fx_ratio"]) * q * g * (f1 - f)
    return pnl


def compare_strategies(panel: pd.DataFrame, h_star: float, cfg: Config, mi: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    hcfg = {**cfg["hedging"], "fx_ratio": cfg["book"]["fx_hedge_ratio"]}
    H = int(hcfg["horizon_days"])
    q = float(cfg["book"]["q_open_mwh"])
    rets = pd.DataFrame({"tge_eur": np.log(panel["tge_eur"]).diff(), "ttf": panel["ttf_risk_ret"],
                         "fx": panel["eurpln_ret"]}).fillna(0.0).iloc[1:]
    cum = rets.rolling(H).sum().dropna()
    lv = {"tge_eur": float(panel["tge_eur"].iloc[-1]), "ttf": float(panel["ttf_c2"].iloc[-1]), "fx": float(panel["eurpln"].iloc[-1])}
    fr = float(hcfg["fixed_ratio"])
    strategies = {"no hedge": (0.0, 0.0), f"futures h*={h_star:.3f}": (h_star * q, 0.0),
                  f"futures {fr:.2f}": (fr * q, 0.0), f"collar {fr:.2f}": (0.0, fr * q)}
    c0, c1 = hcfg["crisis_window"]
    rows = []
    for fx_fwd in (False, True):
        for name, (nf, nc) in strategies.items():
            for sample, R in (("all windows", cum), ("crisis windows", cum.loc[c0:c1])):
                pnl = strategy_pnl(lv, R.to_numpy(), q, nf, nc, fx_fwd, mi, hcfg, H)
                m = var_es(pnl, [0.99], [0.975])
                rows.append({"strategy": name, "eur_forward": fx_fwd, "sample": sample, "windows": len(pnl),
                             "mean_pnl": pnl.mean(), "std_pnl": pnl.std(ddof=1), "var_99": m["var_99"], "es_97.5": m["es_97.5"]})
    res = pd.DataFrame(rows)
    base = res[(res["strategy"] == "no hedge") & (~res["eur_forward"])].set_index("sample")
    res["variance_reduction"] = 1 - res["std_pnl"] ** 2 / res["sample"].map(base["std_pnl"]) ** 2
    res["var99_reduction"] = 1 - res["var_99"] / res["sample"].map(base["var_99"])
    res["es97_5_reduction"] = 1 - res["es_97.5"] / res["sample"].map(base["es_97.5"])
    t, s, T = lv["ttf"], mi["sigma"], mi["T"]
    call = float(black76(t, hcfg["collar_call_moneyness"] * t, T, s, 0.0, "call", False))
    put = float(black76(t, hcfg["collar_put_moneyness"] * t, T, s, 0.0, "put", False))
    cost = pd.DataFrame([
        {"strategy": "futures", "premium_pct_F": 0.0, "premium_eur": 0.0, "premium_pln": 0.0,
         "note": "no premium; margin funding in Stage 12"},
        {"strategy": f"collar {fr:.2f}", "call_pct_F": call / t, "put_pct_F": put / t, "premium_pct_F": (call - put) / t,
         "premium_eur": (call - put) * fr * q, "premium_pln": (call - put) * fr * q * lv["fx"],
         "note": f"long {hcfg['collar_call_moneyness']:.2f}F call, short {hcfg['collar_put_moneyness']:.2f}F put, "
                 f"{mi['delivery_month']} options, futures-style value at the volatility assumption"},
    ])
    return res, cost


def effectiveness(m: pd.DataFrame, hcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dollar-offset and regression tests per 21-day block, and the exact ineffectiveness decomposition."""
    lo, hi = hcfg["dollar_offset_band"]
    rows, per = [], []
    for label, hedge in (("TTF futures only", m["dF"]), ("TTF futures and EUR forward", m["dF"] + m["dFXfwd"])):
        ratio = hedge / m["dS"]  # -(hedge change) / (hedged item change), hedged item = short TGE purchase cost
        ok = ratio.between(lo, hi)
        small = m["dS"].abs() < m["dS"].abs().median()
        X = sm.add_constant(hedge.rename("hedge"))
        fit = sm.OLS(m["dS"], X).fit()
        slope = float(fit.params["hedge"])
        k = int(hcfg["cumulative_periods"])
        cum = hedge.rolling(k).sum() / m["dS"].rolling(k).sum()
        cum = cum.dropna()
        rows.append({"hedge": label, "periods": len(m), "dollar_offset_pass_rate": ok.mean(),
                     "fails_with_small_dS_share": float((~ok & small).sum() / max((~ok).sum(), 1)),
                     f"rolling_{k}_period_cumulative_pass_rate": float(cum.between(lo, hi).mean()),
                     f"rolling_{k}_period_cumulative_median": float(cum.median()),
                     "regression_slope": slope, "regression_r2": fit.rsquared,
                     "regression_effective": fit.rsquared >= hcfg["regression_r2_min"]
                     and hcfg["regression_slope_band"][0] <= slope <= hcfg["regression_slope_band"][1]})
        per.append(pd.DataFrame({"hedge": label, "end": m.index, "dS": m["dS"].values, "hedge_change": hedge.values,
                                 "dollar_offset": ratio.values, "passes": ok.values}))
    # exact decomposition of the futures-only ineffectiveness
    ineff = m["dS"] - m["dF"]
    assert np.allclose(ineff, m["basis_term"] + m["fx_term"], atol=1e-6)
    v = ineff.var(ddof=1)
    dec = []
    for name, col in (("basis term dB_adj x FX", "basis_term"), ("FX term TGE_EUR x dFX", "fx_term")):
        dec.append({"component": name, "std_per_mwh_pln": m[col].std(ddof=1),
                    "covariance_share_of_ineffectiveness_variance": float(np.cov(m[col], ineff)[0, 1] / v)})
    dec.append({"component": "total ineffectiveness dS - dF", "std_per_mwh_pln": float(np.sqrt(v)),
                "covariance_share_of_ineffectiveness_variance": 1.0})
    dec = pd.DataFrame(dec)
    fail = ~(m["dF"] / m["dS"]).between(lo, hi)
    dec_fail = pd.DataFrame([{"subset": s, "periods": int(k.sum()),
                              "mean_abs_basis_term": m.loc[k, "basis_term"].abs().mean(),
                              "mean_abs_fx_term": m.loc[k, "fx_term"].abs().mean(),
                              "basis_share_of_abs_ineffectiveness": float(m.loc[k, "basis_term"].abs().sum()
                                                                          / (m.loc[k, "basis_term"].abs() + m.loc[k, "fx_term"].abs()).sum())}
                             for s, k in (("failing periods", fail), ("passing periods", ~fail))])
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True), pd.concat([dec, dec_fail], ignore_index=True)


def run(cfg: Config, offline: bool = False) -> dict[str, pd.DataFrame]:
    hcfg = dict(cfg["hedging"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    d = hedge_series(panel)
    m = blocks(d, int(hcfg["month_days"]))
    ratios, rolling = ratio_table(d, m, hcfg)
    h_star = float(ratios.loc[ratios["sample"].str.startswith("latest rolling") & (ratios["frequency"] == "21-day"), "h_star"].iloc[0])
    mi = market_inputs(cfg, offline=True) if offline else market_inputs(cfg)
    strat, cost = compare_strategies(panel, h_star, cfg, mi)
    eff, per, dec = effectiveness(m, hcfg)
    tables = {"ratios": ratios, "rolling_summary": rolling, "strategies": strat, "cost": cost,
              "effectiveness": eff, "dollar_offset_periods": per.assign(end=per["end"].dt.date), "ineffectiveness": dec}
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage11_{k}.csv", index=False)
    from gasrisk.hedging_report import write_stage11_report

    write_stage11_report(cfg, tables, h_star)
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    t = run(load_config(), ap.parse_args().offline)
    for k in ("ratios", "strategies", "cost", "effectiveness", "ineffectiveness"):
        print(t[k].to_string(index=False))


if __name__ == "__main__":
    main()
