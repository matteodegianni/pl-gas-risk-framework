"""Stage 12: variation and initial margin forecasting for the SYNTHETIC TTF futures hedge.

Usage::

    python -m gasrisk.margin

Position: long N = ttf_hedge_ratio x q_open_mwh MWh of TTF second-month futures (EUR), as in the Stage 5 book.

* Variation margin (VM): daily mark-to-market N x dTTF (roll-adjusted), EUR; an outflow when TTF falls.
* Initial margin (IM): 2-day 99% VaR of the position by filtered historical simulation: two-day GARCH(1,1) paths
  driven by bootstrapped standardised residuals of the Stage 4 TTF model, refitted on the last 750 returns. The
  historical IM series uses the Stage 4 one-step conditional volatility and the empirical residual quantile
  (2-day variance approximated by 2 x the one-step variance; full-sample parameters, stated as a limitation).
* Base forecast: 20,000 FHS-GARCH paths over 10 days; each day VM plus the change in IM (IM recomputed from the
  path's price and conditional variance) gives the margin call.
* Stress: Stage 8 scenarios plus the worst historical TTF falls of the crisis window, VM at today's position and IM
  recomputed after the scenario (volatility at the end of a historical episode; today's volatility for a hypothetical
  shock), plus a volatility-only stress at the crisis-window 99th percentile volatility.
* Liquidity buffer: 99% quantile of the 10-day cumulative margin outflow, in EUR and in PLN (today's EUR/PLN and
  EUR/PLN +5%), against a synthetic bank credit line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config
from gasrisk.ttf_sim import fit_ttf_garch, simulate_returns


def im_from_vol(n_mwh: float, price, sigma1, q_z: float, mpor: int):
    """IM of a long futures position: N x F x (1 - exp(sqrt(mpor) x sigma_t+1 x q_z)), q_z < 0 the 1% residual quantile."""
    return n_mwh * np.asarray(price) * (1.0 - np.exp(np.sqrt(mpor) * np.asarray(sigma1) * q_z))


def im_fhs(m, n_mwh: float, price: float, mpor: int, conf: float, n: int, rng: np.random.Generator) -> float:
    """IM today: 2-day 99% loss quantile of the long position from FHS-GARCH paths."""
    r, _ = simulate_returns(m, n, mpor, rng, "fhs")
    loss = -n_mwh * price * np.expm1(r.sum(axis=1))
    return float(np.quantile(loss, conf))


def historical_margin(panel: pd.DataFrame, cvol: pd.Series, z: np.ndarray, n_mwh: float, mcfg: dict) -> pd.DataFrame:
    """Daily VM and approximate IM for a constant position at historical prices (EUR)."""
    ttf = panel["ttf_c2"]
    d_ttf = (ttf.shift(1) * np.expm1(panel["ttf_risk_ret"].fillna(0.0)))
    q_z = float(np.quantile(z, 1 - float(mcfg["im_confidence"])))
    sigma_next = (cvol / 100.0).shift(-1)  # sigma_t+1 is known at t
    im = im_from_vol(n_mwh, ttf, sigma_next, q_z, int(mcfg["im_mpor_days"]))
    df = pd.DataFrame({"vm_eur": n_mwh * d_ttf, "im_eur": im}, index=panel.index)
    df["margin_call_eur"] = -df["vm_eur"] + df["im_eur"].diff()   # positive = cash paid to the clearing house
    return df.dropna()


def simulate_margin(m, n_mwh: float, price: float, mcfg: dict, q_z: float, rng: np.random.Generator) -> dict:
    """Base-case daily margin calls over the buffer horizon (EUR, positive = outflow)."""
    H, n, mpor = int(mcfg["buffer_horizon_days"]), int(mcfg["n_paths"]), int(mcfg["im_mpor_days"])
    r, s2 = simulate_returns(m, n, H, rng, "fhs")
    p = price * np.exp(np.cumsum(r, axis=1))
    p_prev = np.column_stack([np.full(n, price), p[:, :-1]])
    vm = n_mwh * (p - p_prev)
    s2_next = m.omega + m.alpha * (r + 0.5 * s2) ** 2 + m.beta * s2   # next-day conditional variance (shock = r + s2/2)
    im = im_from_vol(n_mwh, p, np.sqrt(s2_next), q_z, mpor)
    im0 = float(im_from_vol(n_mwh, price, np.sqrt(m.sigma2_next), q_z, mpor))
    d_im = np.diff(np.column_stack([np.full(n, im0), im]), axis=1)
    call = -vm + d_im
    return {"call": call, "cum": np.cumsum(call, axis=1), "im0": im0, "im": im, "vm": vm}


def forecast_table(sim: dict, fx: float) -> pd.DataFrame:
    rows = []
    for d in range(sim["cum"].shape[1]):
        c = sim["cum"][:, d]
        rows.append({"day": d + 1, "mean_cum_outflow_eur": c.mean(), "q95_cum_outflow_eur": np.quantile(c, 0.95),
                     "q99_cum_outflow_eur": np.quantile(c, 0.99), "q99_cum_outflow_pln": np.quantile(c, 0.99) * fx,
                     "mean_daily_call_eur": sim["call"][:, d].mean(), "q99_daily_call_eur": np.quantile(sim["call"][:, d], 0.99),
                     "mean_im_eur": sim["im"][:, d].mean()})
    return pd.DataFrame(rows)


def stress_margin(cfg: Config, panel: pd.DataFrame, cvol: pd.Series, n_mwh: float, price: float, fx: float, im0: float,
                  sigma_today: float, sigma_stress: float, q_z: float) -> pd.DataFrame:
    """Margin call per scenario: -VM + (IM after the scenario - IM today), positive = cash out.

    IM after a historical scenario uses the conditional volatility observed at the end of the episode (Stage 4);
    after a hypothetical price shock it uses today's volatility; the volatility stress row uses the crisis-window
    stress volatility at today's price.
    """
    out = cfg.path("outputs_dir")
    mpor = int(cfg["margin"]["im_mpor_days"])
    sig = (cvol / 100.0)
    rows = []
    hist = pd.read_csv(out / "stage8_historical.csv")
    for _, s in hist.iterrows():
        rows.append({"scenario": s["scenario"], "source": "Stage 8 historical", "from": s["from"], "to": s["to"],
                     "ttf_move": s["move_ttf"], "fx_move": s["move_fx"], "sigma_after": float(sig.loc[:s["to"]].iloc[-1])})
    for s in cfg["stress"]["hypothetical"]:
        rows.append({"scenario": s["name"], "source": "Stage 8 hypothetical", "ttf_move": s["ttf"], "fx_move": s["fx"],
                     "sigma_after": sigma_today})
    c0, c1 = cfg["margin"]["crisis_window"]
    r = panel["ttf_risk_ret"].loc[c0:c1].fillna(0.0)
    idx = r.index
    for k in (1, 5, 10, 21):
        cum = r.rolling(k).sum().dropna()
        end = cum.idxmin()
        fxr = np.log(panel["eurpln"]).diff().loc[c0:c1].rolling(k).sum().loc[end]
        rows.append({"scenario": f"worst {k}-day TTF fall in crisis", "source": "margin-specific historical",
                     "from": str(idx[idx.get_loc(end) - k + 1].date()), "to": str(end.date()),
                     "ttf_move": float(np.expm1(cum.min())), "fx_move": float(np.expm1(fxr)),
                     "sigma_after": float(sig.loc[:end].iloc[-1])})
    rows.append({"scenario": "volatility stress: crisis 99th percentile volatility, price unchanged",
                 "source": "margin-specific", "ttf_move": 0.0, "fx_move": 0.0, "sigma_after": sigma_stress})
    df = pd.DataFrame(rows)
    p1 = price * (1 + df["ttf_move"])
    df["vm_eur"] = n_mwh * (p1 - price)
    df["im_after_eur"] = im_from_vol(n_mwh, p1, df["sigma_after"], q_z, mpor)
    df["margin_call_eur"] = -df["vm_eur"] + df["im_after_eur"] - im0
    df["margin_call_pln"] = df["margin_call_eur"] * fx * (1 + df["fx_move"])
    line = float(cfg["margin"]["bank_credit_line_eur"])
    df["credit_line_utilisation"] = df["margin_call_eur"].clip(lower=0) / line
    return df


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    mcfg = dict(cfg["margin"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    as_of = panel.index[-1]
    n_mwh = float(cfg["book"]["ttf_hedge_ratio"]) * float(cfg["book"]["q_open_mwh"])
    price, fx = float(panel["ttf_c2"].iloc[-1]), float(panel["eurpln"].iloc[-1])
    m = fit_ttf_garch(cfg, panel["ttf_risk_ret"], as_of)
    rng = np.random.default_rng(cfg.seed)
    q_z = float(np.quantile(m.std_resid, 1 - float(mcfg["im_confidence"])))
    im_today = im_fhs(m, n_mwh, price, int(mcfg["im_mpor_days"]), float(mcfg["im_confidence"]), int(mcfg["im_sims"]), rng)
    im_formula = float(im_from_vol(n_mwh, price, np.sqrt(m.sigma2_next), q_z, int(mcfg["im_mpor_days"])))

    cvol = pd.read_csv(cfg.path("clean_dir") / "garch_conditional_vol.csv", index_col="date", parse_dates=["date"])["ttf_t"]
    z4 = pd.read_csv(cfg.path("clean_dir") / "garch_std_resid.csv", index_col="date", parse_dates=["date"])["ttf_t"].dropna().to_numpy()
    hist = historical_margin(panel, cvol.reindex(panel.index), z4, n_mwh, mcfg)
    H = int(mcfg["buffer_horizon_days"])
    hist_cum = hist["margin_call_eur"].rolling(H).sum()
    c0, c1 = mcfg["crisis_window"]
    sigma_stress = float(cvol.loc[c0:c1].quantile(0.99) / 100.0)

    sim = simulate_margin(m, n_mwh, price, mcfg, q_z, rng)
    fc = forecast_table(sim, fx)
    cum_H = sim["cum"][:, -1]
    peak_H = sim["cum"].max(axis=1)
    conf = float(mcfg["buffer_confidence"])
    buffer_eur = float(np.quantile(cum_H, conf))
    line = float(mcfg["bank_credit_line_eur"])
    # historical check of the buffer: 10-day returns of the full sample applied to today's position (VM part only)
    r10 = panel["ttf_risk_ret"].fillna(0.0).rolling(H).sum().dropna()
    vm10_today = -n_mwh * price * np.expm1(r10)
    stress = stress_margin(cfg, panel, cvol, n_mwh, price, fx, im_formula, float(np.sqrt(m.sigma2_next)), sigma_stress, q_z)

    summary = pd.DataFrame([{
        "as_of": as_of.date(), "position_mwh": n_mwh, "notional_eur": n_mwh * price,
        "im_today_fhs_eur": im_today, "im_today_fhs_pln": im_today * fx, "im_today_pct_notional": im_today / (n_mwh * price),
        "im_today_formula_eur": im_formula, "garch_sigma_next_daily": np.sqrt(m.sigma2_next),
        "residual_q01": q_z, "crisis_q99_sigma_daily": sigma_stress,
        "buffer_10d_q99_eur": buffer_eur, "buffer_10d_q99_pln": buffer_eur * fx,
        "buffer_10d_q99_pln_fx_stress": buffer_eur * fx * (1 + float(mcfg["fx_stress"])),
        "peak_cum_outflow_10d_q99_eur": float(np.quantile(peak_H, conf)),
        "mean_cum_outflow_10d_eur": float(cum_H.mean()),
        "bank_credit_line_eur": line, "buffer_over_credit_line": buffer_eur / line,
        "hist_share_10d_vm_outflow_above_buffer": float((vm10_today > buffer_eur).mean()),
        "hist_windows_10d": len(vm10_today),
    }])
    hist_summary = pd.DataFrame([{
        "sample": name, "days": len(h), "max_daily_call_eur": h["margin_call_eur"].max(),
        "q99_daily_call_eur": h["margin_call_eur"].quantile(0.99),
        "max_10d_cum_call_eur": hist_cum.loc[h.index].max(), "max_im_eur": h["im_eur"].max(),
        "date_max_im": h["im_eur"].idxmax().date(), "mean_im_eur": h["im_eur"].mean()}
        for name, h in (("full sample", hist), ("crisis window", hist.loc[c0:c1]), ("last 250 days", hist.iloc[-250:]))])
    tables = {"summary": summary, "forecast": fc, "stress": stress, "historical_summary": hist_summary}
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage12_{k}.csv", index=False)
    hist.to_csv(cfg.path("clean_dir") / "margin_daily.csv")
    from gasrisk.margin_report import write_stage12_report

    write_stage12_report(cfg, tables)
    return tables


def main() -> None:
    t = run(load_config())
    print(t["summary"].T.to_string())
    print(t["stress"].to_string(index=False))
    print(t["historical_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
