"""Stage 13: counterparty credit exposure (CE, EE, PFE) of OTC TTF forwards with SYNTHETIC counterparties.

Usage::

    python -m gasrisk.credit

* Trades: monthly delivery strips of TTF forwards per counterparty (synthetic), strikes set relative to today's TTF.
* Price model: Monte Carlo of the TTF second-month price with the refitted Stage 4 GARCH(1,1)-t (Student-t shocks,
  driftless), simulated daily and read on a monthly grid. Single factor: every delivery month moves with the
  simulated price (flat curve, parallel moves), because no TTF forward curve is available.
* Value of the forward for delivery month m at time t (supplier buys): Q_m (F_t - K) DF(t, T_m); a sale has the opposite
  sign; a delivery month leaves the netting set once it starts. Discounting on the ECB AAA curve (Stage 10).
* Exposure per counterparty with netting: E(t) = max(sum of trade values, 0); without netting: sum of max(V_i, 0).
  CE = E(0); EE(t) = mean E(t); PFE_q(t) = q-quantile of E(t); EPE = average EE over the first 12 months.
* Scorecard: five synthetic financial ratios scored 1-5 by thresholds, weighted, mapped to an internal grade; the grade
  sets the limit on the peak 95% PFE; utilisation and breach status as in Stage 7 (amber from 80%).
* **Wrong-way risk is not modelled** (exposure and default are independent); the report quantifies how much higher
  exposure is in high-price paths, where weaker counterparties are most likely to be stressed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config
from gasrisk.pricing import curve_on, fetch_ecb_curve, zero_rate
from gasrisk.ttf_sim import fit_ttf_garch, simulate_returns


def score_ratio(value: float, higher_is_better: bool, thresholds: list[float]) -> int:
    """Score 1-5: one point plus one for each threshold passed."""
    if higher_is_better:
        return 1 + sum(value >= t for t in thresholds)
    return 1 + sum(value <= t for t in thresholds)


def scorecard(cps: list[dict], sc: dict, grades: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cps:
        row = {"counterparty": c["id"], "type": c["type"]}
        total = 0.0
        for ratio, (w, hib, th) in sc.items():
            s = score_ratio(float(c[ratio]), bool(hib), list(th))
            row[ratio] = c[ratio]
            row[f"score_{ratio}"] = s
            total += float(w) * s
        g = next(g for g in sorted(grades, key=lambda g: -g["min_score"]) if total >= g["min_score"])
        row.update({"weighted_score": total, "grade": g["grade"], "grade_label": g["label"], "pfe_limit_eur": g["pfe_limit_eur"]})
        rows.append(row)
    return pd.DataFrame(rows)


def trade_table(cps: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cps:
        for i, t in enumerate(c["trades"]):
            for m in range(int(t["start"]), int(t["end"]) + 1):
                rows.append({"counterparty": c["id"], "trade": f"{c['id']}-T{i + 1}", "side": t["side"],
                             "sign": 1.0 if t["side"] == "buy" else -1.0, "delivery_month": m,
                             "mwh": float(t["mwh_per_month"]), "strike_x_f": float(t["strike_x_f"])})
    return pd.DataFrame(rows)


def trade_values(trades: pd.DataFrame, F: np.ndarray, F0: float, k: int, curve_row: pd.Series) -> np.ndarray:
    """Values (EUR) at grid month ``k`` of each trade row for prices ``F`` (n_paths). Returns (n_paths x n_rows)."""
    m = trades["delivery_month"].to_numpy()
    alive = m > k
    tau = np.clip(m - k, 0, None) / 12.0
    df = np.exp(-zero_rate(curve_row, tau) * tau)
    payoff = (F[:, None] - trades["strike_x_f"].to_numpy()[None, :] * F0)
    v = trades["sign"].to_numpy() * trades["mwh"].to_numpy() * df * alive
    return payoff * v[None, :]


def exposure_profiles(trades: pd.DataFrame, paths: np.ndarray, F0: float, curve_row: pd.Series, levels: list[float]) -> tuple[pd.DataFrame, dict]:
    """EE and PFE per counterparty and month (with and without netting); also the netted exposures."""
    n_paths, n_months = paths.shape[0], paths.shape[1] - 1
    rows, netted = [], {}
    for cp, tr in trades.groupby("counterparty"):
        E = np.zeros((n_paths, n_months + 1))
        for k in range(n_months + 1):
            V = trade_values(tr, paths[:, k], F0, k, curve_row)
            E[:, k] = np.maximum(V.sum(axis=1), 0.0)
            gross_by_trade = pd.DataFrame(V).T.groupby(tr["trade"].to_numpy()).sum().T.to_numpy()
            gross = np.maximum(gross_by_trade, 0.0).sum(axis=1)
            row = {"counterparty": cp, "month": k, "ee_eur": E[:, k].mean(), "ee_no_netting_eur": gross.mean()}
            for q in levels:
                row[f"pfe_{q * 100:g}_eur"] = float(np.quantile(E[:, k], q))
            row["mean_mtm_eur"] = V.sum(axis=1).mean()
            rows.append(row)
        netted[cp] = E
    return pd.DataFrame(rows), netted


def limit_report(prof: pd.DataFrame, score: pd.DataFrame, ccfg: dict) -> pd.DataFrame:
    lvl = f"pfe_{float(ccfg['limit_pfe_level']) * 100:g}_eur"
    rows = []
    for cp, p in prof.groupby("counterparty"):
        s = score.set_index("counterparty").loc[cp]
        peak = float(p[lvl].max())
        lim = float(s["pfe_limit_eur"])
        util = peak / lim if lim > 0 else (np.inf if peak > 0 else 0.0)
        status = "breach" if util > 1 else ("amber" if util >= float(ccfg["amber_threshold"]) else "green")
        rows.append({"counterparty": cp, "grade": int(s["grade"]), "ce_eur": float(p.loc[p["month"] == 0, "ee_eur"].iloc[0]),
                     "epe_12m_eur": float(p.loc[(p["month"] >= 1) & (p["month"] <= 12), "ee_eur"].mean()),
                     "peak_ee_eur": float(p["ee_eur"].max()),
                     "peak_pfe_95_eur": float(p["pfe_95_eur"].max()), "peak_pfe_97.5_eur": float(p["pfe_97.5_eur"].max()),
                     "month_of_peak_pfe": int(p.loc[p[lvl].idxmax(), "month"]),
                     "netting_benefit_peak_ee": max(0.0, 1 - float(p["ee_eur"].max()) / float(p["ee_no_netting_eur"].max())),
                     "pfe_limit_eur": lim, "utilisation": util, "status": status})
    return pd.DataFrame(rows)


def wrong_way(netted: dict, paths: np.ndarray, top: float) -> pd.DataFrame:
    """Peak-month EE in the top ``top`` share of price paths vs all paths (illustration only)."""
    rows = []
    for cp, E in netted.items():
        k = int(np.argmax(E.mean(axis=0)))
        hi = paths[:, k] >= np.quantile(paths[:, k], 1 - top)
        rows.append({"counterparty": cp, "month": k, "ee_all_paths_eur": E[:, k].mean(),
                     "ee_high_price_paths_eur": E[hi, k].mean(),
                     "ratio_high_to_all": E[hi, k].mean() / E[:, k].mean() if E[:, k].mean() > 0 else np.nan})
    return pd.DataFrame(rows)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    ccfg = dict(cfg["credit"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    as_of = panel.index[-1]
    F0, fx = float(panel["ttf_c2"].iloc[-1]), float(panel["eurpln"].iloc[-1])
    m = fit_ttf_garch(cfg, panel["ttf_risk_ret"], as_of)
    months, dpm, n = int(ccfg["months"]), int(ccfg["days_per_month"]), int(ccfg["n_paths"])
    rng = np.random.default_rng(cfg.seed + 13)
    r, _ = simulate_returns(m, n, months * dpm, rng, "t")
    logp = np.cumsum(r, axis=1)[:, dpm - 1::dpm]
    paths = F0 * np.exp(np.column_stack([np.zeros(n), logp]))
    _, curve_row = curve_on(fetch_ecb_curve(cfg, as_of, offline=True), as_of)
    trades = trade_table(ccfg["counterparties"])
    score = scorecard(ccfg["counterparties"], ccfg["scorecard"], ccfg["grades"])
    prof, netted = exposure_profiles(trades, paths, F0, curve_row, list(ccfg["pfe_levels"]))
    lim = limit_report(prof, score, ccfg)
    lim["peak_pfe_95_pln"] = lim["peak_pfe_95_eur"] * fx
    wwr = wrong_way(netted, paths, float(ccfg["wwr_top_share"]))
    q = np.quantile(paths, [0.025, 0.5, 0.975], axis=0) / F0
    price_fan = pd.DataFrame({"month": range(months + 1), "q2_5_x_f0": q[0], "median_x_f0": q[1], "q97_5_x_f0": q[2]})
    trades_out = trades.groupby(["counterparty", "trade", "side", "strike_x_f", "mwh"]).agg(
        first_month=("delivery_month", "min"), last_month=("delivery_month", "max")).reset_index()
    trades_out["total_mwh"] = trades_out["mwh"] * (trades_out["last_month"] - trades_out["first_month"] + 1)
    tables = {"scorecard": score, "trades": trades_out, "exposure_profile": prof, "limits": lim, "wrong_way": wwr,
              "price_fan": price_fan,
              "model": pd.DataFrame([{"as_of": as_of.date(), "paths": n, "months": months, "garch_mean": m.mean_name,
                                      "nu": m.nu, "persistence": m.alpha + m.beta, "sigma_next_daily": np.sqrt(m.sigma2_next),
                                      "long_run_sigma_daily": np.sqrt(m.omega / (1 - m.alpha - m.beta)),
                                      "ecb_curve_date": curve_on(fetch_ecb_curve(cfg, as_of, offline=True), as_of)[0].date()}])}
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage13_{k}.csv", index=False)
    from gasrisk.credit_report import write_stage13_report

    write_stage13_report(cfg, tables)
    return tables


def main() -> None:
    t = run(load_config())
    for k in ("scorecard", "limits", "wrong_way", "model"):
        print(t[k].to_string(index=False))


if __name__ == "__main__":
    main()
