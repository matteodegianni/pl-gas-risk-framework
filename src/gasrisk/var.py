"""Stage 5: one-day VaR and Expected Shortfall of the SYNTHETIC book in PLN.

Usage::

    python -m gasrisk.var

Book (synthetic, ``config.yaml`` ``book``): short floating TGE purchases behind
fixed-price PLN sales (open volume Q), long TTF futures (hedge ratio x Q, EUR)
and a long EUR/PLN forward (EUR notional = FX hedge ratio x Q x TGE in EUR).

Risk factors: TGE in EUR (TGE PLN / ECB EUR/PLN), TTF second month held to
roll (EUR/MWh) and EUR/PLN (ECB). Every scenario is fully revalued:

* TGE leg:  -Q x (TGE_EUR' x FX' - TGE_EUR x FX)      (PLN)
* TTF leg:  N x (TTF' - TTF) x FX'                     (EUR P&L converted at the new rate)
* FX leg:   N_fx x (FX' - FX)

Methods (walk-forward from ``start_obs``; each origin uses only data up to that day):

1. ``hs``: historical simulation, last ``hs_window`` joint returns.
2. ``normal``: multivariate normal, covariance of the last ``normal_window`` returns.
3. ``garch_t_gauss``: Stage 4 selected GARCH(1,1)-t per factor, Gaussian copula.
4. ``garch_t_tcop``: same marginals, Student-t copula (tail dependence), copula df by ML.
5. ``fhs``: filtered historical simulation: joint standardised residuals rescaled
   by tomorrow's conditional volatility and mean.
6. ``ewma``: RiskMetrics EWMA covariance (lambda 0.94), multivariate normal.

VaR and ES are reported as positive losses in PLN at 95% and 99% (VaR) and
95%, 97.5% (FRTB) and 99% (ES).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from gasrisk.config import Config, load_config
from gasrisk.volatility import MeanSpec, garch_model, garch_variance_path

FACTORS = ["tge_eur", "ttf", "fx"]
METHODS = ["hs", "normal", "garch_t_gauss", "garch_t_tcop", "fhs", "ewma"]
SCALE = 100.0  # GARCH models are fitted on percent returns


# --------------------------------------------------------------------------- book and P&L


def positions(cfg_book: dict, tge_eur: float) -> dict[str, float]:
    """SYNTHETIC book positions on a given day."""
    q = float(cfg_book["q_open_mwh"])
    return {"q_open_mwh": q, "ttf_mwh": float(cfg_book["ttf_hedge_ratio"]) * q,
            "fx_eur": float(cfg_book["fx_hedge_ratio"]) * q * tge_eur}


def pnl_legs(levels: dict[str, float], rets: np.ndarray, pos: dict[str, float]) -> dict[str, np.ndarray]:
    """Full revaluation P&L (PLN) per leg for scenario log returns ``rets`` (n x 3: tge_eur, ttf, fx)."""
    rets = np.atleast_2d(rets)
    g, t, f = levels["tge_eur"], levels["ttf"], levels["fx"]
    g1, t1, f1 = g * np.exp(rets[:, 0]), t * np.exp(rets[:, 1]), f * np.exp(rets[:, 2])
    tge = -pos["q_open_mwh"] * (g1 * f1 - g * f)
    ttf = pos["ttf_mwh"] * (t1 - t) * f1
    fx = pos["fx_eur"] * (f1 - f)
    return {"tge": tge, "ttf": ttf, "fx": fx, "total": tge + ttf + fx}


def var_es(pnl: np.ndarray, var_levels: list[float], es_levels: list[float]) -> dict[str, float]:
    """VaR and ES as positive losses from a P&L sample."""
    loss = -np.asarray(pnl)
    out = {}
    for a in var_levels:
        out[f"var_{int(round(a * 1000)) / 10:g}"] = float(np.quantile(loss, a))
    for a in es_levels:
        q = np.quantile(loss, a)
        out[f"es_{int(round(a * 1000)) / 10:g}"] = float(loss[loss >= q].mean())
    return out


# --------------------------------------------------------------------------- data


def build_factors(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor levels and log returns on the panel calendar.

    TTF returns are the roll-adjusted returns of the second month held to roll; the two
    roll days excluded in Stage 1 are NaN and excluded from estimation windows and backtests.
    """
    levels = pd.DataFrame({"tge_eur": panel["tge_eur"], "ttf": panel["ttf_c2"], "fx": panel["eurpln"]})
    rets = pd.DataFrame({"tge_eur": np.log(panel["tge_eur"]).diff(), "ttf": panel["ttf_risk_ret"],
                         "fx": panel["eurpln_ret"]})
    return levels, rets


def factor_specs(selected: pd.DataFrame, mean_candidates: list[dict]) -> dict[str, tuple[MeanSpec, str]]:
    """Mean spec and distribution per factor from Stage 4 (TGE in EUR uses TGE's specification)."""
    by = selected.set_index("series")
    cands = {d["name"]: MeanSpec.from_cfg(d) for d in mean_candidates}
    m = {"tge_eur": "tge", "ttf": "ttf", "fx": "eurpln"}
    return {f: (cands[by.loc[s, "mean"]], by.loc[s, "dist"]) for f, s in m.items()}


# --------------------------------------------------------------------------- GARCH pieces


def mean_forecast(params: pd.Series, spec: MeanSpec, hist: np.ndarray, next_dow: int) -> float:
    """One-step conditional mean: const + sum phi_i y_{t+1-i} + day-of-week effect of the next day."""
    mu = float(params.get("Const", params.get("mu", 0.0)))
    for name, v in params.items():
        if name.startswith(("alpha", "beta", "omega")):
            continue
        m = re.search(r"\[(\d+)\]$", str(name))
        if m:
            mu += v * hist[-int(m.group(1))]
    if spec.dow and next_dow > 0:
        mu += float(params.get(f"dow_{next_dow}", 0.0))
    return mu


def fit_copula_nu(z: np.ndarray, corr: np.ndarray, grid: list[float]) -> float:
    """Student-t copula degrees of freedom by maximum likelihood over a grid (pseudo-observations)."""
    n, d = z.shape
    u = (stats.rankdata(z, axis=0) - 0.5) / n
    best, best_ll = grid[-1], -np.inf
    for nu in grid:
        x = stats.t.ppf(u, df=nu)
        ll = stats.multivariate_t.logpdf(x, loc=np.zeros(d), shape=corr, df=nu).sum() - stats.t.logpdf(x, df=nu).sum()
        if ll > best_ll:
            best, best_ll = nu, ll
    return float(best)


def copula_shocks(rng: np.random.Generator, corr: np.ndarray, n: int, copula: str, nu_c: float | None) -> np.ndarray:
    """Uniforms (n x d) from a Gaussian or Student-t copula with correlation ``corr``."""
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0])) @ L.T
    if copula == "gauss":
        return stats.norm.cdf(z)
    w = rng.chisquare(nu_c, size=(n, 1)) / nu_c
    return stats.t.cdf(z / np.sqrt(w), df=nu_c)


def _nearest_corr(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    c2 = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


# --------------------------------------------------------------------------- walk-forward


def run_var(levels: pd.DataFrame, rets: pd.DataFrame, specs: dict, cfg_book: dict, vcfg: dict, seed: int) -> pd.DataFrame:
    """VaR/ES forecasts for every origin and method, with the realised next-day P&L."""
    idx = rets.index
    R = rets[FACTORS].to_numpy()
    n = len(idx)
    hs_w, nw, gw, every = int(vcfg["hs_window"]), int(vcfg["normal_window"]), int(vcfg["garch_window"]), int(vcfg["refit_every"])
    lam, nsim = float(vcfg["ewma_lambda"]), int(vcfg["mc_sims"])
    vl, el = list(vcfg["var_levels"]), list(vcfg["es_levels"])
    start = int(vcfg["start_obs"])

    # EWMA covariance recursion (zero mean), skipping NaN rows
    S = np.cov(R[1:64][~np.isnan(R[1:64]).any(1)].T)
    ewma_cov = [None] * (n + 1)
    for t in range(64, n):
        if not np.isnan(R[t]).any():
            S = lam * S + (1 - lam) * np.outer(R[t], R[t])
        ewma_cov[t + 1] = S.copy()

    params: dict[str, pd.Series] = {}
    lo = 0
    corr, nu_c, zjoint = None, None, None
    rows = []
    for pos in range(start, n):
        lv = {f: float(levels[f].iloc[pos]) for f in FACTORS}
        pz = positions(cfg_book, lv["tge_eur"])
        rng = np.random.default_rng(seed + pos)
        out: dict[str, dict] = {}

        hist = R[max(0, pos - hs_w + 1):pos + 1]
        hist = hist[~np.isnan(hist).any(1)]
        out["hs"] = var_es(pnl_legs(lv, hist, pz)["total"], vl, el)

        win = R[max(0, pos - nw + 1):pos + 1]
        win = win[~np.isnan(win).any(1)]
        sims = rng.multivariate_normal(win.mean(0), np.cov(win.T), size=nsim)
        out["normal"] = var_es(pnl_legs(lv, sims, pz)["total"], vl, el)

        sims = rng.multivariate_normal(np.zeros(3), ewma_cov[pos + 1], size=nsim)
        out["ewma"] = var_es(pnl_legs(lv, sims, pz)["total"], vl, el)

        # GARCH marginals (percent returns), refit every ``every`` origins on ``gw`` observations
        if (pos - start) % every == 0:
            lo = pos - gw + 1
            zs = {}
            for f in FACTORS:
                y = rets[f].iloc[lo:pos + 1].dropna() * SCALE
                spec, dist = specs[f]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = garch_model(y, dist, spec).fit(disp="off", options={"maxiter": 2000})
                params[f] = res.params
                zs[f] = pd.Series(res.std_resid, index=y.index)
            zjoint = pd.DataFrame(zs).dropna()
            corr = _nearest_corr(np.corrcoef(zjoint.to_numpy().T))
            nu_c = fit_copula_nu(zjoint.to_numpy(), corr, list(vcfg["copula_nu_grid"]))
        mu, sig, nus = np.zeros(3), np.zeros(3), np.zeros(3)
        next_dow = int(idx[pos + 1].day_of_week) if pos + 1 < n else int((idx[pos] + pd.offsets.BDay(1)).day_of_week)
        for i, f in enumerate(FACTORS):
            y = rets[f].iloc[lo:pos + 1].dropna() * SCALE
            spec, dist = specs[f]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fx_ = garch_model(y, dist, spec).fix(params[f].to_numpy())
            s2 = garch_variance_path(params[f], float(np.asarray(fx_.resid)[-1]),
                                     float(np.asarray(fx_.conditional_volatility)[-1]) ** 2, 1)[0]
            mu[i] = mean_forecast(params[f], spec, y.to_numpy(), next_dow) / SCALE
            sig[i] = np.sqrt(s2) / SCALE
            nus[i] = float(params[f]["nu"])
        for name, cop in (("garch_t_gauss", "gauss"), ("garch_t_tcop", "t")):
            u = copula_shocks(rng, corr, nsim, cop, nu_c)
            zsim = np.column_stack([stats.t.ppf(u[:, i], df=nus[i]) / np.sqrt(nus[i] / (nus[i] - 2)) for i in range(3)])
            out[name] = var_es(pnl_legs(lv, mu + sig * zsim, pz)["total"], vl, el)
        out["fhs"] = var_es(pnl_legs(lv, mu + sig * zjoint.to_numpy(), pz)["total"], vl, el)

        realised = {k: np.nan for k in ("pnl", "pnl_tge", "pnl_ttf", "pnl_fx")}
        if pos + 1 < n and not np.isnan(R[pos + 1]).any():
            legs = pnl_legs(lv, R[pos + 1], pz)
            realised = {"pnl": float(legs["total"][0]), "pnl_tge": float(legs["tge"][0]),
                        "pnl_ttf": float(legs["ttf"][0]), "pnl_fx": float(legs["fx"][0])}
        state = {"copula_nu": nu_c, "corr_tge_ttf": corr[0, 1], "corr_tge_fx": corr[0, 2], "corr_ttf_fx": corr[1, 2],
                 **{f"sig_{f}": sig[i] for i, f in enumerate(FACTORS)}, **{f"mu_{f}": mu[i] for i, f in enumerate(FACTORS)},
                 **{f"level_{f}": lv[f] for f in FACTORS}}
        for m in METHODS:
            rows.append({"origin": idx[pos], "target": idx[pos + 1] if pos + 1 < n else pd.NaT, "method": m,
                         **out[m], **realised, **state})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    vcfg, book = dict(cfg["var"]), dict(cfg["book"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    levels, rets = build_factors(panel)
    selected = pd.read_csv(cfg.path("outputs_dir") / "stage4_selected_spec.csv")
    specs = factor_specs(selected, cfg["volatility"]["mean_candidates"])
    fc = run_var(levels, rets, specs, book, vcfg, cfg.seed)
    fc.to_csv(cfg.path("clean_dir") / "var_forecasts.csv", index=False)

    from gasrisk.var_report import summarise, write_stage5_report

    tables = summarise(fc, levels, book, cfg)
    out = cfg.path("outputs_dir")
    for name, t in tables.items():
        t.to_csv(out / f"stage5_{name}.csv", index=False)
    write_stage5_report(cfg, tables, fc)
    return tables


def main() -> None:
    cfg = load_config()
    t = run(cfg)
    print(t["var_today"].to_string(index=False))


if __name__ == "__main__":
    main()
