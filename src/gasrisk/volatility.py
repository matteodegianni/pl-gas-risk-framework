"""Stage 4: GARCH(1,1) volatility models and out-of-sample volatility forecast evaluation.

Usage::

    python -m gasrisk.volatility

Series: TTF (roll-adjusted), TGE, EUR/PLN (ECB) and the location plus tenor
basis (TGE day-ahead spot minus TTF second month). Log returns are scaled to
percent; basis changes stay in EUR/MWh.

1. **Mean equation.** Stage 5 filtered historical simulation uses the
   standardised residuals, so they must be free of autocorrelation. Candidate
   mean equations (constant, AR(p), day-of-week dummies, combinations) are
   fitted with GARCH(1,1)-t. Rule: keep the constant mean if its standardised
   residuals pass Ljung-Box at every tested lag; otherwise choose the lowest-BIC
   candidate that passes. The selection is repeated on the first training
   window only, to check that it does not depend on future data.
2. **Innovation distribution.** Normal and Student-t for every series, plus
   skewed-t (Hansen) where configured. Chosen by BIC and by two-sided tail
   calibration at the 1% level; disagreements are reported.
3. **Out-of-sample.** Rolling window, parameters re-estimated every
   ``refit_every`` days and the variance filter updated daily with fixed
   parameters. Cumulative h-day variance forecasts account for the AR mean
   (the h-day forecast error is a weighted sum of future shocks). Benchmarks:
   EWMA (RiskMetrics) and rolling historical variance. Realised variance is
   proxied by the sum of squared daily returns; forecasts are ranked with MSE
   and QLIKE (Patton 2011), DM tests (HAC h-1 lags, HLN) with Holm adjustment,
   and Mincer-Zarnowitz regressions at h=1.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from gasrisk.config import Config, load_config
from gasrisk.forecasting import diebold_mariano_full, holm

INTEGRATED_THRESHOLD = 0.999  # persistence at or above this is reported as near-integrated (IGARCH-like)
UNITS = {"ttf": "% log return", "tge": "% log return", "eurpln": "% log return", "basis": "EUR/MWh change"}


# --------------------------------------------------------------------------- data and specification


def build_returns(panel: pd.DataFrame, scale: float) -> dict[str, pd.Series]:
    """Return series for volatility modelling (percent log returns; basis changes in EUR/MWh).

    The two TTF roll days excluded as misaligned in Stage 1 are dropped rather than set to zero.
    """
    return {
        "ttf": (panel["ttf_risk_ret"] * scale).dropna().rename("ttf"),
        "tge": (panel["tge_ret"] * scale).dropna().rename("tge"),
        "eurpln": (panel["eurpln_ret"] * scale).dropna().rename("eurpln"),
        "basis": panel["basis_tge_ttfc2_change"].dropna().rename("basis"),
    }


@dataclass(frozen=True)
class MeanSpec:
    name: str
    lags: int | tuple[int, ...]
    dow: bool

    @staticmethod
    def from_cfg(d: dict) -> "MeanSpec":
        lags = d["lags"]
        return MeanSpec(d["name"], tuple(lags) if isinstance(lags, list) else int(lags), bool(d["dow"]))

    @property
    def lag_list(self) -> list[int]:
        if isinstance(self.lags, tuple):
            return list(self.lags)
        return list(range(1, self.lags + 1))


def dow_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.get_dummies(pd.Series(index.day_of_week, index=index), prefix="dow").astype(float)
    return d.drop(columns=[c for c in d.columns if c == "dow_0"])


def garch_model(y: pd.Series, dist: str, spec: MeanSpec):
    """arch model: GARCH(1,1) with the given mean specification and innovation distribution."""
    from arch import arch_model

    x = dow_dummies(y.index) if spec.dow else None
    if not spec.lag_list and not spec.dow:
        return arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    lags = list(spec.lags) if isinstance(spec.lags, tuple) else spec.lags
    return arch_model(y, x=x, mean="ARX", lags=lags, vol="GARCH", p=1, q=1, dist=dist, rescale=False)


def ar_coefficients(params: pd.Series, spec: MeanSpec) -> np.ndarray:
    """AR coefficients phi_1..phi_pmax (zeros for lags not in the model)."""
    pmax = max(spec.lag_list, default=0)
    phi = np.zeros(pmax)
    for name, v in params.items():
        if name.startswith(("alpha", "beta")):
            continue
        m = re.search(r"\[(\d+)\]$", str(name))
        if m:
            phi[int(m.group(1)) - 1] = v
    return phi


def psi_weights(phi: np.ndarray, n: int) -> np.ndarray:
    """MA(infinity) weights of an AR process: psi_0 = 1, psi_j = sum_i phi_i psi_{j-i}."""
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = sum(phi[i - 1] * psi[j - i] for i in range(1, min(j, len(phi)) + 1))
    return psi


def aggregate_variance(sigma2_path: np.ndarray, phi: np.ndarray, h: int) -> float:
    """Variance of the h-day cumulative forecast error under an AR mean.

    sum_{k=1..h} y_{t+k} - E_t[...] = sum_{k=1..h} eps_{t+k} * sum_{j=0..h-k} psi_j, so its
    variance is sum_k sigma2_{t+k} * (sum_{j<=h-k} psi_j)^2. With no AR terms this is sum_k sigma2_{t+k}.
    """
    cum = np.cumsum(psi_weights(phi, h))
    return float(sum(sigma2_path[k - 1] * cum[h - k] ** 2 for k in range(1, h + 1)))


def garch_variance_path(params: pd.Series, resid_last: float, sigma2_last: float, H: int) -> np.ndarray:
    """GARCH(1,1) conditional variance forecasts for t+1..t+H made at t."""
    w, a, b = params["omega"], params["alpha[1]"], params["beta[1]"]
    out = np.empty(H)
    out[0] = w + a * resid_last**2 + b * sigma2_last
    for k in range(1, H):
        out[k] = w + (a + b) * out[k - 1]
    return out


# --------------------------------------------------------------------------- distributions


def _dist_params(params: pd.Series, dist: str) -> list[float]:
    if dist == "t":
        return [float(params["nu"])]
    if dist == "skewt":
        return [float(params["eta"]), float(params["lambda"])]
    return []


def standardized_cdf(z: np.ndarray, dist: str, dparams: list[float]) -> np.ndarray:
    """CDF of the fitted standardised (unit-variance) innovation."""
    if dist == "t":
        nu = dparams[0]
        return stats.t.cdf(z * np.sqrt(nu / (nu - 2)), df=nu)
    if dist == "skewt":
        from arch.univariate import SkewStudent

        return np.asarray(SkewStudent().cdf(np.asarray(z), dparams))
    return stats.norm.cdf(z)


def standardized_ppf(q: float, dist: str, dparams: list[float]) -> float:
    if dist == "t":
        nu = dparams[0]
        return float(stats.t.ppf(q, df=nu) / np.sqrt(nu / (nu - 2)))
    if dist == "skewt":
        from arch.univariate import SkewStudent

        return float(SkewStudent().ppf(q, dparams))
    return float(stats.norm.ppf(q))


# --------------------------------------------------------------------------- full-sample fits


def _fit(y: pd.Series, dist: str, spec: MeanSpec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return garch_model(y, dist, spec).fit(disp="off", options={"maxiter": 2000})


def fit_summary(r: pd.Series, dist: str, spec: MeanSpec, lb_lags: list[int]) -> tuple[dict, pd.Series, pd.Series]:
    """Fit one specification; return a summary row, conditional volatility and standardised residuals."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    res = _fit(r, dist, spec)
    p, se = res.params, res.std_err
    a, b = p["alpha[1]"], p["beta[1]"]
    dp = _dist_params(p, dist)
    z = pd.Series(res.std_resid, index=r.index).dropna()
    lb = acorr_ljungbox(z, lags=lb_lags, return_df=True)["lb_pvalue"]
    q = {k: standardized_ppf(k, dist, dp) for k in (0.01, 0.05, 0.95, 0.99)}
    persistence = a + b
    row = {
        "mean": spec.name, "dist": dist, "n": int(res.nobs), "loglik": float(res.loglikelihood),
        "aic": float(res.aic), "bic": float(res.bic),
        "ar_coefs": ", ".join(f"{c:.3f}" for c in ar_coefficients(p, spec)) if spec.lag_list else "",
        "omega": p["omega"], "alpha": a, "alpha_se": se["alpha[1]"], "beta": b, "beta_se": se["beta[1]"],
        "nu": dp[0] if dist in ("t", "skewt") else None, "skew_lambda": dp[1] if dist == "skewt" else None,
        "persistence": persistence, "near_integrated": bool(persistence >= INTEGRATED_THRESHOLD),
        "half_life_days": float(np.log(0.5) / np.log(persistence)) if 0 < persistence < INTEGRATED_THRESHOLD else np.nan,
        **{f"lb_z_p{L}": float(lb.loc[L]) for L in lb_lags},
        "lb_z2_p10": float(acorr_ljungbox(z**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]),
        "arch_lm_z_p5": float(het_arch(z - z.mean(), nlags=5)[1]),
        "z_excess_kurtosis": float(stats.kurtosis(z, bias=False)),
        "pit_ks_p": float(stats.kstest(standardized_cdf(z.to_numpy(), dist, dp), "uniform").pvalue),
        "share_below_q01": float((z < q[0.01]).mean()), "share_below_q05": float((z < q[0.05]).mean()),
        "share_above_q95": float((z > q[0.95]).mean()), "share_above_q99": float((z > q[0.99]).mean()),
    }
    row["calib_error_1pct"] = abs(row["share_below_q01"] - 0.01) + abs(row["share_above_q99"] - 0.01)
    row["lb_passes"] = bool(all(row[f"lb_z_p{L}"] >= 0.05 for L in lb_lags))
    return row, pd.Series(res.conditional_volatility, index=r.index), z


def select_mean(r: pd.Series, candidates: list[MeanSpec], lb_lags: list[int], alpha: float) -> tuple[MeanSpec, pd.DataFrame]:
    """Mean-equation selection with GARCH(1,1)-t (see module docstring for the rule)."""
    rows = []
    for spec in candidates:
        row, _, _ = fit_summary(r, "t", spec, lb_lags)
        row["lb_passes"] = bool(all(row[f"lb_z_p{L}"] >= alpha for L in lb_lags))
        rows.append(row)
    t = pd.DataFrame(rows)
    if t.loc[t["mean"] == "constant", "lb_passes"].any():
        chosen, reason = "constant", "constant mean passes Ljung-Box at all lags"
    elif t["lb_passes"].any():
        chosen = t[t["lb_passes"]].sort_values("bic").iloc[0]["mean"]
        reason = "lowest BIC among candidates passing Ljung-Box at all lags"
    else:
        chosen = t.sort_values("bic").iloc[0]["mean"]
        reason = "no candidate passes Ljung-Box; lowest BIC used (flagged)"
    t["selected"] = t["mean"] == chosen
    t["selection_reason"] = np.where(t["selected"], reason, "")
    return next(c for c in candidates if c.name == chosen), t


def select_dist(fits: pd.DataFrame) -> tuple[str, str]:
    """Distribution chosen by BIC and two-sided 1% tail calibration; returns (dist, reason)."""
    by_bic = fits.loc[fits["bic"].idxmin(), "dist"]
    by_cal = fits.loc[fits["calib_error_1pct"].idxmin(), "dist"]
    if by_bic == by_cal:
        return by_bic, "best on both BIC and two-sided 1% tail calibration"
    return by_cal, f"BIC prefers {by_bic}, calibration prefers {by_cal}; calibration chosen because the model feeds 99% VaR"


# --------------------------------------------------------------------------- forecasts


def ewma_variance(r: np.ndarray, lam: float, init_window: int = 63) -> np.ndarray:
    """One-step-ahead EWMA variance: s2[t] is the forecast for day t made at t-1 (zero mean).

    s2[t+1] = lam * s2[t] + (1 - lam) * r[t]^2, initialised with the mean square of the
    first ``init_window`` returns. Entries before ``init_window`` are NaN.
    """
    n = len(r)
    s2 = np.full(n + 1, np.nan)
    s2[init_window] = np.mean(r[:init_window] ** 2)
    for t in range(init_window, n):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    return s2  # length n + 1: s2[n] is the forecast for the day after the sample


def walk_forward_vol(r: pd.Series, cfg: dict, spec: MeanSpec | None = None, dists: list[str] | None = None) -> pd.DataFrame:
    """Out-of-sample cumulative-variance forecasts for every model and horizon.

    At origin t (after observing r[t]) each model forecasts the variance of
    sum_{k=1..h} r[t+k]; the realised proxy is sum_{k=1..h} r[t+k]^2.
    """
    spec = spec or MeanSpec("constant", 0, False)
    dists = dists or list(cfg["distributions"])
    y = r.to_numpy()
    idx = r.index
    n, W, every = len(y), int(cfg["train_window"]), int(cfg["refit_every"])
    horizons = list(cfg["horizons"])
    H = max(horizons)
    ewma = ewma_variance(y, float(cfg["ewma_lambda"]))
    roll = pd.Series(y**2).rolling(int(cfg["rolling_window"])).mean().to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(y**2)])
    params: dict[str, pd.Series] = {}
    lo = 0
    rows = []
    for pos in range(W, n - min(horizons)):
        if (pos - W) % every == 0:
            lo = pos - W + 1
            for dist in dists:
                params[dist] = _fit(r.iloc[lo:pos + 1], dist, spec).params
        fc = {}
        for dist in dists:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fx = garch_model(r.iloc[lo:pos + 1], dist, spec).fix(params[dist].to_numpy())
            resid = np.asarray(fx.resid)[-1]
            s2 = np.asarray(fx.conditional_volatility)[-1] ** 2
            path = garch_variance_path(params[dist], resid, s2, H)
            phi = ar_coefficients(params[dist], spec)
            fc[f"garch_{dist}"] = np.array([aggregate_variance(path, phi, h) for h in range(1, H + 1)])
        fc["ewma"] = ewma[pos + 1] * np.arange(1, H + 1)
        fc["rolling"] = roll[pos] * np.arange(1, H + 1)
        for h in horizons:
            if pos + h > n - 1:
                continue
            realised = csum[pos + h + 1] - csum[pos + 1]
            for m, v in fc.items():
                rows.append((idx[pos], h, m, float(v[h - 1]), float(realised)))
    return pd.DataFrame(rows, columns=["origin", "h", "model", "forecast_var", "realised_var"])


# --------------------------------------------------------------------------- evaluation


def qlike(realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss log(F) + RV/F: robust to noise in the variance proxy, defined when RV = 0."""
    return np.log(forecast) + realised / forecast


def mincer_zarnowitz(realised: pd.Series, forecast: pd.Series) -> dict:
    """RV = a + b F + e with HAC errors; an unbiased forecast has a = 0, b = 1."""
    import statsmodels.api as sm

    res = sm.OLS(realised.to_numpy(), sm.add_constant(forecast.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    b, b_se = res.params[1], res.bse[1]
    return {"mz_a": float(res.params[0]), "mz_b": float(b), "mz_b_se": float(b_se),
            "mz_b_eq_1_p": float(2 * stats.norm.sf(abs((b - 1) / b_se))), "mz_r2": float(res.rsquared)}


def _dm_on_loss(loss_m: np.ndarray, loss_b: np.ndarray, h: int) -> float:
    """One-sided DM p-value that model loss is lower (losses passed as square roots of non-negative values)."""
    return diebold_mariano_full(np.sqrt(loss_m), np.sqrt(loss_b), h)["p_better"]


def evaluate_vol(fc: pd.DataFrame, benchmark: str, alpha: float) -> pd.DataFrame:
    rows = []
    for (s, h), g in fc.groupby(["series", "h"]):
        b = g[g["model"] == benchmark].set_index("origin")
        for m, gm in g.groupby("model"):
            gm = gm.set_index("origin")
            common = gm.index.intersection(b.index)
            gm, bb = gm.loc[common], b.loc[common]
            ok = (gm["forecast_var"] > 0) & (bb["forecast_var"] > 0)
            gm, bb = gm[ok], bb[ok]
            mse = (gm["realised_var"] - gm["forecast_var"]) ** 2
            mse_b = (bb["realised_var"] - bb["forecast_var"]) ** 2
            ql = qlike(gm["realised_var"].to_numpy(), gm["forecast_var"].to_numpy())
            ql_b = qlike(bb["realised_var"].to_numpy(), bb["forecast_var"].to_numpy())
            shift = -min(ql.min(), ql_b.min()) + 1.0  # QLIKE can be negative; a common shift keeps differences
            row = {"series": s, "h": int(h), "model": m, "n": len(gm), "mse": float(mse.mean()),
                   "qlike": float(ql.mean()), "mse_ratio_vs_bench": float(mse.mean() / mse_b.mean()),
                   "qlike_diff_vs_bench": float(ql.mean() - ql_b.mean())}
            if m == benchmark:
                row.update(dm_p_qlike=np.nan, dm_p_mse=np.nan)
            else:
                row["dm_p_qlike"] = _dm_on_loss(ql + shift, ql_b + shift, int(h))
                row["dm_p_mse"] = _dm_on_loss(mse.to_numpy(), mse_b.to_numpy(), int(h))
            if h == 1:
                row.update(mincer_zarnowitz(gm["realised_var"], gm["forecast_var"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    tested = out["model"] != benchmark
    out["holm_p_qlike"] = np.nan
    out.loc[tested, "holm_p_qlike"] = holm(out.loc[tested, "dm_p_qlike"].to_numpy())
    out["verdict_qlike"] = np.where(~tested, "benchmark",
                                    np.where(out["holm_p_qlike"] < alpha, "better than benchmark (Holm)",
                                             np.where(out["dm_p_qlike"] < alpha, "better (raw only)", "not better")))
    order = {"garch_normal": 0, "garch_t": 1, "garch_skewt": 2, "ewma": 3, "rolling": 4}
    return out.sort_values(["series", "h", "model"], key=lambda c: c.map(order) if c.name == "model" else c).reset_index(drop=True)


# --------------------------------------------------------------------------- run


def _run_series(name: str, r: pd.Series, cfg: dict) -> dict:
    alpha = float(cfg["alpha"])
    lb_lags = list(cfg["lb_lags"])
    cands = [MeanSpec.from_cfg(d) for d in cfg["mean_candidates"]]
    spec, mean_tbl = select_mean(r, cands, lb_lags, alpha)
    spec_first, _ = select_mean(r.iloc[: int(cfg["train_window"])], cands, lb_lags, alpha)
    dists = list(cfg["distributions"]) + list(cfg.get("extra_distributions", {}).get(name, []))
    fits, vols, zs = [], {}, {}
    for dist in dists:
        row, vol, z = fit_summary(r, dist, spec, lb_lags)
        fits.append({"series": name, "unit": UNITS[name], **row})
        vols[f"{name}_{dist}"] = vol
        zs[f"{name}_{dist}"] = z
    const_row, _, _ = fit_summary(r, "t", MeanSpec("constant", 0, False), lb_lags)  # "before" for the report
    fit_df = pd.DataFrame(fits)
    dist, dist_reason = select_dist(fit_df)
    fc = walk_forward_vol(r, cfg, spec, dists).assign(series=name)
    sel = {"series": name, "mean": spec.name, "mean_lags": str(spec.lag_list), "mean_dow": spec.dow, "dist": dist,
           "mean_reason": mean_tbl.loc[mean_tbl["selected"], "selection_reason"].iloc[0], "dist_reason": dist_reason,
           "mean_selected_on_first_window": spec_first.name, "same_on_first_window": spec_first.name == spec.name}
    return {"fc": fc, "fits": fit_df, "mean_tbl": mean_tbl.assign(series=name), "vol": pd.DataFrame(vols),
            "z": pd.DataFrame(zs), "sel": sel, "before": {"series": name, **const_row}}


def conditional_vol_figure(vol: pd.DataFrame, selected: pd.DataFrame, path) -> None:
    """Annualised conditional volatility of the selected GARCH model (a statistic, no price levels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(selected["series"])
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2 * len(names)), sharex=True)
    for ax, (n, d) in zip(axes, zip(selected["series"], selected["dist"])):
        s = vol[f"{n}_{d}"].dropna() * np.sqrt(252)
        ax.plot(s.index, s.values, lw=0.8, color="#3b6ea5")
        unit = "% p.a." if n != "basis" else "EUR/MWh x sqrt(252)"
        ax.set_title(f"{n}: selected GARCH(1,1) ({d}) conditional volatility, annualised ({unit})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    from joblib import Parallel, delayed

    vcfg = dict(cfg["volatility"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    rets = build_returns(panel, float(vcfg["return_scale"]))
    parts = Parallel(n_jobs=int(vcfg["n_jobs"]))(delayed(_run_series)(n, rets[n], vcfg) for n in vcfg["series"])
    fc = pd.concat([p["fc"] for p in parts], ignore_index=True)
    selected = pd.DataFrame([p["sel"] for p in parts])
    clean = cfg.path("clean_dir")
    fc.to_csv(clean / "vol_forecasts.csv", index=False)
    pd.concat([p["vol"] for p in parts], axis=1).to_csv(clean / "garch_conditional_vol.csv")
    pd.concat([p["z"] for p in parts], axis=1).to_csv(clean / "garch_std_resid.csv")

    mean_tbl = pd.concat([p["mean_tbl"] for p in parts], ignore_index=True)
    before = pd.DataFrame([p["before"] for p in parts])
    after = mean_tbl[mean_tbl["selected"]]
    lb_cols = [f"lb_z_p{L}" for L in vcfg["lb_lags"]]
    ar1 = mean_tbl[mean_tbl["mean"] == "ar1"]
    lb_compare = pd.concat([
        before[["series"] + lb_cols + ["bic"]].assign(model="constant mean (before)"),
        ar1[["series"] + lb_cols + ["bic"]].assign(model="AR(1)"),
        after[["series", "mean"] + lb_cols + ["bic"]].rename(columns={"mean": "selected"}).assign(model="selected (after)"),
    ], ignore_index=True)[["series", "model", "selected"] + lb_cols + ["bic"]].sort_values(["series", "model"], kind="stable")

    tables = {
        "selected_spec": selected,
        "mean_selection": mean_tbl[["series", "mean", "bic", *lb_cols, "lb_passes", "selected", "selection_reason", "ar_coefs"]],
        "lb_before_after": lb_compare,
        "garch_fit": pd.concat([p["fits"] for p in parts], ignore_index=True),
        "vol_forecast_eval": evaluate_vol(fc, vcfg["benchmark"], float(vcfg["alpha"])),
        "test_window": fc.groupby(["series", "h"]).agg(first_origin=("origin", "min"), last_origin=("origin", "max"),
                                                       n=("origin", "nunique")).reset_index(),
    }
    out = cfg.path("outputs_dir")
    for name, t in tables.items():
        t.to_csv(out / f"stage4_{name}.csv", index=False)
    (out / "figures").mkdir(exist_ok=True)
    conditional_vol_figure(pd.concat([p["vol"] for p in parts], axis=1), selected, out / "figures" / "stage4_conditional_vol.png")
    from gasrisk.volatility_report import write_stage4_report

    write_stage4_report(cfg, tables)
    return tables


def main() -> None:
    cfg = load_config()
    t = run(cfg)
    print(t["selected_spec"].to_string(index=False))
    print(t["lb_before_after"].to_string(index=False))


if __name__ == "__main__":
    main()
