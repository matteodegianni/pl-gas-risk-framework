"""Stage 2: statistical diagnostics and EUR/PLN source timing.

Usage::

    python -m gasrisk.diagnostics

Reads ``data/clean/panel.csv`` (Stage 1) and writes ``outputs/stage2_*.csv``,
``outputs/figures/stage2_acf.png`` and ``outputs/stage2_report.md``.

Contents:

* **FX timing.** Daily returns of series sampled at different times of day are
  asynchronous: news between the two sampling times shows up in one series
  today and in the other tomorrow. The EUR/PLN source whose correlation with
  TTF is concentrated at lag 0 (least leakage into lags -1 and +1) is the one
  sampled closest to the TTF settlement.
* **Stationarity.** ADF (null: unit root) and KPSS (null: stationary) on log
  price levels and on returns; reading both together separates "stationary",
  "unit root", "conflicting" and "inconclusive".
* **Serial dependence.** Ljung-Box on returns (predictability of the mean) and
  on squared returns plus ARCH-LM (volatility clustering).
* **Distribution.** Skewness, excess kurtosis, Jarque-Bera and tail frequencies
  against the normal distribution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from gasrisk.config import Config, load_config

# --------------------------------------------------------------------------- series


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    description: str
    level: pd.Series        # log price (or basis level in EUR/MWh)
    ret: pd.Series          # log return (or basis change in EUR/MWh)
    level_label: str
    ret_label: str


def build_series(panel: pd.DataFrame) -> list[SeriesSpec]:
    """Risk-factor series analysed in Stage 2.

    The TTF level is a roll-adjusted index (cumulated same-contract returns), so
    contract switches do not create artificial jumps in the unit-root tests.
    The two roll days excluded as misaligned in Stage 1 contribute zero to the
    index.
    """
    ttf_ret = panel["ttf_risk_ret"]
    ttf_index = np.log(panel["ttf_c2"].iloc[0]) + ttf_ret.fillna(0.0).cumsum()
    return [
        SeriesSpec("ttf", "TTF second month held to roll (ICE Endex), EUR/MWh", ttf_index, ttf_ret,
                   "log roll-adjusted index", "log return (roll-adjusted)"),
        SeriesSpec("tge", "TGEgasDA by trade date, PLN/MWh", np.log(panel["tge_pln"]), panel["tge_ret"],
                   "log price", "log return"),
        SeriesSpec("eurpln", "EUR/PLN risk source", np.log(panel["eurpln"]), panel["eurpln_ret"],
                   "log rate", "log return"),
        SeriesSpec("basis", BASIS_LABEL + ", EUR/MWh", panel["basis_tge_ttfc2_eur"],
                   panel["basis_tge_ttfc2_change"], "location plus tenor basis level (EUR/MWh)",
                   "location plus tenor basis change (EUR/MWh, roll-adjusted)"),
    ]


BASIS_LABEL = "location plus tenor basis (TGE day-ahead spot in EUR minus TTF second month)"


def basis_tail_attribution(panel: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Which leg drives the extreme moves of the location plus tenor basis.

    Basis change = change in TGE (EUR) minus the within-contract change in TTF.
    On days where the basis change exceeds ``z_threshold`` standard deviations,
    compare the size of the two legs and check whether the TGE leg reverses the
    next day (the signature of a spot spike rather than a level shift).
    """
    d_tge = panel["tge_eur"].diff()
    d_ttf = panel["ttf_c2"].shift(1) * np.expm1(panel["ttf_r_c2_adj"])
    df = pd.concat({"basis": panel["basis_tge_ttfc2_change"], "tge": d_tge, "ttf": d_ttf,
                    "tge_next": d_tge.shift(-1)}, axis=1).dropna(subset=["basis", "tge", "ttf"])
    z = (df["basis"] - df["basis"].mean()) / df["basis"].std()
    tail = df[z.abs() > z_threshold]
    tge_dom = tail["tge"].abs() > tail["ttf"].abs()
    dom = tail[tge_dom]
    reverses = (np.sign(dom["tge_next"]) == -np.sign(dom["tge"])) & (dom["tge_next"].abs() >= 0.5 * dom["tge"].abs())
    rows = {
        "days": len(df),
        "tail_days_abs_z_gt_3": len(tail),
        "tail_days_tge_leg_larger": int(tge_dom.sum()),
        "share_tail_days_tge_leg_larger": float(tge_dom.mean()) if len(tail) else float("nan"),
        "median_abs_tge_leg_over_abs_basis_tail": float((tail["tge"].abs() / tail["basis"].abs()).median()),
        "median_abs_ttf_leg_over_abs_basis_tail": float((tail["ttf"].abs() / tail["basis"].abs()).median()),
        "tge_dominated_tail_days_reversing_next_day": int(reverses.sum()),
        "share_tail_sum_sq_of_total": float((tail["basis"] ** 2).sum() / (df["basis"] ** 2).sum()),
        "excess_kurtosis_basis_change": float(stats.kurtosis(df["basis"], bias=False)),
        "excess_kurtosis_tge_leg": float(stats.kurtosis(df["tge"], bias=False)),
        "excess_kurtosis_ttf_leg": float(stats.kurtosis(df["ttf"], bias=False)),
    }
    return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})


# --------------------------------------------------------------------------- FX timing


def cross_correlations(a: pd.Series, b: pd.Series, max_lag: int) -> pd.Series:
    """corr(a_t, b_{t+k}) for k = -max_lag..max_lag on the common sample."""
    df = pd.concat({"a": a, "b": b}, axis=1)
    return pd.Series({k: df["a"].corr(df["b"].shift(-k)) for k in range(-max_lag, max_lag + 1)})


def capture_ratio(cc: pd.Series) -> float:
    """Share of the lead-lag summed correlation (lags -1, 0, +1) that sits at lag 0.

    The sum c(-1) + c(0) + c(+1) is a correlation estimate that is robust to
    asynchronous sampling; a well-synchronised pair puts nearly all of it at
    lag 0, so the ratio approaches 1.
    """
    total = cc.loc[-1] + cc.loc[0] + cc.loc[1]
    return float(cc.loc[0] / total) if total != 0 else float("nan")


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def fx_timing(
    anchor: pd.Series,
    fx: dict[str, pd.Series],
    max_lag: int,
    reps: int,
    block: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lead-lag table and moving-block bootstrap of capture ratios.

    Args:
        anchor: The series the FX source must be synchronised with (TTF returns).
        fx: Candidate FX return series by name.

    Returns:
        (table, bootstrap) where ``table`` has one row per source with
        cross-correlations, the lag-summed correlation, the capture ratio, its
        bootstrap interval and the share of draws in which the source is best;
        ``bootstrap`` holds pairwise differences in capture ratio.
    """
    df = pd.concat({"anchor": anchor, **fx}, axis=1)
    # Build lagged copies once on the full calendar, then drop rows with gaps.
    cols = {}
    for name in fx:
        for k in (-1, 0, 1):
            cols[(name, k)] = df[name].shift(-k)
    lagged = pd.DataFrame(cols)
    lagged.columns = pd.MultiIndex.from_tuples(lagged.columns)
    full = pd.concat({"anchor": df[["anchor"]]}, axis=1).join(lagged).dropna()
    a = full[("anchor", "anchor")].to_numpy()

    def ratios(rows: np.ndarray) -> dict[str, float]:
        out = {}
        aa = a[rows]
        for name in fx:
            c = [np.corrcoef(aa, full[(name, k)].to_numpy()[rows])[0, 1] for k in (-1, 0, 1)]
            out[name] = c[1] / sum(c) if sum(c) != 0 else np.nan
        return out

    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([ratios(_block_indices(len(full), block, rng)) for _ in range(reps)])
    best = draws.idxmax(axis=1)

    rows = []
    for name, series in fx.items():
        cc = cross_correlations(anchor, series, max_lag)
        rows.append({
            "source": name,
            **{f"corr_lag_{k:+d}": cc.loc[k] for k in cc.index},
            "lag_summed_corr": cc.loc[-1] + cc.loc[0] + cc.loc[1],
            "capture_ratio": capture_ratio(cc),
            "capture_ci_low": draws[name].quantile(0.025),
            "capture_ci_high": draws[name].quantile(0.975),
            "share_draws_best": float((best == name).mean()),
        })
    table = pd.DataFrame(rows)
    pairs = []
    names = list(fx)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = draws[x] - draws[y]
            pairs.append({"comparison": f"{x} minus {y}", "mean_diff": d.mean(), "ci_low": d.quantile(0.025),
                          "ci_high": d.quantile(0.975), "share_positive": float((d > 0).mean())})
    return table, pd.DataFrame(pairs)


def fx_pair_leadlag(fx: dict[str, pd.Series]) -> pd.DataFrame:
    """Lead-lag between FX sources: who is sampled earlier in the day.

    If A is sampled earlier than B, corr(A_t, B_{t-1}) > 0 and corr(A_{t-1}, B_t)
    is about zero. ``implied_gap_share`` = c(-1) / (c(-1) + c(0)) approximates the
    share of daily variance accumulated between the two sampling times.
    """
    rows = []
    names = list(fx)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            cc = cross_correlations(fx[x], fx[y], 1)
            later = y if cc.loc[-1] > cc.loc[1] else x
            earlier = x if later == y else y
            lead = max(cc.loc[-1], cc.loc[1])
            rows.append({"pair": f"{x} vs {y}", "corr_x_t_y_t-1": cc.loc[-1], "corr_x_t_y_t": cc.loc[0],
                         "corr_x_t_y_t+1": cc.loc[1], "sampled_earlier": earlier, "sampled_later": later,
                         "implied_gap_share": lead / (lead + cc.loc[0]),
                         # A lagged correlation above the same-day one means the earlier source's value dated t
                         # is sampled more than half a (variance-weighted) day before the later source's value.
                         "date_offset_suspected": bool(lead > cc.loc[0])})
    return pd.DataFrame(rows)


def robust_portmanteau(r: pd.Series, lags: list[int]) -> dict[int, tuple[float, float]]:
    """Heteroskedasticity-robust portmanteau test for zero autocorrelation.

    Uses ``rho_k = sum(e_t e_t-k) / sqrt(sum(e_t^2 e_t-k^2))``, which is
    asymptotically N(0, 1) under a martingale difference null that allows
    conditional heteroskedasticity (Lo and MacKinlay 1989 type variance), and
    ``Q*(m) = sum_k rho_k^2 ~ chi2(m)``. The standard Ljung-Box test assumes
    i.i.d. returns and over-rejects when volatility clusters.

    Returns:
        {m: (Q*, p-value)}.
    """
    e = (r - r.mean()).dropna().to_numpy()
    rho2 = []
    for k in range(1, max(lags) + 1):
        num = np.sum(e[k:] * e[:-k])
        den = np.sqrt(np.sum(e[k:] ** 2 * e[:-k] ** 2))
        rho2.append((num / den) ** 2)
    out = {}
    for m in lags:
        q = float(np.sum(rho2[:m]))
        out[m] = (q, float(stats.chi2.sf(q, m)))
    return out


# --------------------------------------------------------------------------- stationarity


def _kpss(x: pd.Series) -> tuple[float, float, str]:
    """KPSS with a flag when the p-value is a table bound (statsmodels interpolates within [0.01, 0.10])."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InterpolationWarning)
        stat, p, *_ = kpss(x, regression="c", nlags="auto")
    flag = ""
    if any(issubclass(m.category, InterpolationWarning) for m in w):
        flag = "<=0.01 (table bound)" if p <= 0.01 else ">=0.10 (table bound)"
    return float(stat), float(p), flag


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float) -> str:
    """Joint reading of ADF (H0 unit root) and KPSS (H0 stationary)."""
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "stationary"
    if not adf_rej and kpss_rej:
        return "unit root"
    if adf_rej and kpss_rej:
        return "conflicting (breaks or long memory)"
    return "inconclusive"


def stationarity_table(series: list[SeriesSpec], alpha: float) -> pd.DataFrame:
    rows = []
    for s in series:
        for kind, x, label in (("level", s.level, s.level_label), ("return", s.ret, s.ret_label)):
            x = x.dropna()
            adf_stat, adf_p, adf_lags, nobs, *_ = adfuller(x, autolag="AIC", regression="c")
            k_stat, k_p, k_flag = _kpss(x)
            rows.append({"series": s.name, "transform": kind, "description": label, "n": int(nobs),
                         "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": int(adf_lags),
                         "kpss_stat": k_stat, "kpss_p": k_p, "kpss_p_note": k_flag,
                         "verdict": stationarity_verdict(adf_p, k_p, alpha)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- serial dependence


def serial_dependence_table(series: list[SeriesSpec], lags: list[int], arch_lags: int) -> pd.DataFrame:
    """Ljung-Box on returns and squared returns, ARCH-LM on demeaned returns."""
    rows = []
    for s in series:
        r = s.ret.dropna()
        e = r - r.mean()
        lb_r = acorr_ljungbox(r, lags=lags, return_df=True)
        lb_r2 = acorr_ljungbox(e**2, lags=lags, return_df=True)
        lm, lm_p, _, _ = het_arch(e, nlags=arch_lags)
        rob = robust_portmanteau(r, lags)
        row = {"series": s.name, "n": len(r)}
        for L in lags:
            row[f"lb_ret_p_{L}"] = float(lb_r.loc[L, "lb_pvalue"])
        for L in lags:
            row[f"robust_q_ret_p_{L}"] = rob[L][1]
        for L in lags:
            row[f"lb_sq_p_{L}"] = float(lb_r2.loc[L, "lb_pvalue"])
        row[f"arch_lm_stat_{arch_lags}"] = float(lm)
        row[f"arch_lm_p_{arch_lags}"] = float(lm_p)
        row["acf1_ret"] = float(r.autocorr(1))
        row["acf1_sq"] = float((e**2).autocorr(1))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- distribution


def distribution_row(r: pd.Series) -> dict[str, float]:
    """Moments, normality test and tail frequencies of one return series."""
    r = r.dropna()
    z = (r - r.mean()) / r.std()
    jb = stats.jarque_bera(r)
    return {
        "n": len(r),
        "mean": r.mean(),
        "sd_daily": r.std(),
        "vol_annualised": r.std() * np.sqrt(252),
        "skew": stats.skew(r, bias=False),
        "excess_kurtosis": stats.kurtosis(r, fisher=True, bias=False),
        "jb_stat": float(jb.statistic),
        "jb_p": float(jb.pvalue),
        "min": r.min(),
        "max": r.max(),
        "share_abs_z_gt_3": float((z.abs() > 3).mean()),
        "normal_share_abs_z_gt_3": float(2 * stats.norm.sf(3)),
        "q01": r.quantile(0.01),
        "q01_normal": r.mean() + r.std() * stats.norm.ppf(0.01),
        "q99": r.quantile(0.99),
        "q99_normal": r.mean() + r.std() * stats.norm.ppf(0.99),
    }


def distribution_table(series: list[SeriesSpec]) -> pd.DataFrame:
    return pd.DataFrame([{"series": s.name, "unit": s.ret_label, **distribution_row(s.ret)} for s in series])


def subperiod_table(series: list[SeriesSpec], subperiods: dict[str, list]) -> pd.DataFrame:
    rows = []
    for s in series:
        for name, (start, end) in subperiods.items():
            r = s.ret.loc[start:end].dropna() if end else s.ret.loc[start:].dropna()
            d = distribution_row(r)
            rows.append({"series": s.name, "subperiod": name, "first": r.index[0].date(), "last": r.index[-1].date(),
                         **{k: d[k] for k in ("n", "vol_annualised", "skew", "excess_kurtosis", "share_abs_z_gt_3", "jb_p")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figure


def acf_figure(series: list[SeriesSpec], path, nlags: int = 20) -> None:
    """ACF of returns and squared returns with approximate 95% bands (statistics only, no price levels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf

    fig, axes = plt.subplots(len(series), 2, figsize=(10, 2.3 * len(series)), sharex=True)
    for i, s in enumerate(series):
        r = s.ret.dropna()
        band = 1.96 / np.sqrt(len(r))
        for j, (x, title) in enumerate(((r, "returns"), ((r - r.mean()) ** 2, "squared returns"))):
            ax = axes[i, j]
            vals = acf(x, nlags=nlags, fft=True)[1:]
            ax.bar(range(1, nlags + 1), vals, color="#3b6ea5" if j == 0 else "#c0504d", width=0.7)
            ax.axhline(band, color="grey", lw=0.8, ls="--")
            ax.axhline(-band, color="grey", lw=0.8, ls="--")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{s.name}: ACF of {title}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("lag (trading days)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- run


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    """Run all Stage 2 diagnostics and write outputs."""
    dcfg = cfg["diagnostics"]
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    fx = {k: panel[f"eurpln_{k}_ret"] for k in cfg["fx"]["sources"]}
    timing, timing_pairs = fx_timing(panel["ttf_risk_ret"], fx, int(dcfg["timing_lags"]),
                                     int(dcfg["bootstrap_reps"]), int(dcfg["bootstrap_block"]), cfg.seed)
    crisis = cfg["sample"]["subperiods"]["crisis"]
    sub = panel.loc[crisis[0]:crisis[1]]
    timing_crisis, _ = fx_timing(sub["ttf_risk_ret"], {k: sub[f"eurpln_{k}_ret"] for k in fx}, 1,
                                 int(dcfg["bootstrap_reps"]), int(dcfg["bootstrap_block"]), cfg.seed)
    timing_tge, _ = fx_timing(panel["tge_ret"], fx, 1, int(dcfg["bootstrap_reps"]), int(dcfg["bootstrap_block"]), cfg.seed)
    pair = fx_pair_leadlag(fx)
    # Robustness: re-date any source whose values look sampled a day early, and re-test.
    redated = dict(fx)
    for r in pair[pair["date_offset_suspected"]].itertuples(index=False):
        redated[f"{r.sampled_earlier}_redated"] = fx[r.sampled_earlier].shift(-1)
    timing_redated, _ = fx_timing(panel["ttf_risk_ret"], redated, 1, int(dcfg["bootstrap_reps"]),
                                  int(dcfg["bootstrap_block"]), cfg.seed)

    series = build_series(panel)
    alpha = float(dcfg["alpha"])
    tables = {
        "fx_timing": timing,
        "fx_timing_pairwise": timing_pairs,
        "fx_timing_crisis": timing_crisis,
        "fx_timing_tge_anchor": timing_tge,
        "fx_source_leadlag": pair,
        "fx_timing_redated": timing_redated,
        "stationarity": stationarity_table(series, alpha),
        "serial_dependence": serial_dependence_table(series, list(dcfg["ljung_box_lags"]), int(dcfg["arch_lm_lags"])),
        "distribution": distribution_table(series),
        "subperiods": subperiod_table(series, cfg["sample"]["subperiods"]),
        "basis_tail_attribution": basis_tail_attribution(panel),
    }
    out = cfg.path("outputs_dir")
    for name, t in tables.items():
        t.to_csv(out / f"stage2_{name}.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    acf_figure(series, fig_dir / "stage2_acf.png")

    from gasrisk.diagnostics_report import write_stage2_report

    write_stage2_report(cfg, tables, panel)
    return tables


def main() -> None:
    cfg = load_config()
    tables = run(cfg)
    t = tables["fx_timing"].set_index("source")
    print("FX timing capture ratio:", t["capture_ratio"].round(3).to_dict(), "| best share:", t["share_draws_best"].round(3).to_dict())
    print(f"Report: {cfg.root / cfg['paths']['outputs_dir'] / 'stage2_report.md'}")


if __name__ == "__main__":
    main()
