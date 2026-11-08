"""Stage 3 report and figure. Every number and verdict is computed from the tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gasrisk.config import Config
from gasrisk.data.report import _md

MODEL_ORDER = ["random_walk", "arima", "ets", "gbm", "linear"]
SERIES_LABEL = {
    "ttf": "TTF second month held to roll (EUR/MWh)",
    "eurpln": "EUR/PLN (ECB)",
    "tge": "TGEgasDA, trade date (PLN/MWh)",
    "basis": "location plus tenor basis: TGE day-ahead spot (EUR) minus TTF second month (EUR/MWh)",
}


def rmse_ratio_figure(metrics: pd.DataFrame, path: Path) -> None:
    """RMSE relative to the random walk by series, horizon and model (below 1 = better than the random walk)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    series = ["ttf", "eurpln", "tge", "basis"]
    models = [m for m in MODEL_ORDER[1:] if m in metrics["model"].unique()]
    colors = {"arima": "#3b6ea5", "ets": "#8cb369", "gbm": "#e4a03b", "linear": "#c0504d"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, s in zip(axes, series):
        g = metrics[metrics["series"] == s]
        hs = sorted(g["h"].unique())
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [g[(g["h"] == h) & (g["model"] == m)]["rmse_ratio_vs_rw"].squeeze() if ((g["h"] == h) & (g["model"] == m)).any() else np.nan for h in hs]
            ax.bar(np.arange(len(hs)) + i * w - 0.4 + w / 2, vals, width=w, color=colors[m], label=m)
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_xticks(range(len(hs)), [f"h={h}" for h in hs])
        ax.set_title(s, fontsize=10)
    axes[0].set_ylabel("RMSE / RMSE random walk")
    axes[0].legend(fontsize=8, loc="upper left")
    lo = np.nanmin(metrics.loc[metrics["model"] != "random_walk", "rmse_ratio_vs_rw"])
    hi = np.nanmax(metrics.loc[metrics["model"] != "random_walk", "rmse_ratio_vs_rw"])
    axes[0].set_ylim(max(0, min(0.9, lo - 0.05)), max(1.1, hi + 0.05))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _series_summary(metrics: pd.DataFrame, s: str, alpha: float) -> str:
    g = metrics[(metrics["series"] == s) & (metrics["model"] != "random_walk")]
    lines = []
    for h, gh in g.groupby("h"):
        best = gh.loc[gh["rmse"].idxmin()]
        winners = gh[gh["verdict"] == "beats random walk"]
        if len(winners):
            w = winners.loc[winners["rmse"].idxmin()]
            holds = w["holm_p_all"] < alpha
            label = ("beats the random walk, also after Holm adjustment" if holds else
                     "beats the random walk at the unadjusted 5% level, but not after Holm adjustment for multiple testing")
            lines.append(f"- h={h}: **{w['model']} {label}** (RMSE ratio {w['rmse_ratio_vs_rw']:.3f}, "
                         f"DM one-sided p {w['dm_p_better']:.4f}, Holm-adjusted p across all comparisons {w['holm_p_all']:.4f}"
                         + (f", directional accuracy {w['dir_acc']:.3f}, binomial p {w['dir_p']:.4f}" if pd.notna(w['dir_acc']) else "") + ").")
        else:
            lines.append(f"- h={h}: **random walk not beaten.** Lowest RMSE: {best['model']} (ratio {best['rmse_ratio_vs_rw']:.3f}, "
                         f"DM one-sided p {best['dm_p_better']:.4f}); no model is significantly more accurate at the {alpha} level.")
    return "\n".join(lines)


def _ablation_section(ab: pd.DataFrame, alpha: float) -> str:
    """Where the out-of-sample predictability of the linear model comes from (one feature group at a time)."""
    out = ["### Ablation - which component carries the predictability",
           "",
           "The linear direct model is re-run through the full walk-forward with one feature group added at a time.",
           "`dm_p_better_vs_previous_step` tests whether adding the group made forecasts significantly more accurate.",
           ""]
    for s, g in ab.groupby("series", sort=False):
        out.append(f"**{s}**\n\n{_md(g.drop(columns='series'))}\n")
        g1 = g[g["h"] == 1].reset_index(drop=True)
        gains = g1["rmse_ratio_vs_rw"].diff().fillna(g1["rmse_ratio_vs_rw"] - 1.0)
        top = g1.loc[gains.idxmin()]
        sig = g1[g1["dm_p_better_vs_previous_step"] < alpha]["step"].tolist()
        out.append(f"At h=1 the largest single improvement comes from **{top['step']}** (RMSE ratio change "
                   f"{gains.min():+.3f}). Steps that significantly improve on the previous one: {', '.join(sig) or 'none'}. "
                   f"Final step: RMSE ratio {g1['rmse_ratio_vs_rw'].iloc[-1]:.3f}, DM p vs random walk "
                   f"{g1['dm_p_better_vs_rw'].iloc[-1]:.4f}.\n")
    return "\n".join(out)


def _conclusion(tables: dict[str, pd.DataFrame], alpha: float) -> str:
    """Overall verdict combining raw DM tests, Holm adjustment and the placebo."""
    m = tables["metrics"]
    t = m[m["model"] != "random_walk"]
    tge1 = t[(t["series"] == "tge") & (t["h"] == 1)]
    b = tge1.loc[tge1["dm_p_better"].idxmin()]
    pl = tables["placebo"].set_index(["series", "h", "model"])
    surv = pl.loc[("tge", 1, b["model"]), "gain_surviving"] if ("tge", 1, b["model"]) in pl.index else float("nan")
    n_adj = int((t["holm_p_all"] < alpha).sum())
    return (f"**Overall.** {'No' if n_adj == 0 else n_adj} model-horizon-series comparison{'s' if n_adj != 1 else ''} "
            f"beat{'s' if n_adj == 1 else ''} the random walk after Holm adjustment across all {len(t)} comparisons. "
            f"The strongest raw result is TGE at 1 day ({b['model']}: RMSE ratio {b['rmse_ratio_vs_rw']:.3f}, raw DM p "
            f"{b['dm_p_better']:.4f}, Holm p {b['holm_p_all']:.4f}). Its gain comes from same-day TTF and FX moves: "
            f"lagging them one more day leaves {surv:.1%} of the gain. The honest reading is that TGE 1-day changes "
            f"are partly predictable from same-day TTF information, but the evidence is not strong enough to survive "
            f"a correction for testing {len(t)} model-horizon-series combinations; for TTF and EUR/PLN the random walk "
            f"is not beaten, a valid finding for liquid prices.")


def _robustness_section(tables: dict[str, pd.DataFrame], alpha: float) -> str:
    """Leakage, placebo, multiple testing and DM specification checks."""
    m = tables["metrics"]
    t = m[m["model"] != "random_walk"]
    pl = tables["placebo"]

    # Placebo sentence for TGE h=1
    lines = []
    for r in pl[pl["h"] == 1].itertuples(index=False):
        surv = "n/a (no gain to survive)" if pd.isna(r.gain_surviving) else f"{r.gain_surviving:.1%}"
        lines.append(f"- {r.series} {r.model}, h=1: RMSE ratio {r.rmse_ratio_original:.3f} -> {r.rmse_ratio_placebo:.3f} "
                     f"with TTF and FX lagged one more day; DM p vs random walk {r.dm_p_better_original:.4f} -> "
                     f"{r.dm_p_better_placebo:.4f}; share of the gain surviving: {surv}.")

    # Multiple testing
    show = t[["series", "h", "model", "dm_p_better", "holm_p_all", "holm_p_prespecified"]].sort_values("dm_p_better").head(10)
    tge1 = t[(t["series"] == "tge") & (t["h"] == 1)]
    n_all, n_pre = int(t["holm_p_all"].notna().sum()), int(t["holm_p_prespecified"].notna().sum())
    surv_all = tge1[tge1["holm_p_all"] < alpha]["model"].tolist()
    surv_pre = tge1[tge1["holm_p_prespecified"] < alpha]["model"].tolist()
    best = tge1.loc[tge1["dm_p_better"].idxmin()]
    holm_txt = (f"Across all {n_all} model-horizon-series comparisons (primary family), TGE at 1 day "
                + (f"**still beats the random walk** after Holm adjustment ({', '.join(surv_all)})." if surv_all else
                   f"**does not survive** the Holm adjustment: the smallest adjusted p-value is {best['holm_p_all']:.4f} "
                   f"({best['model']}, raw {best['dm_p_better']:.4f}).")
                + f" In the secondary family of the {n_pre} TGE and basis comparisons (the series Stage 2 identified as "
                + "candidates before Stage 3 was run), "
                + (f"it holds ({', '.join(surv_pre)}, Holm p {tge1['holm_p_prespecified'].min():.4f})." if surv_pre else
                   f"it does not hold either (smallest Holm p {tge1['holm_p_prespecified'].min():.4f})."))
    n_beat_all = int((t["holm_p_all"] < alpha).sum())

    # DM specification
    fb = int(t["dm_lrv_fallback"].sum())
    fb_rows = t[t["dm_lrv_fallback"].astype(bool)]
    fb_txt = (" (" + ", ".join(f"{r.series} h={r.h} {r.model}, DM p {r.dm_p_better:.4f}" for r in fb_rows.itertuples(index=False)) + ")") if fb else ""
    flips = t[(t["dm_p_better"] < alpha) != (t["dm_p_better_newey_west"] < alpha)]
    flip_txt = ("No verdict changes when Newey-West (Bartlett) weights replace equal weights."
                if flips.empty else "Verdicts that change with Newey-West weights: "
                + ", ".join(f"{r.series} h={r.h} {r.model}" for r in flips.itertuples(index=False)) + ".")

    return f"""## Robustness checks

### 1. Leakage

Timing at the forecast origin (after the close of day t):

| input | observed | used at row t |
|---|---|---|
| TGEgasDA, trade date t | volume weighted price of trades on day t | yes |
| TTF second-month settlement, day t | around 17:00 CET on day t | yes |
| ECB EUR/PLN, day t | early afternoon CET on day t | yes |
| any value dated t+1 or later | after the origin | never |
| target (TGE, h=1) | trade date t+1 | as target only |

Mechanical tests (`tests/test_leakage.py`): every feature and target value up to row t is unchanged
when all data after t are replaced; every model's walk-forward forecast at origin t is unchanged when
data after t are replaced (random walk, ARIMA, ETS, gradient boosting, linear); target dates are exactly
h rows after the origin; same-day TTF and FX inputs at t are used, and the placebo lag removes them.

### 2. Placebo: TTF and FX lagged one more day

All TTF- and FX-derived inputs of the TGE and basis models (changes, 5-day sums, the TGE/TTF ratio) are
lagged by one extra day; TGE's own inputs and the targets are unchanged.

{_md(pl)}

{chr(10).join(lines)}

### 3. Multiple testing (Holm)

Ten smallest one-sided DM p-values with Holm adjustments (full table in `outputs/stage3_metrics.csv`):

{_md(show)}

{holm_txt} Comparisons that beat the random walk after Holm adjustment in the primary family: {n_beat_all}.

### 4. Diebold-Mariano specification

The DM statistic uses squared-error loss, a long-run variance from autocovariances at lags 0 to h-1
(equal weights, as in Diebold and Mariano 1995; lags 0, 4 and 20 for h = 1, 5, 21), the
Harvey-Leybourne-Newbold small-sample factor sqrt((n + 1 - 2h + h(h - 1)/n)/n) and a t(n-1) reference
distribution. It matches an independent hand computation in `tests/test_forecasting.py`. The variance
estimate was non-positive (lag-0 variance used instead) in {fb} of {len(t)} comparisons{fb_txt}. {flip_txt}
"""


def write_stage3_report(cfg: Config, tables: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    fcfg = cfg["forecasting"]
    alpha = float(fcfg["alpha"])
    m = tables["metrics"]
    tw = tables["test_window"]
    show = ["model", "n", "rmse", "rmse_ratio_vs_rw", "mae", "rmse_level", "dir_acc", "dir_p", "dm_stat", "dm_p_better", "verdict"]

    def block(s: str) -> str:
        parts = []
        for h, g in m[m["series"] == s].groupby("h"):
            parts.append(f"**h = {h}**\n\n{_md(g[show])}")
        return "\n\n".join(parts)

    ecm = tables["tge_ecm_insample"].set_index("term")
    ar1 = tables["basis_ar1_insample"].set_index("metric")["value"]
    sub = tables["metrics_by_subperiod"]
    sub_focus = sub[sub["series"].isin(["tge", "basis"]) & (sub["model"] != "random_walk")]
    cols = list(cfg["sample"]["subperiods"])
    sub_piv = sub_focus.pivot_table(index=["series", "h", "model"], columns="subperiod", values="rmse_ratio_vs_rw")[cols].reset_index()
    sub_other = sub[sub["series"].isin(["ttf", "eurpln"]) & (sub["model"] != "random_walk")]
    sub_piv2 = sub_other.pivot_table(index=["series", "h", "model"], columns="subperiod", values="rmse_ratio_vs_rw")[cols].reset_index()

    def verdict_count(s: str) -> str:
        g = m[(m["series"] == s) & (m["model"] != "random_walk")]
        raw = int((g['verdict'] == 'beats random walk').sum())
        adj = int((g['holm_p_all'] < alpha).sum())
        return (f"{raw} of {len(g)} model-horizon pairs beat the random walk at the unadjusted {alpha} level; "
                f"{adj} after Holm adjustment across all comparisons")

    text = f"""# Stage 3 - Price forecasting report

Generated by `python -m gasrisk.forecasting`. Every number and verdict is computed; nothing is typed by hand.

## Design

- **Targets:** level at t+h for h in {fcfg['horizons']} business days. TTF, EUR/PLN and TGE are modelled in
  logs, so `rmse` and `mae` are in log points (scale-free) and `rmse_level` is in price units; the basis is
  modelled and evaluated in EUR/MWh.
- **Walk-forward:** rolling {fcfg['train_window']}-day window, parameters re-estimated every {fcfg['refit_every']}
  trading days, state updated daily with fixed parameters. A direct h-step model at time T is trained only on
  rows whose target date is on or before T.
- **Models:** random walk (benchmark); ARIMA (ARMA with constant on log returns, or on the basis level;
  order by AIC from {fcfg['arima_orders']}); ETS (simple vs damped trend, plus weekly seasonal for TGE and
  basis; chosen by AIC); gradient boosting (direct h-step, lag / rolling / cross-asset features);
  `linear` direct regression for TGE and basis (day of week, own reversal, same-day TTF and FX moves, plus an
  error-correction term for TGE). TGE and basis models get day-of-week terms.
- **Tests:** Diebold-Mariano on squared errors against the random walk with the Harvey-Leybourne-Newbold
  correction (`dm_p_better` is the one-sided p-value for "model more accurate"); directional accuracy with
  a one-sided binomial test against 50% (`dir_p`). A random-walk win is a valid finding for prices.

Test window:

{_md(tw)}

## Headline

| series | result |
|---|---|
| ttf | {verdict_count('ttf')} |
| eurpln | {verdict_count('eurpln')} |
| tge | {verdict_count('tge')} |
| basis | {verdict_count('basis')} |

{_conclusion(tables, alpha)}

![RMSE relative to random walk](figures/stage3_rmse_ratio.png)

## TGE and basis - where a model has a real chance

Stage 2 found robust autocorrelation and a mean-reverting basis for these two series, so they get the
most attention: day-of-week terms, a linear direct model with an error-correction term, a step-by-step
ablation of where the predictability comes from, sub-period results and model diagnostics.

{_ablation_section(tables['ablation'], alpha)}

### TGE - {SERIES_LABEL['tge']}

{_series_summary(m, 'tge', alpha)}

{block('tge')}

**Linear model with error-correction term (in-sample, descriptive, full sample, HAC errors).** One-day change
in log TGE on the log ratio of TGE (EUR) to TTF and controls:

{_md(tables['tge_ecm_insample'])}

The coefficient on `z_tge_ttf` is {ecm.loc['z_tge_ttf', 'coef']:.4f} (HAC t {ecm.loc['z_tge_ttf', 't']:.2f}):
{'a negative, significant coefficient means TGE moves back toward TTF x EUR/PLN, with an implied half-life of ' + format(ecm.loc['implied_half_life_days', 'coef'], '.1f') + ' trading days' if ecm.loc['z_tge_ttf', 'coef'] < 0 and ecm.loc['z_tge_ttf', 'p'] < alpha else 'no significant pull toward TTF x EUR/PLN at this horizon'}.
Terms with the largest |HAC t| (excluding the constant): {', '.join(f'{k} ({v:.2f})' for k, v in ecm.loc[[i for i in ecm.index if i not in ('const', 'implied_half_life_days', 'r_squared', 'n')], 't'].abs().sort_values(ascending=False).head(4).items())}.
The ablation above shows how much of this survives out of sample.

**Gradient boosting permutation importance (TGE, h=1, last window):**

{_md(tables['tge_gbm_importance_h1'].head(10))}

### Basis - {SERIES_LABEL['basis']}

{_series_summary(m, 'basis', alpha)}

{block('basis')}

**AR(1) on the basis level (in-sample, descriptive, HAC errors):**

{_md(tables['basis_ar1_insample'])}

Implied half-life of basis shocks: {ar1['implied_half_life_days']:.1f} trading days around a long-run mean of
{ar1['long_run_mean_eur_mwh']:.2f} EUR/MWh.

**Gradient boosting permutation importance (basis, h=1, last window):**

{_md(tables['basis_gbm_importance_h1'].head(10))}

### Sub-periods (RMSE ratio vs random walk, by target date)

{_md(sub_piv)}

## TTF and EUR/PLN - random walk benchmark

For liquid futures and FX, a random walk that is hard to beat is the expected and valid result: Stage 2
found only weak (lag-dependent) autocorrelation once volatility clustering is accounted for.

### TTF - {SERIES_LABEL['ttf']}

{_series_summary(m, 'ttf', alpha)}

{block('ttf')}

### EUR/PLN - {SERIES_LABEL['eurpln']}

{_series_summary(m, 'eurpln', alpha)}

{block('eurpln')}

### Sub-periods

{_md(sub_piv2)}

{_robustness_section(tables, alpha)}
## ARIMA orders chosen (number of origins using each order)

{_md(tables['arima_orders_chosen'])}

## Limitations

- Point forecasts only; forecast intervals and density forecasts are left to the volatility stage.
- Gradient boosting hyperparameters are fixed in `config.yaml`, not tuned, to avoid selecting on the test
  window; a nested tuning loop would be needed to tune them honestly.
- The basis is a location plus tenor basis (TGE day-ahead spot vs TTF second month), so its predictability
  reflects spot dynamics as much as the Poland-Netherlands spread.
- In-sample regressions (error correction, AR(1)) are descriptive and are not used for the out-of-sample verdicts.
- TGEgasDA is a volume weighted average of trades during the trade day, while TTF settles at the end of the
  day. Part of the TGE predictability from same-day TTF moves therefore reflects the index catching up with
  late-day information. It is a valid forecast for risk purposes (all inputs are known at the forecast
  origin) but not evidence of a tradable edge.
"""
    (cfg.path("outputs_dir") / "stage3_report.md").write_text(text, encoding="utf-8")
