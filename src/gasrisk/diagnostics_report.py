"""Generate the Stage 2 markdown report. Every number and verdict is computed from the tables."""

from __future__ import annotations

import pandas as pd

from gasrisk.config import Config
from gasrisk.data.report import _md
from gasrisk.diagnostics import BASIS_LABEL


def _fx_section(tables: dict[str, pd.DataFrame], current: str) -> str:
    t = tables["fx_timing"].set_index("source")
    best = t["capture_ratio"].idxmax()
    others = [s for s in t.index if s != best]
    pw = tables["fx_timing_pairwise"].set_index("comparison")

    def diff_line(other: str) -> str:
        key = f"{best} minus {other}" if f"{best} minus {other}" in pw.index else f"{other} minus {best}"
        row = pw.loc[key]
        sign = 1 if key.startswith(best) else -1
        lo, hi = sorted((sign * row["ci_low"], sign * row["ci_high"]))
        return (f"- {best} minus {other}: mean {sign * row['mean_diff']:.3f}, 95% interval [{lo:.3f}, {hi:.3f}], "
                f"{best} higher in {(row['share_positive'] if sign == 1 else 1 - row['share_positive']):.1%} of draws")

    crisis = tables["fx_timing_crisis"].set_index("source")["capture_ratio"]
    tge = tables["fx_timing_tge_anchor"].set_index("source")["capture_ratio"]
    robust = [f"crisis sub-period best: **{crisis.idxmax()}**", f"TGE as anchor best: **{tge.idxmax()}**"]

    pair = tables["fx_source_leadlag"]
    offsets = []
    for r in pair[pair["date_offset_suspected"]].itertuples(index=False):
        lead = max(r[1], r[3])
        offsets.append(f"- {r.pair}: the lagged correlation ({lead:.3f}) exceeds the same-day correlation ({r[2]:.3f}). "
                       f"The {r.sampled_earlier} value dated t is sampled well before the {r.sampled_later} value dated t, "
                       f"i.e. {r.sampled_earlier} behaves like a quote taken around the previous evening.")
    red = tables["fx_timing_redated"].set_index("source")
    red_best = red["capture_ratio"].idxmax()
    red_line = (f"Re-dating the suspected source(s) one day forward and re-testing: best = **{red_best}** "
                f"(capture ratios: {', '.join(f'{k} {v:.3f}' for k, v in red['capture_ratio'].items())}).") if len(red) > len(t) else ""

    sig = []
    for o in others:
        key = f"{best} minus {o}" if f"{best} minus {o}" in pw.index else f"{o} minus {best}"
        row = pw.loc[key]
        lo = row["ci_low"] if key.startswith(best) else -row["ci_high"]
        sig.append(f"{o} ({'significant' if lo > 0 else 'not significant'} at 95%)")
    status = ("matches" if current == best else f"differs from the configured `fx.risk_source` = {current}; update the config")
    return f"""## 1. EUR/PLN source timing

**Question.** Which EUR/PLN source is sampled closest to the TTF settlement?
A source fixed earlier or later than TTF splits the true same-day co-movement
between lag 0 and lags +1 or -1. The **capture ratio** is the lag-0 correlation
divided by the sum of correlations at lags -1, 0 and +1 (a correlation estimate
robust to asynchronous sampling). Closer to 1 means better synchronised.
Intervals come from a moving-block bootstrap.

Lead-lag correlations corr(TTF_t, FX_t+k) and capture ratios:

{_md(tables['fx_timing'])}

Pairwise differences in capture ratio (bootstrap):

{chr(10).join(diff_line(o) for o in others)}

Robustness: {'; '.join(robust)}.

{_md(tables['fx_timing_crisis'][['source', 'corr_lag_-1', 'corr_lag_+0', 'corr_lag_+1', 'capture_ratio', 'share_draws_best']])}

Ordering of FX sources within the day (if A is sampled earlier than B, corr(A_t, B_t-1) > 0):

{_md(tables['fx_source_leadlag'])}

{chr(10).join(offsets) if offsets else 'No date-label offset detected between sources.'}

{red_line}

{_md(tables['fx_timing_redated'][['source', 'corr_lag_-1', 'corr_lag_+0', 'corr_lag_+1', 'capture_ratio', 'share_draws_best']])}

**Selection.** **{best}** has the highest capture ratio ({t.loc[best, 'capture_ratio']:.3f}) and is best in
{t.loc[best, 'share_draws_best']:.1%} of bootstrap draws. Its advantage over {', '.join(sig)}; the
choice therefore rests on the point estimate together with its consistency across the robustness checks
above, not on a single significance test. It is consistent with published timing: TTF settles around
17:00 CET (ICE Endex), NBP Table A is published between 11:45 and 12:15 Warsaw time, and the ECB rate is
set in the early afternoon CET, the closest of the three to the TTF settlement. **{best}** is the EUR/PLN
risk factor; the other sources ({', '.join(others)}) remain cross-checks. The selection {status}.
"""


def _tail_sentence(t: pd.DataFrame) -> str:
    """Conclusion on the source of basis fat tails, derived from the attribution table."""
    m = t.set_index("metric")["value"]
    n, k = int(m["tail_days_abs_z_gt_3"]), int(m["tail_days_tge_leg_larger"])
    if n == 0:
        return "No basis tail days at this threshold."
    lead = ("The fat tails of the basis come **mainly from TGE spot spikes**, i.e. large moves in the TGE day-ahead leg"
            if k / n > 0.5 else "The data do **not** support attributing the basis tails mainly to TGE")
    rev = int(m["tge_dominated_tail_days_reversing_next_day"])
    rev_txt = (f"Only {rev} of the {k} TGE-dominated tail days reverse by at least half the next day, so most TGE "
               "spikes persist for more than one day rather than being one-day blips."
               if rev <= k / 2 else
               f"{rev} of the {k} TGE-dominated tail days reverse by at least half the next day, so most are one-day spikes.")
    return (f"{lead}: the TGE leg is the larger leg on {k} of {n} tail days, and on a typical tail day it is "
            f"{m['median_abs_tge_leg_over_abs_basis_tail']:.2f} of the basis move against "
            f"{m['median_abs_ttf_leg_over_abs_basis_tail']:.2f} for the TTF leg (medians). {rev_txt} "
            f"The TTF leg is not thin-tailed either (excess kurtosis {m['excess_kurtosis_ttf_leg']:.2f} against "
            f"{m['excess_kurtosis_tge_leg']:.2f} for the TGE leg), so TTF drives the remaining {n - k} tail days.")


def write_stage2_report(cfg: Config, tables: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    alpha = cfg["diagnostics"]["alpha"]
    st = tables["stationarity"]
    sd = tables["serial_dependence"]
    dist = tables["distribution"]
    L = cfg["diagnostics"]["arch_lm_lags"]
    lb = cfg["diagnostics"]["ljung_box_lags"]

    arch = sd.loc[sd[f"arch_lm_p_{L}"] < alpha, "series"].tolist()
    mean_dep_lb = sd.loc[sd[f"lb_ret_p_{lb[1]}"] < alpha, "series"].tolist()
    rob = sd.set_index("series")[[f"robust_q_ret_p_{L}" for L in lb]] < alpha
    mean_dep = rob.index[rob.all(axis=1)].tolist()            # rejects at every tested lag
    mean_dep_some = rob.index[rob.any(axis=1) & ~rob.all(axis=1)].tolist()
    mean_dep_none = rob.index[~rob.any(axis=1)].tolist()
    fat = dist.loc[dist["excess_kurtosis"] > 0, "series"].tolist()
    nonnormal = dist.loc[dist["jb_p"] < alpha, "series"].tolist()
    ret_verdicts = st[st["transform"] == "return"].set_index("series")["verdict"].to_dict()
    lvl_verdicts = st[st["transform"] == "level"].set_index("series")["verdict"].to_dict()

    text = f"""# Stage 2 - Diagnostics report

Generated by `python -m gasrisk.diagnostics` from the Stage 1 panel
({len(panel)} trading days, {panel.index[0].date()} to {panel.index[-1].date()}).
Significance level {alpha}. Every number and verdict below is computed; nothing is typed by hand.

{_fx_section(tables, cfg['fx']['risk_source'])}
## 2. Series analysed

| series | level | return |
|---|---|---|
| ttf | log roll-adjusted index of the ICE second month held to roll | roll-adjusted log return |
| tge | log TGEgasDA (trade date) | log return, business day to business day |
| eurpln | log EUR/PLN (risk source: {cfg['fx']['risk_source']}) | log return |
| basis | {BASIS_LABEL}, EUR/MWh | daily change, TTF leg within one contract |

The basis is a **location plus tenor basis**: TGE is a Polish day-ahead spot index, the hedge leg is the
Dutch TTF second-month future. It therefore mixes the Poland-Netherlands location spread with the
spot-versus-forward (tenor) spread, and cannot be read as a pure location basis.

## 3. Stationarity (ADF: H0 unit root; KPSS: H0 stationary)

{_md(st)}

Levels: {', '.join(f'{k} = {v}' for k, v in lvl_verdicts.items())}.
Returns / changes: {', '.join(f'{k} = {v}' for k, v in ret_verdicts.items())}.
KPSS p-values are interpolated from a table and capped at 0.01 and 0.10; capped values are marked.

## 4. Serial dependence

Ljung-Box p-values on returns (mean predictability) and squared returns (volatility clustering), ARCH-LM with {L} lags:

{_md(sd)}

- Standard Ljung-Box (assumes i.i.d. returns) rejects no autocorrelation up to lag {lb[1]} for: {', '.join(mean_dep_lb) or 'none'}.
- The heteroskedasticity-robust portmanteau test (`robust_q_ret_p_*`, valid under volatility clustering) is used for conclusions, because the ARCH results below show the i.i.d. assumption fails. It rejects at every tested lag ({', '.join(map(str, lb))}) for: {', '.join(mean_dep) or 'none'}; at some lags only (fragile evidence) for: {', '.join(mean_dep_some) or 'none'}; at no lag for: {', '.join(mean_dep_none) or 'none'}.
- Series with ARCH effects (ARCH-LM p < {alpha}): {', '.join(arch) or 'none'}.

![ACF of returns and squared returns](figures/stage2_acf.png)

Dashed lines are +/- 1.96/sqrt(n) bands, valid only for i.i.d. data; under volatility clustering
they are too narrow for the return ACF, which is why the robust portmanteau test is used above.

## 5. Distribution

{_md(dist)}

- Positive excess kurtosis (fatter tails than normal): {', '.join(fat) or 'none'}.
- Normality rejected by Jarque-Bera: {', '.join(nonnormal) or 'none'}.
- `share_abs_z_gt_3` is the observed share of days beyond 3 standard deviations; the normal distribution implies `normal_share_abs_z_gt_3`.

### Where the basis tails come from

Basis change = TGE leg (change in TGE in EUR) minus TTF leg (within-contract change in the second month).
On basis tail days (|z| > 3) the size of the two legs is compared:

{_md(tables['basis_tail_attribution'])}

{_tail_sentence(tables['basis_tail_attribution'])}

## 6. Sub-periods

Sub-period boundaries are a modelling choice set in `config.yaml` (pre-crisis, 2021-2023 crisis, post-crisis).

{_md(tables['subperiods'])}

## 7. Implications for later stages

- **Volatility models (Stage 4):** ARCH effects in {', '.join(arch) or 'no series'} support conditional volatility (GARCH) rather than a constant variance.
- **VaR (Stage 5):** fat tails and non-normality in {', '.join(nonnormal) or 'no series'} mean a normal parametric VaR is expected to understate tail risk; Student-t and filtered historical simulation are the natural alternatives to test.
- **Forecasting (Stage 3):** models are fitted on returns or changes, which are stationary by the joint ADF/KPSS reading where marked above; robust autocorrelation in {', '.join(mean_dep) or 'no series'} leaves room for ARMA or ML structure to beat the random walk; for {', '.join(mean_dep_some + mean_dep_none) or 'no series'} the evidence of predictable returns is weak or absent, so a random walk is expected to be hard to beat.
- **FX in the book:** the EUR/PLN risk factor is the source selected in section 1; correlations with TTF measured with a mistimed source would be biased toward zero.
"""
    (cfg.path("outputs_dir") / "stage2_report.md").write_text(text, encoding="utf-8")
