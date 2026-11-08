"""Stage 7: limits, daily utilisation, breach detection and breach analysis (synthetic book).

Usage::

    python -m gasrisk.limits

* Official VaR method chosen by rule from the Stage 6 backtest.
* Limits (synthetic, ``config.yaml``): 99% VaR, open gas exposure value (unhedged volume x TGE in PLN) and
  FX net open position (EUR value of TGE purchases not covered by the forward).
* Utilisation = measure / limit; amber from ``amber_threshold``, breach above 100%.
* Breach analysis: consecutive breach days form an episode. For VaR episodes, the official VaR is split into
  risk-factor components by Euler allocation (linear sensitivities, the official method's GARCH volatilities and
  copula correlations, scaled so the components sum to the official VaR); the factor whose component rose most
  since ``driver_lookback_days`` before the episode is the driver. For exposure limits the driver is the
  TGE price and EUR/PLN move over the same lookback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config

FACTOR_LABEL = {"tge_eur": "TGE price (EUR)", "ttf": "TTF price", "fx": "EUR/PLN"}


def choose_official(backtest: pd.DataFrame, traffic: pd.DataFrame) -> tuple[str, str]:
    """Methods passing conditional coverage at every level; among them the highest share of green windows."""
    ok = backtest.groupby("method")["pass_cc"].all()
    cands = ok[ok].index.tolist()
    if not cands:
        best = traffic.sort_values("share_windows_green", ascending=False).iloc[0]["method"]
        return best, "no method passes conditional coverage at all levels; highest green share used (flagged)"
    t = traffic[traffic["method"].isin(cands)].sort_values("share_windows_green", ascending=False)
    return t.iloc[0]["method"], (f"passes conditional coverage at all levels ({', '.join(cands)}); highest share of "
                                 f"green 250-day windows among them ({t.iloc[0]['share_windows_green']:.4f})")


def sensitivities(row: pd.Series, book: dict) -> np.ndarray:
    """Linear P&L sensitivity (PLN per unit log return) to TGE in EUR, TTF and EUR/PLN."""
    q = float(book["q_open_mwh"])
    g, t, f = row["level_tge_eur"], row["level_ttf"], row["level_fx"]
    n_ttf, n_fx = float(book["ttf_hedge_ratio"]) * q, float(book["fx_hedge_ratio"]) * q * g
    return np.array([-q * g * f, n_ttf * t * f, -q * g * f + n_fx * f])


def euler_components(s: np.ndarray, sig: np.ndarray, corr: np.ndarray, total_var: float) -> np.ndarray:
    """Euler risk contributions s_k (Sigma s)_k / sigma_p, rescaled to sum to ``total_var``."""
    cov = np.outer(sig, sig) * corr
    sp = np.sqrt(s @ cov @ s)
    comp = s * (cov @ s) / sp
    return comp / comp.sum() * total_var


def _corr(row: pd.Series) -> np.ndarray:
    return np.array([[1, row["corr_tge_ttf"], row["corr_tge_fx"]],
                     [row["corr_tge_ttf"], 1, row["corr_ttf_fx"]],
                     [row["corr_tge_fx"], row["corr_ttf_fx"], 1]])


def utilisation(fc: pd.DataFrame, book: dict, lim: dict) -> pd.DataFrame:
    q = float(book["q_open_mwh"])
    d = fc.set_index("origin").sort_index()
    out = pd.DataFrame(index=d.index)
    out["var_99"] = d["var_99"]
    out["open_gas_exposure_pln"] = q * (1 - float(book["ttf_hedge_ratio"])) * d["level_tge_eur"] * d["level_fx"]
    out["fx_net_open_eur"] = q * (1 - float(book["fx_hedge_ratio"])) * d["level_tge_eur"]
    out["util_var_99"] = out["var_99"] / float(lim["var99_limit_pln"])
    out["util_open_gas"] = out["open_gas_exposure_pln"] / float(lim["open_gas_exposure_limit_pln"])
    out["util_fx_net_open"] = out["fx_net_open_eur"] / float(lim["fx_net_open_limit_eur"])
    comps = np.array([euler_components(sensitivities(r, book), np.array([r["sig_tge_eur"], r["sig_ttf"], r["sig_fx"]]),
                                       _corr(r), r["var_99"]) for _, r in d.iterrows()])
    for i, f in enumerate(["tge_eur", "ttf", "fx"]):
        out[f"comp_var99_{f}"] = comps[:, i]
    for f in ["tge_eur", "ttf", "fx"]:
        out[f"sig_{f}"] = d[f"sig_{f}"]
        out[f"level_{f}"] = d[f"level_{f}"]
    return out


def episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Runs of consecutive True values: (start, end, length)."""
    runs, start, prev, n = [], None, None, 0
    for d, v in flag.items():
        if v and start is None:
            start, n = d, 0
        if v:
            prev, n = d, n + 1
        if not v and start is not None:
            runs.append((start, prev, n))
            start = None
    if start is not None:
        runs.append((start, prev, n))
    return runs


def breach_analysis(u: pd.DataFrame, lookback: int, amber: float = 0.8) -> pd.DataFrame:
    rows = []
    idx = list(u.index)
    for limit, col in (("var_99", "util_var_99"), ("open_gas", "util_open_gas"), ("fx_net_open", "util_fx_net_open")):
        for start, end, n in episodes(u[col] > 1.0):
            i0 = idx.index(start)
            ref = idx[max(0, i0 - lookback)]
            r = {"limit": limit, "start": start.date(), "end": end.date(), "days": n,
                 "peak_utilisation": u.loc[start:end, col].max(), "reference_date": ref.date()}
            if limit == "var_99":
                ch = {f: u.at[start, f"comp_var99_{f}"] - u.at[ref, f"comp_var99_{f}"] for f in ("tge_eur", "ttf", "fx")}
                shares = {f: u.at[start, f"comp_var99_{f}"] / u.at[start, "var_99"] for f in ("tge_eur", "ttf", "fx")}
                r.update({f"share_{f}_at_start": shares[f] for f in shares})
                r.update({f"vol_ratio_{f}": u.at[start, f"sig_{f}"] / u.at[ref, f"sig_{f}"] for f in ("tge_eur", "ttf", "fx")})
                main = max(shares, key=shares.get)
                inc = max(ch, key=ch.get)
                change_txt = (f"largest increase {FACTOR_LABEL[inc]} ({ch[inc]:+,.0f} PLN since reference date)" if ch[inc] > 0
                              else "no component increased since the reference date (VaR was already elevated)")
                r["main_contributor"] = FACTOR_LABEL[main]
                r["driver"] = f"{FACTOR_LABEL[main]} (share {shares[main]:.0%} at start); {change_txt}"
            else:
                # Exposure utilisation is proportional to TGE_EUR (x FX for the PLN gas limit), so the log change
                # since utilisation was last below amber splits exactly into a TGE price part and an FX part.
                below = u.loc[:start, col]
                below = below[below < amber]
                ref = below.index[-1] if len(below) else idx[max(0, i0 - lookback)]
                r["reference_date"] = ref.date()
                dg = np.log(u.at[start, "level_tge_eur"] / u.at[ref, "level_tge_eur"])
                df_ = np.log(u.at[start, "level_fx"] / u.at[ref, "level_fx"]) if limit == "open_gas" else 0.0
                tot = dg + df_
                r["share_tge_eur_at_start"] = dg / tot if tot else np.nan
                r["share_fx_at_start"] = df_ / tot if tot else np.nan
                r["driver"] = (f"TGE price (EUR) ({dg / tot:.0%} of the utilisation increase since it was last below "
                               f"amber; EUR/PLN {df_ / tot:.0%})") if tot else "no change since reference"
            rows.append(r)
    return pd.DataFrame(rows)


def summary(u: pd.DataFrame, amber: float) -> pd.DataFrame:
    rows = []
    for limit, col in (("var_99", "util_var_99"), ("open_gas", "util_open_gas"), ("fx_net_open", "util_fx_net_open")):
        s = u[col]
        rows.append({"limit": limit, "days": len(s), "mean_utilisation": s.mean(), "max_utilisation": s.max(),
                     "days_amber": int(((s >= amber) & (s <= 1)).sum()), "days_breach": int((s > 1).sum()),
                     "breach_episodes": len(episodes(s > 1)), "latest_utilisation": s.iloc[-1],
                     "latest_status": "breach" if s.iloc[-1] > 1 else ("amber" if s.iloc[-1] >= amber else "green")})
    return pd.DataFrame(rows)


def utilisation_figure(u: pd.DataFrame, amber: float, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    for col, lab in (("util_var_99", "99% VaR"), ("util_open_gas", "open gas exposure"), ("util_fx_net_open", "FX net open")):
        ax.plot(u.index, u[col] * 100, lw=0.9, label=lab)
    ax.axhline(100, color="red", ls="--", lw=0.8)
    ax.axhline(amber * 100, color="orange", ls="--", lw=0.8)
    ax.set_ylabel("utilisation (%)")
    ax.set_title("SYNTHETIC book: daily limit utilisation (amber and breach thresholds dashed)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    lim, book = dict(cfg["limits"]), dict(cfg["book"])
    out = cfg.path("outputs_dir")
    bt, tl = pd.read_csv(out / "stage6_backtest.csv"), pd.read_csv(out / "stage6_traffic_light.csv")
    method, reason = choose_official(bt, tl) if lim["official_method"] == "auto" else (lim["official_method"], "set in config")
    fc = pd.read_csv(cfg.path("clean_dir") / "var_forecasts.csv", parse_dates=["origin", "target"])
    u = utilisation(fc[fc["method"] == method], book, lim)
    u.drop(columns=[c for c in u.columns if c.startswith("level_")]).to_csv(cfg.path("clean_dir") / "limit_utilisation.csv")
    latest = u.iloc[-1]
    today = pd.DataFrame([
        {"limit": "99% VaR (PLN)", "measure": latest["var_99"], "limit_value": lim["var99_limit_pln"], "utilisation": latest["util_var_99"]},
        {"limit": "open gas exposure (PLN)", "measure": latest["open_gas_exposure_pln"], "limit_value": lim["open_gas_exposure_limit_pln"], "utilisation": latest["util_open_gas"]},
        {"limit": "FX net open position (EUR)", "measure": latest["fx_net_open_eur"], "limit_value": lim["fx_net_open_limit_eur"], "utilisation": latest["util_fx_net_open"]},
    ])
    comp_today = pd.DataFrame([{"factor": FACTOR_LABEL[f], "component_var99_pln": latest[f"comp_var99_{f}"],
                                "share": latest[f"comp_var99_{f}"] / latest["var_99"]} for f in ("tge_eur", "ttf", "fx")])
    tables = {"official_method": pd.DataFrame([{"method": method, "reason": reason}]),
              "limits_today": today, "var_components_today": comp_today,
              "utilisation_summary": summary(u, float(lim["amber_threshold"])),
              "breaches": breach_analysis(u, int(lim["driver_lookback_days"]), float(lim["amber_threshold"]))}
    for k, t in tables.items():
        t.to_csv(out / f"stage7_{k}.csv", index=False)
    (out / "figures").mkdir(exist_ok=True)
    utilisation_figure(u, float(lim["amber_threshold"]), out / "figures" / "stage7_utilisation.png")
    from gasrisk.limits_report import write_stage7_report

    write_stage7_report(cfg, tables, u.index[-1])
    return tables


def main() -> None:
    t = run(load_config())
    for k in ("official_method", "limits_today", "utilisation_summary"):
        print(t[k].to_string(index=False))
    print(t["breaches"][["limit", "start", "end", "days", "peak_utilisation", "driver"]].to_string(index=False))


if __name__ == "__main__":
    main()
