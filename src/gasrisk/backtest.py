"""Stage 6: VaR backtesting of the Stage 5 forecasts (synthetic book).

Usage::

    python -m gasrisk.backtest

* Kupiec (1995) proportion-of-failures test: is the exception rate equal to 1 - level?
* Christoffersen (1998) independence test (first-order Markov) and conditional coverage
  (POF + independence, chi-square with 2 df).
* Basel traffic light on 99% VaR over the latest 250 days (green 0-4, yellow 5-9, red 10+)
  and the share of rolling 250-day windows in each zone.
* Descriptive ES check: average loss / ES on days the 99% VaR is exceeded (about 1 if ES is right).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from gasrisk.config import Config, load_config


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(exceptions: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec POF likelihood ratio and p-value (chi-square, 1 df). ``p`` is the expected exception rate."""
    n, x = len(exceptions), int(np.sum(exceptions))
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(exceptions: np.ndarray, p: float) -> dict[str, float]:
    """Christoffersen independence and conditional coverage tests."""
    e = np.asarray(exceptions).astype(int)
    a, b = e[:-1], e[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr_ind = -2 * (ll_null - ll_alt)
    lr_pof, _ = kupiec(e, p)
    lr_cc = lr_pof + lr_ind
    return {"n11": n11, "pi0": pi0, "pi1": pi1, "lr_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def traffic_light(n_exceptions: int) -> str:
    """Basel zone for 99% VaR exceptions in a 250-day window."""
    return "green" if n_exceptions <= 4 else ("yellow" if n_exceptions <= 9 else "red")


def backtest_table(fc: pd.DataFrame, levels: list[float], alpha: float) -> pd.DataFrame:
    rows = []
    for m, g in fc.dropna(subset=["pnl"]).groupby("method", sort=False):
        g = g.sort_values("target")
        loss = -g["pnl"].to_numpy()
        for lv in levels:
            col = f"var_{int(round(lv * 1000)) / 10:g}"
            exc = loss > g[col].to_numpy()
            p = 1 - lv
            lr_pof, p_pof = kupiec(exc, p)
            ch = christoffersen(exc, p)
            rows.append({"method": m, "level": lv, "n": len(exc), "expected": len(exc) * p, "observed": int(exc.sum()),
                         "rate": exc.mean(), "lr_pof": lr_pof, "p_pof": p_pof, **ch,
                         "pass_pof": p_pof >= alpha, "pass_ind": ch["p_ind"] >= alpha, "pass_cc": ch["p_cc"] >= alpha})
    return pd.DataFrame(rows)


def traffic_light_table(fc: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    rows = []
    for m, g in fc.dropna(subset=["pnl"]).groupby("method", sort=False):
        g = g.sort_values("target")
        exc = pd.Series((-g["pnl"] > g["var_99"]).astype(int).to_numpy(), index=g["target"])
        roll = exc.rolling(window).sum().dropna()
        zones = roll.map(lambda x: traffic_light(int(x)))
        last = int(exc.iloc[-window:].sum())
        rows.append({"method": m, "latest_250d_exceptions": last, "latest_zone": traffic_light(last),
                     "share_windows_green": float((zones == "green").mean()), "share_windows_yellow": float((zones == "yellow").mean()),
                     "share_windows_red": float((zones == "red").mean()), "max_250d_exceptions": int(roll.max())})
    return pd.DataFrame(rows)


def es_check(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in fc.dropna(subset=["pnl"]).groupby("method", sort=False):
        loss = -g["pnl"]
        exc = loss > g["var_99"]
        ratio = (loss[exc] / g.loc[exc, "es_99"])
        rows.append({"method": m, "exceptions_99": int(exc.sum()), "mean_loss_over_es99": ratio.mean(),
                     "median_loss_over_es99": ratio.median(), "share_loss_above_es99": float((ratio > 1).mean())})
    return pd.DataFrame(rows)


def by_subperiod(fc: pd.DataFrame, subperiods: dict) -> pd.DataFrame:
    rows = []
    d = fc.dropna(subset=["pnl"])
    for name, (a, b) in subperiods.items():
        sel = d[(d["target"] >= pd.Timestamp(a)) & ((d["target"] <= pd.Timestamp(b)) if b else True)]
        for m, g in sel.groupby("method", sort=False):
            loss = -g["pnl"]
            rows.append({"subperiod": name, "method": m, "n": len(g), "exceptions_99": int((loss > g["var_99"]).sum()),
                         "expected_99": len(g) * 0.01, "exceptions_95": int((loss > g["var_95"]).sum()),
                         "expected_95": len(g) * 0.05})
    return pd.DataFrame(rows)


def exceptions_figure(fc: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    d = fc.dropna(subset=["pnl"])
    for m, g in d.groupby("method"):
        g = g.sort_values("target")
        ax.plot(g["target"], np.cumsum(-g["pnl"] > g["var_99"]), lw=1, label=m)
    n = d.groupby("method").size().iloc[0]
    tgt = d[d["method"] == "hs"].sort_values("target")["target"]
    ax.plot(tgt, 0.01 * np.arange(1, n + 1), color="black", ls="--", lw=1, label="expected (1%)")
    ax.set_ylabel("cumulative 99% VaR exceptions")
    ax.set_title("SYNTHETIC book: cumulative 99% VaR exceptions by method", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    fc = pd.read_csv(cfg.path("clean_dir") / "var_forecasts.csv", parse_dates=["origin", "target"])
    alpha = float(cfg["forecasting"]["alpha"])
    tables = {
        "backtest": backtest_table(fc, list(cfg["var"]["var_levels"]), alpha),
        "traffic_light": traffic_light_table(fc),
        "es_check": es_check(fc),
        "by_subperiod": by_subperiod(fc, cfg["sample"]["subperiods"]),
    }
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage6_{k}.csv", index=False)
    (out / "figures").mkdir(exist_ok=True)
    exceptions_figure(fc, out / "figures" / "stage6_exceptions.png")
    from gasrisk.backtest_report import write_stage6_report

    write_stage6_report(cfg, tables, fc)
    return tables


def main() -> None:
    t = run(load_config())
    print(t["backtest"][["method", "level", "expected", "observed", "p_pof", "p_ind", "p_cc", "pass_cc"]].to_string(index=False))
    print(t["traffic_light"].to_string(index=False))


if __name__ == "__main__":
    main()
