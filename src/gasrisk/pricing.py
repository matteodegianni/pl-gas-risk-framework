"""Stage 10: Black-76 valuation of European options on TTF futures, with Greeks.

Usage::

    python -m gasrisk.pricing [--offline]

* Underlying: the TTF second-month future (ICE Endex) on the last panel date; its delivery month follows from the
  futures expiry rule of Stage 1.
* Expiry: ICE Endex TTF options cease trading 5 calendar days before the first day of the delivery month, moved to
  the nearest prior UK business day if that day is not one. Time to expiry ACT/365.
* Rate: ECB euro area AAA yield curve spot rates (Svensson, percent, continuously compounded per the ECB technical
  notes), linearly interpolated to the expiry.
* Margining: ICE Endex TTF options are futures-style margined, so the premium is not paid upfront and is not
  discounted (Black-76 with discount factor 1). The premium-paid (discounted) value is shown for OTC options.
* Volatility: **assumption**. There is no free TTF implied volatility data, so the volatility input is the Stage 4
  GARCH-t forecast to expiry (annualised over ACT/365), with historical and EWMA volatility as alternatives. All
  option values are model values, not market prices.
* Validation: two published textbook examples, put-call parity on a strike grid, finite-difference Greeks and an
  implied-volatility round trip.

Option values are reported as a share of the futures price (Black-76 is homogeneous of degree one in F and K), so no
raw TTF price level is written to ``outputs/``.
"""

from __future__ import annotations

import argparse
import io
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from gasrisk.config import Config, load_config
from gasrisk.data.calendars import ttf_expiry, uk_business_days
from gasrisk.volatility import MeanSpec, aggregate_variance, ar_coefficients, garch_model, garch_variance_path

# Published reference examples (inputs and the value printed in the source).
REFERENCE_EXAMPLES = [
    {"source": "Hull, Options, Futures, and Other Derivatives: Black's model, European futures put",
     "F": 20.0, "K": 20.0, "T": 4 / 12, "sigma": 0.25, "r": 0.09, "kind": "put", "published": 1.12, "decimals": 2},
    {"source": "Haug, The Complete Guide to Option Pricing Formulas: Black-76 example (call)",
     "F": 19.0, "K": 19.0, "T": 0.75, "sigma": 0.28, "r": 0.10, "kind": "call", "published": 1.7011, "decimals": 4},
    {"source": "Haug, The Complete Guide to Option Pricing Formulas: Black-76 example (put)",
     "F": 19.0, "K": 19.0, "T": 0.75, "sigma": 0.28, "r": 0.10, "kind": "put", "published": 1.7011, "decimals": 4},
]


# --------------------------------------------------------------------------- Black-76


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def black76(F, K, T, sigma, r=0.0, kind: str = "call", discount: bool = True):
    """Black-76 value of a European option on a futures price.

    ``discount=False`` gives the futures-style margined value (no discounting of the premium).
    """
    d1, d2 = _d1d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r) * np.asarray(T)) if discount else 1.0
    if kind == "call":
        return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    if kind == "put":
        return df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
    raise ValueError(kind)


def greeks(F, K, T, sigma, r=0.0, kind: str = "call", discount: bool = True) -> dict[str, np.ndarray]:
    """Analytic Black-76 Greeks.

    delta = dV/dF, gamma = d2V/dF2, vega = dV/dsigma (per 1.00 of volatility), theta = dV/dt (per year of calendar
    time, i.e. -dV/dT), rho = dV/dr.
    """
    d1, d2 = _d1d2(F, K, T, sigma)
    T = np.asarray(T)
    df = np.exp(-np.asarray(r) * T) if discount else np.ones_like(T, dtype=float)
    pdf = stats.norm.pdf(d1)
    v = black76(F, K, T, sigma, r, kind, discount)
    if kind == "call":
        delta = df * stats.norm.cdf(d1)
    else:
        delta = -df * stats.norm.cdf(-d1)
    gamma = df * pdf / (F * sigma * np.sqrt(T))
    vega = df * F * pdf * np.sqrt(T)
    decay = -df * F * pdf * sigma / (2 * np.sqrt(T))
    theta = decay + (r * v if discount else 0.0)
    rho = -T * v if discount else np.zeros_like(v)
    return {"value": v, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, F: float, K: float, T: float, r: float = 0.0, kind: str = "call", discount: bool = True) -> float:
    """Black-76 implied volatility by Brent's method."""
    return float(optimize.brentq(lambda s: black76(F, K, T, s, r, kind, discount) - price, 1e-6, 10.0, xtol=1e-12))


def fd_greeks(F, K, T, sigma, r, kind, discount, bump: float) -> dict[str, float]:
    """Central finite-difference Greeks (relative bumps)."""
    f = lambda **kw: float(black76(**{"F": F, "K": K, "T": T, "sigma": sigma, "r": r, "kind": kind, "discount": discount, **kw}))  # noqa: E731
    hF, hs, hT, hr = F * bump, sigma * bump, T * bump, 1e-5
    return {"delta": (f(F=F + hF) - f(F=F - hF)) / (2 * hF),
            "gamma": (f(F=F + hF) - 2 * f() + f(F=F - hF)) / hF**2,
            "vega": (f(sigma=sigma + hs) - f(sigma=sigma - hs)) / (2 * hs),
            "theta": -(f(T=T + hT) - f(T=T - hT)) / (2 * hT),
            "rho": (f(r=r + hr) - f(r=r - hr)) / (2 * hr)}


# --------------------------------------------------------------------------- contract calendar


def option_expiry(delivery_month: pd.Period, days_before: int = 5, subdiv: str = "ENG") -> pd.Timestamp:
    """ICE Endex TTF option expiry: ``days_before`` calendar days before the delivery month, else prior UK business day."""
    target = delivery_month.start_time.normalize() - pd.Timedelta(days=days_before)
    bd = uk_business_days(target - pd.Timedelta(days=14), target, subdiv)
    return bd[-1]


def second_month(as_of: pd.Timestamp, n_days_before: int = 2, subdiv: str = "ENG") -> pd.Period:
    """Delivery month of the TTF second-month future on ``as_of`` (front = first month not yet expired)."""
    m = as_of.to_period("M")
    while ttf_expiry(m, n_days_before, subdiv) < as_of:
        m += 1
    return m + 1


# --------------------------------------------------------------------------- inputs


def fetch_ecb_curve(cfg: Config, as_of: pd.Timestamp, offline: bool = False) -> pd.DataFrame:
    """ECB AAA curve spot rates (percent) for the configured tenors, last 60 days up to ``as_of``."""
    from gasrisk.data.http import get
    from gasrisk.data.sources import record_provenance

    p = cfg["pricing"]
    path = cfg.path("raw_dir") / "ecb" / "yield_curve.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not offline:
        frames = []
        for tenor, key in p["ecb_curve_keys"].items():
            resp = get(f"{p['ecb_curve_url']}/{key}", {"format": "csvdata",
                                                         "startPeriod": str((as_of - pd.Timedelta(days=60)).date())}, cfg["http"])
            resp.raise_for_status()
            d = pd.read_csv(io.StringIO(resp.text))[["TIME_PERIOD", "OBS_VALUE"]]
            frames.append(d.assign(tenor_years=float(tenor)))
        pd.concat(frames).to_csv(path, index=False)
        record_provenance(cfg, "ecb_yield_curve", path, {"source": "ECB Data API YC", "url": p["ecb_curve_url"]})
    d = pd.read_csv(path)
    d["TIME_PERIOD"] = pd.to_datetime(d["TIME_PERIOD"])
    return d.pivot(index="TIME_PERIOD", columns="tenor_years", values="OBS_VALUE").sort_index()


def curve_on(curve: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series]:
    """Latest curve observation on or before ``as_of``."""
    c = curve.loc[:as_of].dropna()
    return c.index[-1], c.iloc[-1]


def zero_rate(curve_row: pd.Series, t: float | np.ndarray) -> np.ndarray:
    """Continuously compounded rate (decimal) linearly interpolated in tenor, flat outside the grid."""
    x = np.asarray(curve_row.index, dtype=float)
    return np.interp(t, x, curve_row.to_numpy(dtype=float) / 100.0)


def ttf_vol_assumptions(cfg: Config, rets: pd.Series, as_of: pd.Timestamp, h_days: int, T: float) -> pd.DataFrame:
    """Annualised volatility candidates (ACT/365) for the option to expiry. All are ASSUMPTIONS, not implied vols."""
    selected = pd.read_csv(cfg.path("outputs_dir") / "stage4_selected_spec.csv").set_index("series").loc["ttf"]
    cand = {d["name"]: MeanSpec.from_cfg(d) for d in cfg["volatility"]["mean_candidates"]}
    spec = cand[selected["mean"]]
    y = rets.loc[:as_of].dropna().iloc[-int(cfg["var"]["garch_window"]):] * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = garch_model(y, selected["dist"], spec).fit(disp="off", options={"maxiter": 2000})
    path = garch_variance_path(res.params, float(np.asarray(res.resid)[-1]),
                               float(np.asarray(res.conditional_volatility)[-1]) ** 2, h_days)
    total_var = aggregate_variance(path, ar_coefficients(res.params, spec), h_days) / 1e4
    hist = rets.loc[:as_of].dropna().iloc[-int(cfg["pricing"]["hist_window"]):]
    lam = float(cfg["volatility"]["ewma_lambda"])
    r = rets.loc[:as_of].dropna().to_numpy()
    s2 = float(np.var(r[:63]))
    for x in r[63:]:
        s2 = lam * s2 + (1 - lam) * x * x
    rows = [
        {"vol_source": "garch_t", "description": f"Stage 4 GARCH(1,1)-t ({selected['mean']} mean) forecast to expiry",
         "total_variance": total_var, "sigma_annual": np.sqrt(total_var / T)},
        {"vol_source": "hist_63d", "description": "63-day historical volatility, held flat to expiry",
         "total_variance": float(hist.var(ddof=1)) * h_days, "sigma_annual": np.sqrt(float(hist.var(ddof=1)) * h_days / T)},
        {"vol_source": "ewma", "description": "EWMA (lambda 0.94) volatility, held flat to expiry",
         "total_variance": s2 * h_days, "sigma_annual": np.sqrt(s2 * h_days / T)},
    ]
    return pd.DataFrame(rows).assign(trading_days_to_expiry=h_days, T_years=T)




def market_inputs(cfg: Config, offline: bool = False) -> dict:
    """As-of date, underlying, expiry, rate and volatility used by Stages 10-13."""
    p = cfg["pricing"]
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    as_of = panel.index[-1]
    nb, sub = int(cfg["ttf"]["expiry_uk_business_days_before"]), cfg["ttf"]["holiday_subdivision"]
    month = second_month(as_of, nb, sub)
    expiry = option_expiry(month, int(p["option_expiry_calendar_days_before"]), sub)
    T = (expiry - as_of).days / float(p["day_count"])
    h = len(uk_business_days(as_of + pd.Timedelta(days=1), expiry, sub))
    try:
        curve = fetch_ecb_curve(cfg, as_of, offline)
    except Exception:  # network failure: fall back to the stored file
        curve = fetch_ecb_curve(cfg, as_of, offline=True)
    curve_date, row = curve_on(curve, as_of)
    vols = ttf_vol_assumptions(cfg, panel["ttf_risk_ret"], as_of, h, T)
    return {"as_of": as_of, "F": float(panel["ttf_c2"].iloc[-1]), "fx": float(panel["eurpln"].iloc[-1]),
            "delivery_month": month, "expiry": expiry, "T": T, "h_days": h, "curve_date": curve_date, "curve": row,
            "r": float(zero_rate(row, T)), "vols": vols,
            "sigma": float(vols.set_index("vol_source").loc[p["vol_source"], "sigma_annual"]),
            "discount": p["margining"] != "futures_style"}


# --------------------------------------------------------------------------- validation tables


def reference_checks() -> pd.DataFrame:
    rows = []
    for e in REFERENCE_EXAMPLES:
        v = float(black76(e["F"], e["K"], e["T"], e["sigma"], e["r"], e["kind"], True))
        rows.append({"source": e["source"], "kind": e["kind"], "F": e["F"], "K": e["K"], "T": e["T"], "sigma": e["sigma"],
                     "r": e["r"], "published": e["published"], "computed": v,
                     "matches_published_rounding": round(v, e["decimals"]) == e["published"]})
    return pd.DataFrame(rows)


def parity_checks(F: float, T: float, sigma: float, r: float, grid: list[float]) -> pd.DataFrame:
    rows = []
    for disc in (False, True):
        for m in grid:
            K = m * F
            c = float(black76(F, K, T, sigma, r, "call", disc))
            p = float(black76(F, K, T, sigma, r, "put", disc))
            df = np.exp(-r * T) if disc else 1.0
            rows.append({"margining": "premium_paid" if disc else "futures_style", "moneyness_K_over_F": m,
                         "call_minus_put_pct_F": (c - p) / F, "df_times_F_minus_K_pct_F": df * (F - K) / F,
                         "abs_error_pct_F": abs((c - p) - df * (F - K)) / F})
    return pd.DataFrame(rows)


def fd_checks(F, T, sigma, r, bump) -> pd.DataFrame:
    rows = []
    for disc in (False, True):
        for kind in ("call", "put"):
            for m in (0.9, 1.0, 1.1):
                a = greeks(F, m * F, T, sigma, r, kind, disc)
                n = fd_greeks(F, m * F, T, sigma, r, kind, disc, bump)
                for g in ("delta", "gamma", "vega", "theta", "rho"):
                    av, nv = float(a[g]), n[g]
                    rows.append({"margining": "premium_paid" if disc else "futures_style", "kind": kind,
                                 "moneyness_K_over_F": m, "greek": g, "analytic": av, "finite_difference": nv,
                                 "abs_diff": abs(av - nv), "rel_diff": abs(av - nv) / max(abs(av), 1e-12)})
    return pd.DataFrame(rows)


def option_grid(F: float, T: float, sigma: float, r: float, grid: list[float], discount: bool) -> pd.DataFrame:
    rows = []
    for kind in ("call", "put"):
        for m in grid:
            g = greeks(F, m * F, T, sigma, r, kind, discount)
            iv = implied_vol(float(g["value"]), F, m * F, T, r, kind, discount)
            rows.append({"kind": kind, "moneyness_K_over_F": m, "value_pct_F": float(g["value"]) / F,
                         "delta": float(g["delta"]), "gamma_x_F": float(g["gamma"]) * F,
                         "vega_pct_F_per_vol_pt": float(g["vega"]) / F / 100, "theta_pct_F_per_day": float(g["theta"]) / F / 365,
                         "implied_vol_roundtrip_error": abs(iv - sigma)})
    return pd.DataFrame(rows)


def vol_sensitivity(F, T, r, vols: pd.DataFrame, discount: bool) -> pd.DataFrame:
    rows = []
    for _, v in vols.iterrows():
        for kind in ("call", "put"):
            rows.append({"vol_source": v["vol_source"], "sigma_annual": v["sigma_annual"], "kind": kind,
                         "atm_value_pct_F": float(black76(F, F, T, v["sigma_annual"], r, kind, discount)) / F})
    return pd.DataFrame(rows)


def run(cfg: Config, offline: bool = False) -> dict[str, pd.DataFrame]:
    p = cfg["pricing"]
    mi = market_inputs(cfg, offline)
    F, T, s, r, disc = mi["F"], mi["T"], mi["sigma"], mi["r"], mi["discount"]
    grid = [float(x) for x in p["moneyness_grid"]]
    inputs = pd.DataFrame([{"as_of": mi["as_of"].date(), "underlying": f"TTF {mi['delivery_month']} future (second month)",
                            "option_expiry": mi["expiry"].date(), "T_years_act365": T,
                            "uk_business_days_to_expiry": mi["h_days"], "ecb_curve_date": mi["curve_date"].date(),
                            "eur_rate_cc": r, "vol_source_used": p["vol_source"], "sigma_annual": s,
                            "margining": p["margining"]}])
    tables = {
        "inputs": inputs,
        "vol_assumptions": mi["vols"],
        "reference": reference_checks(),
        "parity": parity_checks(F, T, s, r, grid),
        "fd_greeks": fd_checks(F, T, s, r, float(p["fd_bump_rel"])),
        "grid": option_grid(F, T, s, r, grid, disc),
        "grid_premium_paid": option_grid(F, T, s, r, grid, True),
        "vol_sensitivity": vol_sensitivity(F, T, r, mi["vols"], disc),
    }
    out = cfg.path("outputs_dir")
    for k, t in tables.items():
        t.to_csv(out / f"stage10_{k}.csv", index=False)
    from gasrisk.pricing_report import write_stage10_report

    write_stage10_report(cfg, tables)
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    t = run(load_config(), ap.parse_args().offline)
    print(t["inputs"].T.to_string())
    print(t["reference"].to_string(index=False))


if __name__ == "__main__":
    main()
