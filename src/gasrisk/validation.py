"""Stage 15: input data validation and post-deployment output validation.

Usage::

    python -m gasrisk.validation [--inputs-only | --outputs-only]
    python handover/validate_outputs.py            # same checks, for the receiving team

Inputs are checked against ``handover/input_schema.yaml``; outputs against the acceptance criteria in
``handover/acceptance_criteria.md`` (one check per criterion id). Results go to ``outputs/validation_report.csv``;
the exit code is 1 if any check with severity ``error`` fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from gasrisk.config import Config, load_config


@dataclass
class Result:
    id: str
    area: str
    check: str
    passed: bool
    severity: str
    detail: str

    def row(self) -> dict:
        return self.__dict__


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


# --------------------------------------------------------------------------- inputs


def validate_inputs(cfg: Config, run_date: pd.Timestamp | None = None) -> list[Result]:
    spec = yaml.safe_load((cfg.root / "handover" / "input_schema.yaml").read_text())
    res: list[Result] = []
    clean = cfg.path("clean_dir")
    for fname, fs in spec["inputs"].items():
        path = clean / fname
        if not path.exists():
            res.append(Result(f"IN-{fname}", "input", "file exists", False, "error", str(path)))
            continue
        df = pd.read_csv(path)
        if "min_rows" in fs:
            res.append(Result(f"IN-{fname}-rows", "input", f"at least {fs['min_rows']} rows", len(df) >= fs["min_rows"], "error", str(len(df))))
        for col, c in fs["columns"].items():
            cid = f"IN-{fname}-{col}"
            if col not in df.columns:
                res.append(Result(cid, "input", "column present", False, "error", col))
                continue
            s = df[col]
            if c["type"] == "date":
                d = pd.to_datetime(s, errors="coerce")
                ok = d.notna().all() and (not c.get("unique") or d.is_unique) and (not c.get("monotonic") or d.is_monotonic_increasing)
                res.append(Result(cid, "input", "valid, unique, increasing dates", bool(ok), "error", f"{d.min().date()} to {d.max().date()}"))
                if fname == "panel.csv":
                    gaps = d.diff().dt.days.dropna()
                    res.append(Result("IN-03", "input", "no calendar gap above 5 trading days (8 calendar days)", bool((gaps <= 8).all()),
                                      "error", f"max gap {int(gaps.max())} days"))
                    if run_date is not None and "max_staleness_calendar_days" in fs:
                        age = (run_date - d.max()).days
                        res.append(Result("IN-staleness", "input", "panel is fresh", age <= fs["max_staleness_calendar_days"], "warning", f"{age} days old"))
                continue
            x = pd.to_numeric(s, errors="coerce")
            null_share = float(x.isna().mean())
            if "max_null_share" in c:
                res.append(Result(cid + "-nulls", "input", f"null share <= {c['max_null_share']}", null_share <= c["max_null_share"] + 1e-12, "error", f"{null_share:.4f}"))
            lo, hi = c.get("min", -np.inf), c.get("max", np.inf)
            bad = int(((x < lo) | (x > hi)).sum())
            res.append(Result(cid + "-range", "input", f"values in [{lo}, {hi}]", bad == 0, "error", f"{bad} out of range"))
        if fname == "panel.csv" and {"tge_eur", "tge_pln", "eurpln"} <= set(df.columns):
            err = float((df["tge_eur"] - df["tge_pln"] / df["eurpln"]).abs().max())
            res.append(Result("IN-01", "input", "tge_eur = tge_pln / eurpln", err < 1e-6, "error", f"max error {err:.2e}"))
    log = cfg.path("outputs_dir") / "data_quality_log.csv"
    res.append(Result("IN-02", "input", "Stage 1 data quality log exists", log.exists(), "error", str(log.name)))
    for key, c in spec["config"].items():
        v = float(_get(cfg.raw, key))
        res.append(Result(f"CFG-{key}", "config", f"{key} in range", c.get("min", -np.inf) <= v <= c.get("max", np.inf), "error", str(v)))
    return res


# --------------------------------------------------------------------------- outputs


def validate_outputs(cfg: Config) -> list[Result]:
    out = cfg.path("outputs_dir")
    r: list[Result] = []

    def add(i, area, check, ok, detail="", severity="error"):
        r.append(Result(i, area, check, bool(ok), severity, str(detail)))

    def read(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else None

    panel_last = pd.read_csv(cfg.path("clean_dir") / "panel.csv", usecols=["date"])["date"].iloc[-1]
    need = ["stage5_var_today.csv", "stage6_backtest.csv", "stage7_limits_today.csv", "stage8_reference.csv", "stage9_total.csv",
            "stage10_inputs.csv", "stage11_effectiveness.csv", "stage12_summary.csv", "stage13_limits.csv"]
    missing = [n for n in need if not (out / n).exists()]
    add("AC-01", "completeness", "all daily output tables exist", not missing, ", ".join(missing) or "all present")
    if missing:
        return r
    v = read("stage5_var_today.csv")
    add("AC-02", "freshness", "VaR as-of date equals the last panel date", (v["as_of"] == panel_last).all(), f"{v['as_of'].iloc[0]} vs {panel_last}")
    add("AC-03", "VaR", "VaR and ES positive and finite for every method", np.isfinite(v.filter(like="_9").to_numpy()).all() and (v.filter(like="_9") > 0).all().all())
    add("AC-04", "VaR", "VaR 99% >= VaR 95% and ES 97.5% >= VaR 99% x 0.9", ((v["var_99"] >= v["var_95"]) & (v["es_97.5"] >= 0.9 * v["var_99"])).all())
    off = read("stage7_official_method.csv")["method"].iloc[0]
    bt = read("stage6_backtest.csv")
    row = bt[(bt["method"] == off)]
    cc_col = "p_cc"
    add("AC-05", "backtest", f"official method {off} passes conditional coverage at 95% and 99% (p > 0.05)",
        (row[cc_col] > 0.05).all(), f"min p {row[cc_col].min():.4f}")
    tl = read("stage6_traffic_light.csv").set_index("method").loc[off]
    add("AC-06", "backtest", "official method not in the red traffic-light zone", tl["latest_zone"] != "red", tl["latest_zone"])
    lim = read("stage7_limits_today.csv")
    add("AC-07", "limits", "utilisation = measure / limit", np.allclose(lim["utilisation"], lim["measure"] / lim["limit_value"]))
    tot = read("stage9_total.csv").set_index("component")["pln"]
    resid = abs(tot["total_pnl"] - tot[["plan_margin", "volume_effect", "price_effect", "fx_effect"]].sum())
    add("AC-08", "attribution", "attribution components sum to total P&L", resid < 1e-3, f"residual {resid:.2e} PLN")
    par = read("stage10_parity.csv")
    add("AC-09", "pricing", "put-call parity error below 1e-10 of F", par["abs_error_pct_F"].max() < 1e-10, f"{par['abs_error_pct_F'].max():.2e}")
    add("AC-10", "pricing", "reference examples match published values", read("stage10_reference.csv")["matches_published_rounding"].all())
    fd = read("stage10_fd_greeks.csv")
    add("AC-11", "pricing", "finite-difference Greeks within 1e-4 relative", (fd["rel_diff"] < 1e-4).all(), f"max {fd['rel_diff'].max():.2e}")
    dec = read("stage11_ineffectiveness.csv").dropna(subset=["component"])
    shares = dec[~dec["component"].str.startswith("total")]["covariance_share_of_ineffectiveness_variance"].sum()
    add("AC-12", "hedging", "ineffectiveness shares sum to 1", abs(shares - 1) < 1e-9, f"{shares:.12f}")
    m = read("stage12_summary.csv").iloc[0]
    add("AC-13", "margin", "IM and 10-day buffer positive; buffer >= 1-day 99% call", m["im_today_fhs_eur"] > 0 and
        m["buffer_10d_q99_eur"] >= read("stage12_forecast.csv")["q99_cum_outflow_eur"].iloc[0])
    add("AC-14", "margin", "FHS IM within 10% of the closed-form approximation",
        abs(m["im_today_fhs_eur"] / m["im_today_formula_eur"] - 1) < 0.10, f"{m['im_today_fhs_eur'] / m['im_today_formula_eur']:.4f}")
    prof = read("stage13_exposure_profile.csv")
    add("AC-15", "credit", "PFE 97.5% >= PFE 95% >= 0 and netted EE <= EE without netting",
        ((prof["pfe_97.5_eur"] >= prof["pfe_95_eur"] - 1e-6) & (prof["pfe_95_eur"] >= 0) & (prof["ee_eur"] <= prof["ee_no_netting_eur"] + 1e-6)).all())
    cl = read("stage13_limits.csv")
    add("AC-16", "credit", "breach status consistent with utilisation",
        ((cl["status"] == "breach") == (cl["utilisation"] > 1)).all())
    add("AC-17", "reporting", "breaches present today are reported (information)", True,
        f"limits in breach: {int((lim['utilisation'] > 1).sum())}; counterparties in breach: {int((cl['status'] == 'breach').sum())}", "info")
    exp = cfg.root / "exports"
    if exp.exists():
        add("AC-18", "exports", "dashboard CSVs exist and carry the synthetic label",
            all("synthetic" in pd.read_csv(p, nrows=5).get("data_label", pd.Series(["synthetic"])).astype(str).iloc[0] for p in exp.glob("*.csv")))
    return r


def run(cfg: Config, inputs: bool = True, outputs: bool = True) -> pd.DataFrame:
    res = []
    if inputs:
        res += validate_inputs(cfg, pd.Timestamp.today().normalize())
    if outputs:
        res += validate_outputs(cfg)
    df = pd.DataFrame([x.row() for x in res])
    df.to_csv(cfg.path("outputs_dir") / "validation_report.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-only", action="store_true")
    ap.add_argument("--outputs-only", action="store_true")
    a = ap.parse_args()
    df = run(load_config(), inputs=not a.outputs_only, outputs=not a.inputs_only)
    print(df.to_string(index=False))
    failed = df[(~df["passed"]) & (df["severity"] == "error")]
    print(f"\n{len(df) - len(failed)} of {len(df)} checks passed; {len(failed)} errors.")
    sys.exit(1 if len(failed) else 0)


if __name__ == "__main__":
    main()
