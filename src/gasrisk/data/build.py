"""Stage 1 build: load every source, run quality checks, align, write outputs.

Usage::

    python -m gasrisk.data.build            # refresh API data, then build
    python -m gasrisk.data.build --offline  # rebuild from stored raw files

Outputs (committed, no raw price levels): ``outputs/data_quality_log.csv`` and
``outputs/stage1_*.csv`` plus ``outputs/stage1_report.md``.
Clean series (gitignored, derived from licensed data): ``data/clean/*.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config
from gasrisk.data import sources
from gasrisk.data.calendars import pl_business_days, roll_schedule, uk_business_days
from gasrisk.data.quality import (
    IssueLog,
    check_duplicates,
    check_extra_days,
    check_missing_days,
    check_nonpositive,
    check_ohlc,
    check_stale,
    handle_outliers,
)
from gasrisk.data.rolls import roll_adjusted_returns, roll_validation

INV = "Investing.com"
INSTRAT = "Instrat (TGE)"
INSTRAT_EEX = "Instrat (EEX)"


@dataclass
class Stage1Result:
    ttf: pd.DataFrame
    tge: pd.DataFrame
    fx: pd.DataFrame
    co2: pd.DataFrame
    panel: pd.DataFrame
    log: IssueLog
    tables: dict[str, pd.DataFrame]


def _logret(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return np.log(s / s.shift(1))


# --------------------------------------------------------------------------- TTF


def _prepare_ttf(df: pd.DataFrame, series: str, cfg: Config, log: IssueLog) -> pd.DataFrame:
    """Structural checks on one Investing.com export; returns weekday rows only."""
    df = check_duplicates(df, INV, series, log)

    first = df.index[0]
    log.add(INV, series, "first_row_change_artefact", "info", "Change % column ignored; returns recomputed from settlement",
            "first row reports a change but the export has no prior row to compute it from", first,
            detail=f"reported change {df['change_pct_reported'].iloc[0]:.4f}")
    implied = df["close"].pct_change()
    gap = (df["change_pct_reported"] - implied).abs().astype(float)
    mism = df.index[1:][(gap > 0.001).to_numpy(dtype=bool, na_value=False)[1:]]
    for d in mism:
        log.add(INV, series, "change_pct_mismatch", "warning", "Change % ignored", "reported change differs from recomputed change by > 0.1pp", d)

    weekend = df.index[df.index.day_of_week >= 5]
    if len(weekend):
        med_vol = df["volume"].median()
        for d in weekend:
            prev = df["close"].loc[:d].iloc[-2]
            nxt = df["close"].loc[d:].iloc[1] if len(df.loc[d:]) > 1 else np.nan
            log.add(INV, series, "weekend_row", "warning", "dropped",
                    "weekend session row; settlement prices are business-day prices and the following Monday row exists",
                    d, detail=f"volume ratio to series median {df.at[d, 'volume'] / med_vol:.4f}; log diff vs previous {np.log(df.at[d, 'close'] / prev):.4f}; vs next {np.log(df.at[d, 'close'] / nxt):.4f}")
        df = df.drop(weekend)

    df = df.assign(close=check_nonpositive(df["close"], INV, series, log))
    check_ohlc(df, INV, series, log)
    uk = uk_business_days(df.index.min(), df.index.max(), cfg["ttf"]["holiday_subdivision"])
    check_missing_days(df.index, uk, INV, series, log, "UK business")
    check_extra_days(df.index, uk, INV, series, log, "UK")
    check_stale(df["close"], int(cfg["quality"]["stale_run_obs"]), INV, series, log)
    return df


def build_ttf(cfg: Config, log: IssueLog) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean c1 and c2, then compute roll-adjusted returns."""
    q = cfg["quality"]
    c1 = _prepare_ttf(sources.load_investing(cfg, "ttf_c1"), "ttf_c1", cfg, log)
    c2 = _prepare_ttf(sources.load_investing(cfg, "ttf_c2"), "ttf_c2", cfg, log)

    rolls = roll_schedule(c2.index, int(cfg["ttf"]["expiry_uk_business_days_before"]), cfg["ttf"]["holiday_subdivision"])
    roll_dates = pd.DatetimeIndex(rolls["roll_date"])

    r1, r2 = _logret(c1["close"]), _logret(c2["close"])
    r1[r1.index.isin(roll_dates)] = np.nan
    r2[r2.index.isin(roll_dates)] = np.nan
    c1_close, t1 = handle_outliers(c1["close"], {"ttf_c2": r2}, q, INV, "ttf_c1", log, exclude=roll_dates)
    c2_close, t2 = handle_outliers(c2["close"], {"ttf_c1": r1}, q, INV, "ttf_c2", log, exclude=roll_dates)

    ttf = roll_adjusted_returns(c1_close.dropna(), c2_close.dropna(), roll_dates, float(q["roll_z"]), int(q["roll_scale_window"]), log)
    ttf.attrs["c1_rows_in_source"] = int(c1_close.notna().sum())  # c1 also has days without a c2 print
    ttf["c1_volume"] = c1["volume"].reindex(ttf.index)
    ttf["c2_volume"] = c2["volume"].reindex(ttf.index)
    outliers = pd.concat([t1.assign(series="ttf_c1"), t2.assign(series="ttf_c2")], ignore_index=True)
    return ttf, rolls, outliers


# --------------------------------------------------------------------------- TGE


def build_tge(cfg: Config, log: IssueLog, offline: bool, ttf_ret: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TGEgasDA by delivery day and trade day, with checks."""
    q, tcfg = cfg["quality"], cfg["tge"]
    raw = sources.fetch_tge_gas(cfg, offline)
    df = raw.set_index("delivery_date")
    df = check_duplicates(df, INSTRAT, "tge_gas_da", log)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    check_missing_days(df.index, full, INSTRAT, "tge_gas_da", log, "calendar (delivery)")
    df = df.assign(price_pln_mwh=check_nonpositive(df["price_pln_mwh"], INSTRAT, "tge_gas_da", log))

    zero = df.index[(df["volume_mwh"].fillna(0) <= 0).to_numpy()]
    log.add_runs(zero, df.index, INSTRAT, "tge_gas_da", "zero_volume", "warning", "kept; flagged", "index published with no traded volume")
    vol_med = df["volume_mwh"].rolling(30, min_periods=10).median().shift(1)
    low = df.index[(df["volume_mwh"] < float(tcfg["low_volume_ratio"]) * vol_med).to_numpy()]
    log.add_runs(low, df.index, INSTRAT, "tge_gas_da", "low_volume", "info", "kept; flagged",
                 f"traded volume below {tcfg['low_volume_ratio']:.0%} of trailing 30-day median; price is a thinner average")

    log.add(INSTRAT, "tge_gas_da", "delivery_to_trade_date", "info", f"trade_date = delivery_date - {tcfg['trade_date_lag_days']} day",
            "TGEgasDA for delivery day D is the volume weighted price of D-1 trades; aligning to TTF settlements and FX fixings needs the trade date",
            df.index.min(), df.index.max(), len(df))

    by_trade = df.reset_index().set_index("trade_date")
    clean_price, table = handle_outliers(by_trade["price_pln_mwh"], {"ttf_c2_adj": ttf_ret}, q, INSTRAT, "tge_gas_da", log)
    by_trade["price_pln_mwh"] = clean_price
    by_trade["is_weekend_trade"] = by_trade.index.day_of_week >= 5
    return by_trade, table


# --------------------------------------------------------------------------- FX


def build_fx(cfg: Config, log: IssueLog, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """NBP (primary), ECB and Yahoo EUR/PLN with cross-source checks."""
    q = cfg["quality"]
    nbp = sources.fetch_nbp(cfg, offline)
    ecb = sources.fetch_ecb(cfg, offline)
    yah = sources.fetch_yahoo_eurpln(cfg, offline)

    nbp = check_duplicates(nbp.to_frame(), "NBP", "eurpln_nbp", log)["eurpln_nbp"]
    nbp = check_nonpositive(nbp, "NBP", "eurpln_nbp", log)
    pl = pl_business_days(nbp.index.min(), nbp.index.max())
    check_missing_days(nbp.index, pl, "NBP", "eurpln_nbp", log, "Polish business")
    check_extra_days(nbp.index, pl, "NBP", "eurpln_nbp", log, "Polish")
    check_stale(nbp, int(q["stale_run_obs"]), "NBP", "eurpln_nbp", log)

    yah = check_duplicates(yah, "Yahoo", "eurpln_yahoo", log)
    yah = yah[yah.index >= nbp.index.min()]
    ycl = check_nonpositive(yah["close"], "Yahoo", "eurpln_yahoo", log)
    outside = yah.index[((yah["close"] > yah["high"] + 1e-9) | (yah["close"] < yah["low"] - 1e-9)).to_numpy()]
    log.add_runs(outside, yah.index, "Yahoo", "eurpln_yahoo", "close_outside_high_low", "info", "kept (cross-check series only)", "close outside the bar's high-low range")

    r_nbp, r_ecb, r_yah = _logret(nbp), _logret(ecb), _logret(ycl)
    nbp_clean, t_nbp = handle_outliers(nbp, {"ecb": r_ecb, "yahoo": r_yah}, q, "NBP", "eurpln_nbp", log)
    yah_clean, t_yah = handle_outliers(ycl, {"nbp": r_nbp, "ecb": r_ecb}, q, "Yahoo", "eurpln_yahoo", log)

    fx = pd.concat({"eurpln_nbp": nbp_clean, "eurpln_ecb": ecb, "eurpln_yahoo": yah_clean}, axis=1, sort=True)
    fx = fx[fx.index >= nbp.index.min()]
    rows = []
    tol = float(q["fx_cross_source_tol"])
    for other in ("eurpln_ecb", "eurpln_yahoo"):
        both = fx[["eurpln_nbp", other]].dropna()
        rel = (both[other] / both["eurpln_nbp"] - 1).abs()
        breach = rel.index[(rel > tol).to_numpy()]
        log.add_runs(breach, both.index, "NBP vs " + other.split("_")[1].upper(), other, "cross_source_gap", "warning",
                     "kept; NBP remains primary",
                     f"relative gap to NBP fixing above {tol:.1%}; sources are observed at different times of day "
                     "(NBP late-morning Warsaw fixing, ECB afternoon reference rate, Yahoo end-of-day quote), so gaps on volatile days are expected")
        rows.append({"comparison": f"{other} vs eurpln_nbp", "common_days": len(both), "median_abs_rel_gap": rel.median(),
                     "p99_abs_rel_gap": rel.quantile(0.99), "max_abs_rel_gap": rel.max(), "days_above_tol": len(breach),
                     "return_correlation": _logret(both["eurpln_nbp"]).corr(_logret(both[other]))})
    outliers = pd.concat([t_nbp.assign(series="eurpln_nbp"), t_yah.assign(series="eurpln_yahoo")], ignore_index=True)
    return fx, pd.DataFrame(rows), outliers


# --------------------------------------------------------------------------- CO2


def build_co2(cfg: Config, log: IssueLog, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EU ETS primary auction clearing prices, one row per auction date."""
    raw = sources.fetch_co2(cfg, offline)
    counts = raw["platform"].value_counts()
    log.add(INSTRAT_EEX, "eua_auction", "auction_platforms", "info", "all platforms kept",
            "series mixes auction platform codes " + ", ".join(f"{k} ({v} auctions)" for k, v in counts.items()),
            raw["auction_date"].min(), raw["auction_date"].max(), len(raw))
    multi = raw.groupby("auction_date").size()
    for d in multi.index[(multi > 1).to_numpy()]:
        log.add(INSTRAT_EEX, "eua_auction", "multiple_auctions_same_day", "info", "volume weighted average price",
                "more than one auction on the same date", d, n_obs=int(multi[d]))
    g = raw.groupby("auction_date")
    co2 = pd.DataFrame({
        "price_eur_t": g.apply(lambda x: np.average(x["price_eur_t"], weights=x["volume_t"]) if x["volume_t"].sum() > 0 else x["price_eur_t"].mean()),
        "volume_t": g["volume_t"].sum(),
    })
    co2.index.name = "date"
    co2["price_eur_t"] = check_nonpositive(co2["price_eur_t"], INSTRAT_EEX, "eua_auction", log)
    gap = co2.index.to_series().diff().dt.days
    long_gaps = gap[gap > int(cfg["quality"]["co2_ffill_max_days"])]
    for d, n in long_gaps.items():
        log.add(INSTRAT_EEX, "eua_auction", "auction_gap", "info", "panel value left missing beyond carry limit",
                f"{int(n)} calendar days since previous auction (auction calendar pauses, e.g. year end and August)", d)
    clean, table = handle_outliers(co2["price_eur_t"], {}, cfg["quality"], INSTRAT_EEX, "eua_auction", log)
    co2["price_eur_t"] = clean
    return co2, table


# --------------------------------------------------------------------------- panel


def _asof(series: pd.Series, index: pd.DatetimeIndex, max_days: int) -> tuple[pd.Series, pd.Series]:
    """Last available value at or before each date, if at most ``max_days`` old."""
    s = series.dropna()
    pos = s.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    src = pd.DatetimeIndex(np.where(pos >= 0, s.index.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT")))
    age = (index - src).days
    ok = (pos >= 0) & (age <= max_days)
    return pd.Series(np.where(ok, vals, np.nan), index=index), pd.Series(np.where(ok, age, np.nan), index=index)


def build_panel(cfg: Config, ttf: pd.DataFrame, tge: pd.DataFrame, fx: pd.DataFrame, co2: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    """Business-day panel on the TTF (c2) trading calendar."""
    q = cfg["quality"]
    starts = {"ttf_c1": ttf["c1"].first_valid_index(), "ttf_c2": ttf["c2"].first_valid_index(),
              "tge": tge["price_pln_mwh"].first_valid_index(), "eurpln_nbp": fx["eurpln_nbp"].first_valid_index()}
    ends = {"ttf_c2": ttf["c2"].last_valid_index(), "tge": tge["price_pln_mwh"].last_valid_index(),
            "eurpln_nbp": fx["eurpln_nbp"].last_valid_index()}
    start = max(starts.values()) if cfg["sample"]["start"] == "auto" else pd.Timestamp(cfg["sample"]["start"])
    end = min(ends.values())
    idx = ttf.index[(ttf.index >= start) & (ttf.index <= end)]

    p = pd.DataFrame(index=idx)
    p.index.name = "date"
    for c in ["c1", "c2", "is_roll", "roll_status", "r_c1_adj", "r_c2_adj"]:
        p["ttf_" + c] = ttf[c].reindex(idx)
    risk = cfg["ttf"]["risk_series"]
    p["ttf_risk_ret"] = p[f"ttf_r_{risk}_adj"]

    p["tge_pln"] = tge["price_pln_mwh"].reindex(idx)
    miss = idx[p["tge_pln"].isna().to_numpy()]
    log.add_runs(miss, idx, INSTRAT, "panel.tge_pln", "panel_missing", "warning", "left missing", "no TGE trade on a TTF trading day")
    p["tge_ret"] = np.log(p["tge_pln"] / p["tge_pln"].shift(1))

    p["eurpln"], age = _asof(fx["eurpln_nbp"], idx, int(q["fx_ffill_max_days"]))
    carried = idx[(age > 0).to_numpy()]
    log.add_runs(carried, idx, "NBP", "panel.eurpln", "fx_carried_forward", "info", "previous NBP fixing used",
                 f"no NBP fixing on this TTF trading day (Polish holiday); carried forward at most {q['fx_ffill_max_days']} days")
    log.add_runs(idx[p["eurpln"].isna().to_numpy()], idx, "NBP", "panel.eurpln", "panel_missing", "warning", "left missing", "no NBP fixing within carry limit")
    p["eurpln_ret"] = np.log(p["eurpln"] / p["eurpln"].shift(1))

    p["tge_eur"] = p["tge_pln"] / p["eurpln"]
    p["basis_tge_ttfc1_eur"] = p["tge_eur"] - p["ttf_c1"]

    p["eua_eur_t"], p["eua_age_days"] = _asof(co2["price_eur_t"], idx, int(q["co2_ffill_max_days"]))
    log.add_runs(idx[p["eua_eur_t"].isna().to_numpy()], idx, INSTRAT_EEX, "panel.eua_eur_t", "panel_missing", "info",
                 "left missing", f"no auction within {q['co2_ffill_max_days']} calendar days")
    return p


# --------------------------------------------------------------------------- outputs


def _coverage(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        rows.append({"column": c, "non_missing": int(s.notna().sum()), "missing": int(s.isna().sum()),
                     "first": s.first_valid_index().date() if s.notna().any() else None,
                     "last": s.last_valid_index().date() if s.notna().any() else None})
    return pd.DataFrame(rows)


def run(cfg: Config, offline: bool = False) -> Stage1Result:
    """Run the full Stage 1 build and write all outputs."""
    log = IssueLog()
    ttf, rolls, ttf_out = build_ttf(cfg, log)
    tge, tge_out = build_tge(cfg, log, offline, ttf[f"r_{cfg['ttf']['risk_series']}_adj"])
    fx, fx_cmp, fx_out = build_fx(cfg, log, offline)
    co2, co2_out = build_co2(cfg, log, offline)
    panel = build_panel(cfg, ttf, tge, fx, co2, log)

    clean, out = cfg.path("clean_dir"), cfg.path("outputs_dir")
    ttf.to_csv(clean / "ttf.csv")
    rolls.to_csv(clean / "ttf_roll_schedule.csv", index=False)
    tge.to_csv(clean / "tge_gas_da.csv")
    fx.to_csv(clean / "eurpln.csv")
    co2.to_csv(clean / "eua_auctions.csv")
    panel.to_csv(clean / "panel.csv")

    src = pd.DataFrame([
        {"series": "ttf_c1", "source": "Investing.com TFAc1 (CME listing)", "unit": "EUR/MWh", "rows": ttf.attrs["c1_rows_in_source"], "first": ttf["c1"].first_valid_index().date(), "last": ttf["c1"].last_valid_index().date()},
        {"series": "ttf_c2", "source": "Investing.com TFMBMc2 (ICE Endex)", "unit": "EUR/MWh", "rows": int(ttf["c2"].notna().sum()), "first": ttf.index[0].date(), "last": ttf.index[-1].date()},
        {"series": "tge_gas_da", "source": "Instrat API, data from TGE", "unit": "PLN/MWh", "rows": int(tge["price_pln_mwh"].notna().sum()), "first": tge.index.min().date(), "last": tge.index.max().date()},
        {"series": "eurpln_nbp", "source": "NBP table A", "unit": "PLN per EUR", "rows": int(fx["eurpln_nbp"].notna().sum()), "first": fx["eurpln_nbp"].first_valid_index().date(), "last": fx["eurpln_nbp"].last_valid_index().date()},
        {"series": "eurpln_ecb", "source": "ECB reference rate", "unit": "PLN per EUR", "rows": int(fx["eurpln_ecb"].notna().sum()), "first": fx["eurpln_ecb"].first_valid_index().date(), "last": fx["eurpln_ecb"].last_valid_index().date()},
        {"series": "eurpln_yahoo", "source": "Yahoo EURPLN=X", "unit": "PLN per EUR", "rows": int(fx["eurpln_yahoo"].notna().sum()), "first": fx["eurpln_yahoo"].first_valid_index().date(), "last": fx["eurpln_yahoo"].last_valid_index().date()},
        {"series": "eua_auction", "source": "Instrat API, data from EEX", "unit": "EUR/tCO2", "rows": int(co2["price_eur_t"].notna().sum()), "first": co2.index.min().date(), "last": co2.index.max().date()},
    ])
    outlier_tbl = pd.concat([ttf_out, tge_out.assign(series="tge_gas_da"), fx_out, co2_out.assign(series="eua_auction")], ignore_index=True)
    outlier_summary = (outlier_tbl.groupby(["series", "status"]).size().unstack(fill_value=0).reset_index()
                       if not outlier_tbl.empty else pd.DataFrame())
    coverage = _coverage(panel, [c for c in panel.columns if c not in ("ttf_is_roll", "ttf_roll_status")])
    tables = {
        "source_summary": src,
        "issue_summary": log.summary(),
        "roll_validation": roll_validation(ttf[ttf["c1"].notna() | ttf["is_roll"]]),
        "fx_cross_source": fx_cmp,
        "outlier_summary": outlier_summary,
        "panel_coverage": coverage,
    }
    log.to_frame().to_csv(out / "data_quality_log.csv", index=False)
    for name, t in tables.items():
        t.to_csv(out / f"stage1_{name}.csv", index=False)
    from gasrisk.data.report import write_stage1_report

    write_stage1_report(cfg, tables, panel, log)
    return Stage1Result(ttf, tge, fx, co2, panel, log, tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: data acquisition and quality.")
    parser.add_argument("--offline", action="store_true", help="rebuild from stored raw files without downloading")
    args = parser.parse_args()
    cfg = load_config()
    res = run(cfg, offline=args.offline)
    s = res.log.to_frame()
    print(f"Panel: {len(res.panel)} rows, {res.panel.index[0].date()} to {res.panel.index[-1].date()}")
    print(f"Issue log: {len(s)} entries ({(s.severity == 'warning').sum()} warning, {(s.severity == 'error').sum()} error)")
    print(f"Report: {cfg.root / cfg['paths']['outputs_dir'] / 'stage1_report.md'}")


if __name__ == "__main__":
    main()
