"""Data quality checks and the issue log.

Principles:

* Nothing is filled, dropped or changed silently: every action is logged with
  a reason in ``outputs/data_quality_log.csv``.
* A large move is presumed real. A price is only removed when a second series
  contradicts the move **and** the move reverses on the next observation
  (the signature of a bad print).
* The committed log holds dates, returns and diagnostics, never raw price
  levels, because the underlying data may not be redistributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "issue_id", "date_start", "date_end", "n_obs", "source", "series", "check",
    "severity", "action", "reason", "detail",
]
SEVERITIES = ("info", "warning", "error")


@dataclass
class IssueLog:
    """Accumulates data quality issues; one entry per event or contiguous run."""

    entries: list[dict] = field(default_factory=list)

    def add(
        self,
        source: str,
        series: str,
        check: str,
        severity: str,
        action: str,
        reason: str,
        date_start: pd.Timestamp | None = None,
        date_end: pd.Timestamp | None = None,
        n_obs: int = 1,
        detail: str = "",
    ) -> None:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        self.entries.append(
            {
                "date_start": None if date_start is None else pd.Timestamp(date_start).date(),
                "date_end": None if date_end is None and date_start is None else pd.Timestamp(date_end if date_end is not None else date_start).date(),
                "n_obs": int(n_obs),
                "source": source,
                "series": series,
                "check": check,
                "severity": severity,
                "action": action,
                "reason": reason,
                "detail": detail,
            }
        )

    def add_runs(
        self,
        dates: pd.DatetimeIndex,
        reference_index: pd.DatetimeIndex,
        source: str,
        series: str,
        check: str,
        severity: str,
        action: str,
        reason: str,
        detail: str = "",
        group_by_index: bool = True,
    ) -> None:
        """Log ``dates`` grouped into runs that are consecutive in ``reference_index``.

        With ``group_by_index=False`` each date is its own entry.
        """
        for start, end, n in contiguous_runs(dates, reference_index) if group_by_index else [(d, d, 1) for d in dates]:
            self.add(source, series, check, severity, action, reason, start, end, n, detail)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.entries, columns=LOG_COLUMNS[1:])
        df.insert(0, "issue_id", range(1, len(df) + 1))
        return df

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return pd.DataFrame(columns=["series", "check", "severity", "action", "entries", "observations"])
        return (
            df.groupby(["series", "check", "severity", "action"], dropna=False)
            .agg(entries=("issue_id", "size"), observations=("n_obs", "sum"))
            .reset_index()
            .sort_values(["series", "severity", "check"])
        )


def contiguous_runs(dates: pd.DatetimeIndex, reference_index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Group ``dates`` into runs that are adjacent positions in ``reference_index``."""
    if len(dates) == 0:
        return []
    ref = pd.DatetimeIndex(reference_index).sort_values()
    pos = ref.get_indexer(pd.DatetimeIndex(dates).sort_values())
    if (pos < 0).any():
        raise ValueError("All dates must be present in the reference index.")
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = prev = pos[0]
    for p in pos[1:]:
        if p != prev + 1:
            runs.append((ref[start], ref[prev], int(prev - start + 1)))
            start = p
        prev = p
    runs.append((ref[start], ref[prev], int(prev - start + 1)))
    return runs


# --------------------------------------------------------------------------- robust statistics


def rolling_robust_scale(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling robust standard deviation (1.4826 x MAD), using only past data.

    The value at t is estimated from observations up to t-1, so a spike does
    not inflate its own scale (no look-ahead).
    """
    med = x.rolling(window, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median()
    return (1.4826 * mad).shift(1)


def robust_zscores(r: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Robust z-score of each return against the trailing median and MAD."""
    med = r.rolling(window, min_periods=min_periods).median().shift(1)
    return (r - med) / rolling_robust_scale(r, window, min_periods)


# --------------------------------------------------------------------------- generic checks


def check_duplicates(df: pd.DataFrame, source: str, series: str, log: IssueLog) -> pd.DataFrame:
    """Drop duplicated index labels (keep first) and log them."""
    dup = df.index.duplicated(keep="first")
    for d in df.index[dup]:
        log.add(source, series, "duplicate_date", "warning", "dropped duplicate row (kept first)", "two rows share one date", d)
    return df[~dup]


def check_nonpositive(s: pd.Series, source: str, series: str, log: IssueLog) -> pd.Series:
    """Set non-positive prices to NaN (log returns are undefined) and log them."""
    bad = s.index[(s <= 0).to_numpy()]
    for d in bad:
        log.add(source, series, "non_positive_price", "error", "set to NaN", "price <= 0 cannot enter log returns", d)
    out = s.copy()
    out.loc[bad] = np.nan
    return out


def check_missing_days(
    index: pd.DatetimeIndex,
    expected: pd.DatetimeIndex,
    source: str,
    series: str,
    log: IssueLog,
    calendar_name: str,
    long_run: int = 3,
) -> pd.DatetimeIndex:
    """Log expected calendar days absent from ``index`` (grouped into runs)."""
    expected = expected[(expected >= index.min()) & (expected <= index.max())]
    missing = expected.difference(index)
    for start, end, n in contiguous_runs(missing, expected):
        sev = "warning" if n >= long_run else "info"
        log.add(source, series, "missing_day", sev, "left missing (no fill)",
                f"expected {calendar_name} day not in source; returns span the gap", start, end, n)
    return missing


def check_extra_days(index: pd.DatetimeIndex, expected: pd.DatetimeIndex, source: str, series: str, log: IssueLog, calendar_name: str) -> pd.DatetimeIndex:
    """Log weekday observations that fall on a holiday of ``calendar_name`` (kept)."""
    weekday = index[index.day_of_week < 5]
    extra = weekday.difference(expected)
    for d in extra:
        log.add(source, series, "holiday_observation", "info", "kept",
                f"price on a {calendar_name} holiday; venue calendar differs", d)
    return extra


def check_ohlc(df: pd.DataFrame, source: str, series: str, log: IssueLog) -> None:
    """Flag settlement-only rows and settlements outside the traded high-low range.

    Both are expected for thinly traded futures: the settlement price is set by
    the exchange and can differ from the last trade. Only ``close`` is used
    downstream, so these are informational.
    """
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    log.add_runs(df.index[flat.to_numpy()], df.index, source, series, "settlement_only_row", "info", "kept; only close is used",
                 "O=H=L=C, typically with empty volume: no traded range, settlement only")
    outside = ~flat & ((df["close"] > df["high"] + 1e-9) | (df["close"] < df["low"] - 1e-9))
    log.add_runs(df.index[outside.to_numpy()], df.index, source, series, "close_outside_high_low", "info", "kept; only close is used",
                 "settlement outside the traded range, common on thin days")
    hl = df["high"] < df["low"]
    log.add_runs(df.index[hl.to_numpy()], df.index, source, series, "high_below_low", "warning", "kept; high/low not used", "inconsistent OHLC")


def check_stale(s: pd.Series, min_run: int, source: str, series: str, log: IssueLog) -> None:
    """Log runs of at least ``min_run`` consecutive unchanged prices."""
    same = s.diff().eq(0)
    grp = (~same).cumsum()
    runs = same.groupby(grp).sum()
    for g, n in runs[runs + 1 >= min_run].items():
        dates = s.index[grp == g]
        log.add(source, series, "stale_price_run", "warning", "kept; flagged",
                f"{int(n) + 1} consecutive identical prices", dates[0], dates[-1], int(n) + 1)


def classify_outliers(
    r: pd.Series,
    z: pd.Series,
    others: dict[str, pd.Series],
    z_threshold: float,
    confirm_ratio: float,
    contradict_ratio: float,
    reversal_ratio: float,
) -> pd.DataFrame:
    """Classify large returns as confirmed, contradicted or unconfirmed.

    * confirmed: at least one other series moved the same way by at least
      ``confirm_ratio`` of the move.
    * contradicted: every available other series moved less than
      ``contradict_ratio`` of the move (or the other way) **and** the series
      reverses by at least ``reversal_ratio`` of the move on the next observation.
    * unconfirmed: everything else (kept, flagged).
    """
    flagged = r.index[(z.abs() > z_threshold).to_numpy()]
    rows = []
    for d in flagged:
        move = r.at[d]
        other_moves = {k: o.get(d, np.nan) for k, o in others.items()}
        avail = {k: v for k, v in other_moves.items() if pd.notna(v)}
        confirmed = any(np.sign(v) == np.sign(move) and abs(v) >= confirm_ratio * abs(move) for v in avail.values())
        weak = bool(avail) and all(np.sign(v) != np.sign(move) or abs(v) < contradict_ratio * abs(move) for v in avail.values())
        pos = r.index.get_loc(d)
        nxt = r.iloc[pos + 1] if pos + 1 < len(r) else np.nan
        reverses = pd.notna(nxt) and np.sign(nxt) == -np.sign(move) and abs(nxt) >= reversal_ratio * abs(move)
        status = "confirmed" if confirmed else ("contradicted" if weak and reverses else "unconfirmed")
        rows.append({"date": d, "ret": move, "z": z.at[d], "next_ret": nxt, "status": status,
                     **{f"ret_{k}": v for k, v in other_moves.items()}})
    return pd.DataFrame(rows)


def handle_outliers(
    prices: pd.Series,
    others: dict[str, pd.Series],
    qcfg: dict,
    source: str,
    series: str,
    log: IssueLog,
    window: int | None = None,
    exclude: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Detect, classify and log return outliers; remove contradicted bad prints.

    Args:
        exclude: Dates whose returns are not tested (e.g. futures roll days,
            whose raw return contains the calendar spread by construction and
            is handled by the roll adjustment instead).

    Returns the cleaned price series and the classification table.
    """
    p = prices.dropna()
    r = np.log(p / p.shift(1))
    if exclude is not None:
        r[r.index.isin(exclude)] = np.nan
    w = window or int(qcfg["outlier_window"])
    z = robust_zscores(r, w, int(qcfg["outlier_min_periods"]))
    table = classify_outliers(r, z, others, float(qcfg["outlier_z"]), float(qcfg["confirm_ratio"]),
                              float(qcfg["contradict_ratio"]), float(qcfg["reversal_ratio"]))
    out = prices.copy()
    for row in table.itertuples(index=False):
        others_txt = ", ".join(f"{k}={getattr(row, 'ret_' + k):.4f}" for k in others if pd.notna(getattr(row, "ret_" + k)))
        detail = f"log return {row.ret:.4f}, robust z {row.z:.1f}, next {row.next_ret:.4f}" + (f"; other series {others_txt}" if others_txt else "")
        if row.status == "confirmed":
            log.add(source, series, "return_outlier", "info", "kept", "large move confirmed by a second series", row.date, detail=detail)
        elif row.status == "unconfirmed":
            log.add(source, series, "return_outlier", "warning", "kept; flagged",
                    "large move not confirmed (no second series moved comparably) but not contradicted", row.date, detail=detail)
        else:
            out.loc[row.date] = np.nan
            log.add(source, series, "return_outlier", "warning", "price set to NaN",
                    "bad print: second series contradicts the move and it reverses next day", row.date, detail=detail)
    return out, table
