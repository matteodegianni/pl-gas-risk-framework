"""Roll-adjusted TTF returns from the first (c1) and second (c2) month series.

A continuous futures series switches contract on the roll day, so its raw
return that day mixes a genuine price move with the calendar spread between
two delivery months. The spread is not a P&L anyone earned.

On a roll day t the contract that is front month today was the second month
yesterday, so the return of one and the same contract is::

    r_roll(t) = ln(c1(t) / c2(t-1))

This is the correct return on the roll day for both continuous series used here:

* ``c1`` front month: the new front month's move from yesterday.
* ``c2`` held to roll: a position in yesterday's second month is, today, the
  front month, so its P&L that day is the same ``r_roll``; the position is then
  rolled into the new second month at today's close.

c1 (CME listing) and c2 (ICE Endex) come from different venues, so the
adjusted roll-day return is validated: if it is extreme relative to normal
c2 volatility, the two series probably did not roll on the same day and the
return is excluded rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gasrisk.data.quality import IssueLog, rolling_robust_scale


def roll_adjusted_returns(
    c1: pd.Series,
    c2: pd.Series,
    roll_dates: pd.DatetimeIndex,
    roll_z: float,
    scale_window: int,
    log: IssueLog | None = None,
) -> pd.DataFrame:
    """Compute raw and roll-adjusted log returns on the c2 trading calendar.

    Args:
        c1: Front-month settlement prices (may start later or miss days).
        c2: Second-month settlement prices; its index defines the calendar.
        roll_dates: Roll days (first trading day after each expiry).
        roll_z: Robust z-score above which an adjusted roll return is excluded.
        scale_window: Window (observations) for the robust scale of non-roll c2 returns.
        log: Optional issue log.

    Returns:
        DataFrame indexed like ``c2`` with prices, raw returns, ``is_roll``,
        ``r_roll``, ``roll_z``, ``roll_status`` and adjusted returns
        ``r_c1_adj`` and ``r_c2_adj``.
    """
    idx = c2.index
    df = pd.DataFrame({"c1": c1.reindex(idx), "c2": c2})
    df["r_c1_raw"] = np.log(df["c1"] / df["c1"].shift(1))
    df["r_c2_raw"] = np.log(df["c2"] / df["c2"].shift(1))
    df["is_roll"] = idx.isin(roll_dates)
    df["r_roll"] = np.where(df["is_roll"], np.log(df["c1"] / df["c2"].shift(1)), np.nan)

    normal = df["r_c2_raw"].where(~df["is_roll"])
    scale = rolling_robust_scale(normal, window=scale_window, min_periods=max(20, scale_window // 3))
    df["roll_z"] = df["r_roll"] / scale

    status = pd.Series(pd.NA, index=idx, dtype="string")
    roll_mask = df["is_roll"].to_numpy()
    no_c1 = roll_mask & df["r_roll"].isna().to_numpy()
    bad = roll_mask & (df["roll_z"].abs() > roll_z).to_numpy()
    status[roll_mask] = "adjusted"
    status[no_c1] = "excluded_no_c1"
    status[bad] = "excluded_misaligned"
    df["roll_status"] = status

    keep = df["roll_status"].eq("adjusted").fillna(False).to_numpy()
    for col, raw in (("r_c1_adj", "r_c1_raw"), ("r_c2_adj", "r_c2_raw")):
        adj = df[raw].copy()
        adj[roll_mask] = np.nan
        adj[keep] = df.loc[keep, "r_roll"]
        df[col] = adj

    if log is not None:
        dropped = idx[no_c1]
        if len(dropped):
            log.add_runs(
                dates=dropped, reference_index=idx, source="Investing.com", series="ttf_c2", check="roll_unadjustable",
                severity="info", action="return set to NaN",
                reason="roll day before c1 history starts (or c1 missing), so the same-contract return cannot be formed",
                group_by_index=False,
            )
        for d in idx[bad]:
            log.add(
                source="Investing.com", series="ttf_c1/ttf_c2", check="roll_misaligned", severity="warning",
                date_start=d, action="roll-day return set to NaN in both adjusted series",
                reason=f"adjusted roll return robust z={df.at[d, 'roll_z']:.2f} exceeds {roll_z}; c1 (CME) and c2 (ICE) likely rolled on different days",
                detail=f"r_roll={df.at[d, 'r_roll']:.4f}, raw r_c1={df.at[d, 'r_c1_raw']:.4f}, raw r_c2={df.at[d, 'r_c2_raw']:.4f}",
            )
        n_adj = int(keep.sum())
        if n_adj:
            log.add(
                source="Investing.com", series="ttf_c1/ttf_c2", check="roll_adjusted", severity="info",
                date_start=idx[keep][0], date_end=idx[keep][-1], n_obs=n_adj,
                action="raw roll-day returns replaced by same-contract return ln(c1_t / c2_t-1)",
                reason="removes the calendar spread from returns on contract roll days",
            )
    return df


def roll_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics showing whether the roll adjustment behaves as intended.

    If the roll days are right, raw c1 returns on roll days load on yesterday's
    c2/c1 spread (slope near 1) and adjusted roll returns look like normal-day
    returns.
    """
    spread_prev = np.log(df["c2"] / df["c1"]).shift(1)

    def slope(y: pd.Series, x: pd.Series) -> float:
        ok = y.notna() & x.notna()
        if ok.sum() < 5:
            return float("nan")
        return float(np.polyfit(x[ok], y[ok], 1)[0])

    roll = df["is_roll"]
    with_c1 = df["c1"].notna() & df["c1"].shift(1).notna()
    rows = {
        "roll_days_total": int(roll.sum()),
        "roll_days_with_c1": int((roll & with_c1).sum()),
        "roll_days_adjusted": int(df["roll_status"].eq("adjusted").sum()),
        "roll_days_excluded_misaligned": int(df["roll_status"].eq("excluded_misaligned").sum()),
        "roll_days_excluded_no_c1": int(df["roll_status"].eq("excluded_no_c1").sum()),
        "slope_raw_c1_on_prev_spread_roll_days": slope(df["r_c1_raw"][roll & with_c1], spread_prev[roll & with_c1]),
        "slope_raw_c1_on_prev_spread_other_days": slope(df["r_c1_raw"][~roll & with_c1], spread_prev[~roll & with_c1]),
        "mean_raw_c2_return_roll_days": float(df["r_c2_raw"][roll & with_c1].mean()),
        "mean_adjusted_return_roll_days": float(df["r_roll"][df["roll_status"].eq("adjusted").fillna(False)].mean()),
        "sd_raw_c2_return_roll_days": float(df["r_c2_raw"][roll & with_c1].std()),
        "sd_adjusted_return_roll_days": float(df["r_roll"][df["roll_status"].eq("adjusted").fillna(False)].std()),
        "sd_c2_return_non_roll_days": float(df["r_c2_raw"][~roll & with_c1].std()),
    }
    return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})
