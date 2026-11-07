"""Business-day calendars and TTF contract expiry dates.

TTF expiry rule (ICE Endex Dutch TTF futures; CME Dutch TTF calendar month
futures use the same London business-day convention): trading ceases two UK
business days before the first calendar day of the delivery month. The first
trading day after expiry is the roll day of a continuous front-month series.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import pandas as pd


@lru_cache(maxsize=8)
def _holiday_set(country: str, subdiv: str | None, first_year: int, last_year: int) -> frozenset[pd.Timestamp]:
    hol = holidays.country_holidays(country, subdiv=subdiv, years=range(first_year, last_year + 1))
    return frozenset(pd.Timestamp(d) for d in hol)


def business_days(start: str | pd.Timestamp, end: str | pd.Timestamp, country: str, subdiv: str | None = None) -> pd.DatetimeIndex:
    """Weekdays between ``start`` and ``end`` (inclusive) that are not public holidays."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hol = _holiday_set(country, subdiv, start.year - 1, end.year + 1)
    days = pd.bdate_range(start, end)
    return days[~days.isin(list(hol))]


def uk_business_days(start: str | pd.Timestamp, end: str | pd.Timestamp, subdiv: str = "ENG") -> pd.DatetimeIndex:
    """UK (England and Wales) business days."""
    return business_days(start, end, "GB", subdiv)


def pl_business_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Polish business days (NBP fixing days, absent ad hoc closures)."""
    return business_days(start, end, "PL")


def ttf_expiry(delivery_month: pd.Period, n_days_before: int = 2, subdiv: str = "ENG") -> pd.Timestamp:
    """Last trading day of the TTF contract for ``delivery_month``.

    Counts back ``n_days_before`` UK business days from the first calendar day
    of the delivery month (that day itself is not counted).
    """
    first = delivery_month.start_time.normalize()
    window = uk_business_days(first - pd.Timedelta(days=20), first - pd.Timedelta(days=1), subdiv)
    if len(window) < n_days_before:
        raise ValueError(f"Not enough business days before {first.date()}.")
    return window[-n_days_before]


def roll_schedule(trading_days: pd.DatetimeIndex, n_days_before: int = 2, subdiv: str = "ENG") -> pd.DataFrame:
    """Expiry and roll dates covering ``trading_days``.

    The roll date is the first observed trading day strictly after the expiry
    of the contract that was the front month. On that day the front month
    becomes the contract that was second month the day before.

    Returns:
        DataFrame with columns ``expiring_month``, ``expiry``, ``roll_date``.
    """
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    months = pd.period_range(trading_days[0].to_period("M"), (trading_days[-1] + pd.offsets.MonthBegin(2)).to_period("M"), freq="M")
    rows = []
    for m in months:
        exp = ttf_expiry(m, n_days_before, subdiv)
        after = trading_days[trading_days > exp]
        if exp < trading_days[0] or len(after) == 0:
            continue
        rows.append({"expiring_month": str(m), "expiry": exp, "roll_date": after[0]})
    return pd.DataFrame(rows)
