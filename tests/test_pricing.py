"""Tests for Stage 10 Black-76 pricing."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.pricing import (REFERENCE_EXAMPLES, black76, fd_greeks, greeks, implied_vol, option_expiry,
                             second_month, zero_rate)


@pytest.mark.parametrize("e", REFERENCE_EXAMPLES)
def test_published_reference_examples(e) -> None:
    v = float(black76(e["F"], e["K"], e["T"], e["sigma"], e["r"], e["kind"], True))
    assert round(v, e["decimals"]) == e["published"]


@pytest.mark.parametrize("disc", [True, False])
@pytest.mark.parametrize("K", [60.0, 85.0, 120.0])
def test_put_call_parity(disc, K) -> None:
    F, T, s, r = 85.0, 0.12, 0.6, 0.025
    c, p = black76(F, K, T, s, r, "call", disc), black76(F, K, T, s, r, "put", disc)
    df = np.exp(-r * T) if disc else 1.0
    assert c - p == pytest.approx(df * (F - K), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("disc", [True, False])
def test_analytic_greeks_match_finite_differences(kind, disc) -> None:
    a = greeks(80.0, 85.0, 0.2, 0.5, 0.03, kind, disc)
    n = fd_greeks(80.0, 85.0, 0.2, 0.5, 0.03, kind, disc, 1e-4)
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        assert float(a[g]) == pytest.approx(n[g], rel=1e-4, abs=1e-7)


def test_futures_style_is_undiscounted_and_homogeneous() -> None:
    assert black76(80, 80, 1.0, 0.4, 0.05, "call", False) > black76(80, 80, 1.0, 0.4, 0.05, "call", True)
    assert black76(160, 170, 0.3, 0.4, 0.0, "call", False) == pytest.approx(2 * black76(80, 85, 0.3, 0.4, 0.0, "call", False))


def test_implied_vol_round_trip() -> None:
    v = float(black76(85, 95, 0.15, 0.73, 0.02, "call", True))
    assert implied_vol(v, 85, 95, 0.15, 0.02, "call", True) == pytest.approx(0.73, abs=1e-9)


def test_ice_option_expiry_rule() -> None:
    # Jan 2027: 27 Dec 2026 is a Sunday, 25 and 28 Dec are UK holidays -> Thursday 24 Dec.
    assert option_expiry(pd.Period("2027-01", "M")) == pd.Timestamp("2026-12-24")
    # Mar 2027: 24 Feb 2027 is a Wednesday business day.
    assert option_expiry(pd.Period("2027-03", "M")) == pd.Timestamp("2027-02-24")


def test_second_month_contract() -> None:
    # On 6 Nov 2026 the Nov contract has expired, Dec is the front month, Jan 2027 the second month.
    assert second_month(pd.Timestamp("2026-11-06")) == pd.Period("2027-01", "M")


def test_zero_rate_interpolation_is_flat_outside_grid() -> None:
    row = pd.Series([2.0, 3.0], index=[0.25, 1.0])
    assert zero_rate(row, 0.1) == pytest.approx(0.02)
    assert zero_rate(row, 0.625) == pytest.approx(0.025)
    assert zero_rate(row, 5.0) == pytest.approx(0.03)
