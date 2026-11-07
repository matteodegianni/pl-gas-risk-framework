"""Offline tests for the source parsers (no network)."""

import io

import pandas as pd
import pytest

from gasrisk.data.sources import parse_instrat_co2, parse_instrat_gas, parse_investing_csv, parse_nbp, parse_volume

INVESTING = '﻿"Date","Price","Open","High","Low","Vol.","Change %"\n' \
    '"11/06/2026","85.588","90.695","90.700","84.950","0.17K","-5.61%"\n' \
    '"11/05/2026","1,090.679","88.500","90.800","88.215","1.2M","2.95%"\n' \
    '"10/23/2017","18.090","18.090","18.090","18.090","","-44.81%"\n'


def test_parse_investing_csv_handles_bom_thousands_volume_and_order() -> None:
    df = parse_investing_csv(io.StringIO(INVESTING), "%m/%d/%Y")
    assert list(df.index) == [pd.Timestamp("2017-10-23"), pd.Timestamp("2026-11-05"), pd.Timestamp("2026-11-06")]
    assert df.loc["2026-11-05", "close"] == pytest.approx(1090.679)
    assert df.loc["2026-11-06", "volume"] == pytest.approx(170.0)
    assert df.loc["2026-11-05", "volume"] == pytest.approx(1.2e6)
    assert pd.isna(df.loc["2017-10-23", "volume"])
    assert df.loc["2017-10-23", "change_pct_reported"] == pytest.approx(-0.4481)


def test_parse_investing_csv_rejects_wrong_columns() -> None:
    with pytest.raises(ValueError):
        parse_investing_csv(io.StringIO("Date,Close\n01/01/2020,1\n"), "%m/%d/%Y")


def test_parse_volume() -> None:
    assert parse_volume(pd.Series(["0.30K", "56.60K", "", "12"])).tolist()[:2] == pytest.approx([300.0, 56600.0])


def test_instrat_gas_trade_date_is_delivery_minus_one_day() -> None:
    payload = [{"date": "2026-11-03T00:00:00Z", "indeks": "tgegasda", "price": 460.5, "volume": 116832.0},
               {"date": "2026-11-02T00:00:00Z", "indeks": "tgegasda", "price": 440.4, "volume": 100000.0}]
    df = parse_instrat_gas(payload, trade_date_lag_days=1)
    assert df["delivery_date"].tolist() == [pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03")]
    assert df["trade_date"].tolist() == [pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-02")]


def test_instrat_co2_normalises_auction_timestamp() -> None:
    df = parse_instrat_co2([{"price": 63.5, "date": "2024-01-15T10:00:36Z", "indeks": "t3pa", "volume": 3099500}])
    assert df.at[0, "auction_date"] == pd.Timestamp("2024-01-15")


def test_parse_nbp_concatenates_chunks() -> None:
    chunks = [{"rates": [{"effectiveDate": "2026-11-05", "mid": 4.3916}]}, {"rates": [{"effectiveDate": "2026-11-06", "mid": 4.3925}]}]
    s = parse_nbp(chunks)
    assert s.loc["2026-11-06"] == pytest.approx(4.3925) and len(s) == 2
