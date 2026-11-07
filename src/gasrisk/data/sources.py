"""Loaders for every external data source.

Each source has a pure ``parse_*`` function (unit tested offline) and a
``fetch_*`` function that downloads, stores the raw payload unchanged under
``data/raw/<source>/`` and records provenance in ``data/raw/manifest.json``.
With ``offline=True`` the fetchers re-read the stored raw payload instead.

Data licences (see README): TGE prices via Instrat are CC BY-NC 4.0 and TGE
terms also apply; Investing.com exports may not be redistributed. Raw files
are therefore gitignored and never committed.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from gasrisk.config import Config
from gasrisk.data.http import get

# --------------------------------------------------------------------------- provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_provenance(cfg: Config, name: str, path: Path, meta: dict[str, Any]) -> None:
    """Add or replace the manifest entry for one raw file."""
    manifest_path = cfg.path("raw_dir") / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {
        "file": str(path.relative_to(cfg.root)),
        "sha256": _sha256(path),
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _raw_file(cfg: Config, source: str, filename: str) -> Path:
    d = cfg.path("raw_dir") / source
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


# --------------------------------------------------------------------------- Investing.com


def parse_volume(text: pd.Series) -> pd.Series:
    """Convert Investing.com volume strings such as ``0.17K`` or ``1.2M`` to floats (empty -> NaN)."""
    s = text.fillna("").astype(str).str.strip()
    mult = s.str[-1:].map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    num = pd.to_numeric(s.str.rstrip("KMB"), errors="coerce")
    return num * mult


def parse_investing_csv(source: str | Path | io.StringIO, date_format: str) -> pd.DataFrame:
    """Parse an Investing.com historical data export.

    Returns a frame indexed by date (ascending) with columns ``open, high, low,
    close, volume, change_pct_reported``. ``close`` is the settlement price shown
    as "Price" on the site.
    """
    df = pd.read_csv(source, encoding="utf-8-sig", thousands=",", dtype={"Vol.": "string", "Change %": "string"})
    expected = {"Date", "Price", "Open", "High", "Low", "Vol.", "Change %"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Investing export missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Price"], errors="coerce"),
            "volume": parse_volume(df["Vol."]),
            "change_pct_reported": pd.to_numeric(df["Change %"].str.rstrip("%"), errors="coerce") / 100.0,
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(df["Date"], format=date_format), name="date")
    return out.sort_index()


def load_investing(cfg: Config, key: str) -> pd.DataFrame:
    """Load one manual Investing.com export from ``paths.investing_dir``."""
    icfg = cfg["sources"]["investing"]
    path = cfg.path("investing_dir") / icfg["files"][key]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Export the series from Investing.com (English site, daily) "
            f"and save it as {path.name} in {path.parent}."
        )
    df = parse_investing_csv(path, icfg["date_format"])
    record_provenance(cfg, key, path, {"source": "Investing.com manual export", "rows": len(df)})
    return df


# --------------------------------------------------------------------------- Instrat (TGE gas, EEX CO2)


def parse_instrat_gas(payload: list[dict[str, Any]], trade_date_lag_days: int) -> pd.DataFrame:
    """Parse Instrat TGEgasDA records.

    Instrat dates the index by **delivery day**. The TGEgasDA value for delivery
    day D is the volume weighted price of trades on D-1, so ``trade_date`` is
    ``delivery_date - lag``.
    """
    df = pd.DataFrame(payload)
    if df.empty:
        raise ValueError("Instrat gas payload is empty.")
    delivery = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    out = pd.DataFrame(
        {
            "delivery_date": delivery,
            "trade_date": delivery - pd.Timedelta(days=trade_date_lag_days),
            "price_pln_mwh": pd.to_numeric(df["price"], errors="coerce"),
            "volume_mwh": pd.to_numeric(df["volume"], errors="coerce"),
            "index_name": df["indeks"].astype(str),
        }
    )
    return out.sort_values("delivery_date").reset_index(drop=True)


def parse_instrat_co2(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse Instrat EU ETS primary auction records (clearing price EUR/t, auction volume)."""
    df = pd.DataFrame(payload)
    if df.empty:
        raise ValueError("Instrat CO2 payload is empty.")
    ts = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    out = pd.DataFrame(
        {
            "auction_date": ts.dt.normalize(),
            "auction_time_utc": ts,
            "price_eur_t": pd.to_numeric(df["price"], errors="coerce"),
            "volume_t": pd.to_numeric(df["volume"], errors="coerce"),
            "platform": df["indeks"].astype(str),
        }
    )
    return out.sort_values("auction_time_utc").reset_index(drop=True)


def _instrat_fetch(cfg: Config, endpoint: str, name: str, offline: bool) -> list[dict[str, Any]]:
    path = _raw_file(cfg, "instrat", f"{name}.json")
    if offline:
        return json.loads(path.read_text())
    icfg = cfg["sources"]["instrat"]
    url = f"{icfg['base_url'].rstrip('/')}/{endpoint}"
    end = (date.today() + timedelta(days=2)).isoformat()
    params = {"date_from": f"{icfg['start']}T00:00:00Z", "date_to": f"{end}T00:00:00Z"}
    resp = get(url, params, cfg["http"])
    resp.raise_for_status()
    payload = resp.json()
    path.write_text(json.dumps(payload))
    record_provenance(
        cfg,
        name,
        path,
        {
            "source": "Instrat energy API (energy.instrat.pl)",
            "url": url,
            "params": params,
            "rows": len(payload),
            "licence": "CC BY-NC 4.0 (TGE gas prices); CC BY 4.0 unless stated otherwise (other series)",
        },
    )
    return payload


def fetch_tge_gas(cfg: Config, offline: bool = False) -> pd.DataFrame:
    """TGEgasDA daily index from Instrat (data originate from TGE)."""
    payload = _instrat_fetch(cfg, cfg["sources"]["instrat"]["gas_endpoint"], "tge_gas_da", offline)
    return parse_instrat_gas(payload, int(cfg["tge"]["trade_date_lag_days"]))


def fetch_co2(cfg: Config, offline: bool = False) -> pd.DataFrame:
    """EU ETS primary auction clearing prices from Instrat (data from EEX)."""
    payload = _instrat_fetch(cfg, cfg["sources"]["instrat"]["co2_endpoint"], "eua_auctions", offline)
    return parse_instrat_co2(payload)


# --------------------------------------------------------------------------- EUR/PLN


def parse_nbp(payloads: list[dict[str, Any]]) -> pd.Series:
    """Parse a list of NBP table A responses into a EUR/PLN mid-rate series."""
    rows = [(r["effectiveDate"], r["mid"]) for p in payloads for r in p.get("rates", [])]
    s = pd.Series({pd.Timestamp(d): float(m) for d, m in rows}, name="eurpln_nbp").sort_index()
    s.index.name = "date"
    return s


def fetch_nbp(cfg: Config, offline: bool = False) -> pd.Series:
    """NBP EUR/PLN mid fixing, requested in chunks because the API limits range length."""
    path = _raw_file(cfg, "nbp", "eurpln_table_a.json")
    if offline:
        return parse_nbp(json.loads(path.read_text()))
    ncfg = cfg["sources"]["nbp"]
    start, end = pd.Timestamp(ncfg["start"]), pd.Timestamp(date.today())
    payloads: list[dict[str, Any]] = []
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=int(ncfg["chunk_days"]) - 1), end)
        url = f"{ncfg['base_url']}/{cur.date()}/{stop.date()}/"
        resp = get(url, {"format": "json"}, cfg["http"])
        if resp.status_code != 404:  # 404 = no fixing in this window
            resp.raise_for_status()
            payloads.append(resp.json())
        cur = stop + pd.Timedelta(days=1)
    path.write_text(json.dumps(payloads))
    s = parse_nbp(payloads)
    record_provenance(cfg, "eurpln_nbp", path, {"source": "NBP API table A", "url": ncfg["base_url"], "rows": len(s)})
    return s


def parse_ecb_csv(text: str) -> pd.Series:
    """Parse the ECB SDMX csvdata response for EXR D.PLN.EUR.SP00.A."""
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(df["OBS_VALUE"].astype(float).values, index=pd.DatetimeIndex(pd.to_datetime(df["TIME_PERIOD"]), name="date"), name="eurpln_ecb")
    return s.sort_index().dropna()


def fetch_ecb(cfg: Config, offline: bool = False) -> pd.Series:
    """ECB EUR/PLN reference rate."""
    path = _raw_file(cfg, "ecb", "eurpln_exr.csv")
    if not offline:
        ecfg = cfg["sources"]["ecb"]
        resp = get(ecfg["url"], {"format": "csvdata", "startPeriod": ecfg["start"]}, cfg["http"])
        resp.raise_for_status()
        path.write_text(resp.text)
        record_provenance(cfg, "eurpln_ecb", path, {"source": "ECB Data API", "url": ecfg["url"]})
    return parse_ecb_csv(path.read_text())


def fetch_yahoo_eurpln(cfg: Config, offline: bool = False) -> pd.DataFrame:
    """Yahoo Finance EUR/PLN daily OHLC (cross-check only)."""
    path = _raw_file(cfg, "yahoo", "eurpln.csv")
    if not offline:
        import yfinance as yf  # imported lazily: only needed when refreshing

        ycfg = cfg["sources"]["yahoo"]
        hist = yf.Ticker(ycfg["eurpln_ticker"]).history(start=ycfg["start"], auto_adjust=False)
        if hist.empty:
            raise RuntimeError("Yahoo returned no EUR/PLN data.")
        hist.index = pd.DatetimeIndex(hist.index.tz_localize(None).normalize(), name="date")
        hist[["Open", "High", "Low", "Close"]].to_csv(path)
        record_provenance(cfg, "eurpln_yahoo", path, {"source": "Yahoo Finance via yfinance", "ticker": ycfg["eurpln_ticker"], "rows": len(hist)})
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    return df
