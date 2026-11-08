"""Tests for the Stage 16 pipeline, Airflow DAG definition and dashboard exports."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from gasrisk.config import load_config
from gasrisk.pipeline import DAILY_STEPS

ROOT = load_config().root
EXPORTS = ROOT / "exports"


def _dag():
    spec = importlib.util.spec_from_file_location("gasrisk_daily", ROOT / "airflow" / "dags" / "gasrisk_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dag_steps_match_pipeline() -> None:
    dag = _dag()
    assert dag.STEPS == DAILY_STEPS
    assert set(dag.UPSTREAM) == set(DAILY_STEPS)


def test_dag_dependencies_respect_sequential_order() -> None:
    dag = _dag()
    pos = {s: i for i, s in enumerate(DAILY_STEPS)}
    for s, ups in dag.UPSTREAM.items():
        assert all(pos[u] < pos[s] for u in ups), s


@pytest.mark.skipif(not EXPORTS.exists(), reason="exports not generated")
def test_exports_are_labelled_and_carry_no_price_levels() -> None:
    banned = {"ttf_c1", "ttf_c2", "tge_pln", "tge_eur", "eurpln", "fx", "level_tge_eur", "level_ttf", "level_fx"}
    for p in EXPORTS.glob("*.csv"):
        df = pd.read_csv(p)
        assert (df["data_label"].str.contains("synthetic")).all(), p.name
        assert not banned & set(df.columns), p.name


@pytest.mark.skipif(not EXPORTS.exists(), reason="exports not generated")
def test_fact_keys_exist_in_dimensions() -> None:
    r = lambda n: pd.read_csv(EXPORTS / f"{n}.csv")  # noqa: E731
    dates = set(r("dim_date")["date"])
    for f in ("fact_var", "fact_limit_utilisation", "fact_attribution", "fact_margin_forecast", "fact_risk_contribution"):
        assert set(r(f)["date"]) <= dates, f
    assert set(r("fact_var")["method_key"]) <= set(r("dim_method")["method_key"])
    assert set(r("fact_limit_utilisation")["limit_key"]) <= set(r("dim_limit")["limit_key"])
    assert set(r("fact_stress")["factor_key"]) <= set(r("dim_risk_factor")["factor_key"])
    assert set(r("fact_stress")["scenario_key"]) <= set(r("dim_scenario")["scenario_key"])
    assert set(r("fact_margin_stress")["scenario_key"]) <= set(r("dim_scenario")["scenario_key"])
    assert set(r("fact_attribution")["driver_key"]) <= set(r("dim_driver")["driver_key"])
    assert set(r("fact_credit")["counterparty"]) <= set(r("dim_counterparty")["counterparty"])
    assert r("dim_date")["date"].is_unique
