"""Tests for Stage 15 input validation."""

import copy
import shutil

import numpy as np
import pandas as pd

from gasrisk.config import Config, load_config
from gasrisk.validation import validate_inputs


def _cfg(tmp_path):
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["paths"] = {**raw["paths"], "clean_dir": "clean", "outputs_dir": "outputs"}
    (tmp_path / "handover").mkdir()
    shutil.copy(base.root / "handover" / "input_schema.yaml", tmp_path / "handover")
    (tmp_path / "config").mkdir()
    shutil.copy(base.root / "config" / "config.yaml", tmp_path / "config")
    cfg = Config(raw=raw, root=tmp_path)
    (cfg.path("outputs_dir") / "data_quality_log.csv").write_text("issue_id\n")
    return cfg


def _write(cfg, n=1600, bad_fx=False):
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(0)
    fx = np.full(n, 4.3)
    if bad_fx:
        fx[10] = 43.0
    tge_pln = 100 + rng.normal(0, 1, n)
    panel = pd.DataFrame({"date": idx, "ttf_c2": 30.0, "ttf_risk_ret": 0.001, "tge_pln": tge_pln, "tge_eur": tge_pln / fx,
                          "eurpln": fx, "eurpln_ret": 0.0})
    panel.to_csv(cfg.path("clean_dir") / "panel.csv", index=False)
    for f in ("garch_conditional_vol.csv", "garch_std_resid.csv"):
        pd.DataFrame({"date": idx, "ttf_t": 1.0}).to_csv(cfg.path("clean_dir") / f, index=False)


def test_clean_inputs_pass(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg)
    res = validate_inputs(cfg)
    assert all(r.passed for r in res if r.severity == "error")


def test_out_of_range_fx_is_caught(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, bad_fx=True)
    failed = {r.id for r in validate_inputs(cfg) if not r.passed}
    assert "IN-panel.csv-eurpln-range" in failed


def test_missing_file_and_short_panel_are_caught(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write(cfg, n=100)
    (cfg.path("clean_dir") / "garch_std_resid.csv").unlink()
    failed = {r.id for r in validate_inputs(cfg) if not r.passed}
    assert {"IN-panel.csv-rows", "IN-garch_std_resid.csv"} <= failed
