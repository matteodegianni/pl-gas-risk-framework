"""The Power BI project must validate (structure, DAX and visual references, relationships, JSON schemas when cached)."""

import importlib.util

from gasrisk.config import load_config

ROOT = load_config().root


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "powerbi" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pbip_validates() -> None:
    assert _load("validate_pbip").validate(offline=True) == []


def test_measure_reference_parser() -> None:
    v = _load("validate_pbip")
    cols, bare = v.dax_refs('CALCULATE ( [VaR 99], DimMethod[is_official] = 1, \'Dim X\'[a b] = "[not a ref]" )')
    assert cols == [("DimMethod", "is_official"), ("Dim X", "a b")]
    assert bare == ["VaR 99"]
