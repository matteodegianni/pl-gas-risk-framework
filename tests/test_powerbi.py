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


def _copy_project(tmp_path, mod):
    import shutil

    for d in ("GasRisk.SemanticModel", "GasRisk.Report"):
        shutil.copytree(ROOT / "powerbi" / d, tmp_path / d)
    shutil.copy(ROOT / "powerbi" / "GasRisk.pbip", tmp_path)
    mod.PBI = tmp_path
    mod.CACHE = ROOT / "powerbi" / ".schema_cache"


def test_measure_named_like_a_column_is_rejected(tmp_path) -> None:
    # The case Power BI Desktop refused: a measure 'Utilisation' next to the column 'utilisation'.
    v = _load("validate_pbip")
    _copy_project(tmp_path, v)
    f = tmp_path / "GasRisk.SemanticModel" / "definition" / "tables" / "FactLimitUtilisation.tmdl"
    text = f.read_text(encoding="utf-8")
    f.write_text(text.replace("\tmeasure 'Limit Utilisation %' =", "\tmeasure Utilisation =", 1), encoding="utf-8")
    errs = v.validate(offline=True)
    assert any("FactLimitUtilisation[Utilisation] has the same name as column" in e for e in errs)


def test_duplicate_measure_names_are_rejected_case_insensitively(tmp_path) -> None:
    v = _load("validate_pbip")
    _copy_project(tmp_path, v)
    f = tmp_path / "GasRisk.SemanticModel" / "definition" / "tables" / "FactCredit.tmdl"
    text = f.read_text(encoding="utf-8")
    f.write_text(text.replace("\tmeasure 'PFE Limit' =", "\tmeasure 'var limit' =", 1), encoding="utf-8")
    errs = v.validate(offline=True)
    assert any("duplicate measure name [var limit]" in e for e in errs)
