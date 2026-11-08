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

    for d in ("GasRisk.SemanticModel", "GasRisk.Report", "GasRiskMinimal.Report"):
        shutil.copytree(ROOT / "powerbi" / d, tmp_path / d)
    for f in ("GasRisk.pbip", "GasRiskMinimal.pbip"):
        shutil.copy(ROOT / "powerbi" / f, tmp_path)
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


# --------------------------------------------------------------------------- report structure (page activation)

import json  # noqa: E402

import pytest  # noqa: E402

PAGES = "GasRisk.Report/definition/pages"


def _edit_json(path, fn):
    doc = json.loads(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _set(key, value):
    return lambda d: d.__setitem__(key, value)


def _drop_base_theme(d):
    del d["themeCollection"]["baseTheme"]


def _rename_folder(t, rel, new):
    (t / rel).rename((t / rel).parent / new)


CASES = {
    # The version Desktop 2.159 could not open: PBIR version 1.0.0 and no base theme.
    "pbir_version_1": (lambda t: _edit_json(t / "GasRisk.Report/definition/version.json", _set("version", "1.0.0")),
                       "version.json: version 1.0.0, Desktop writes and expects 2.0.0"),
    "no_base_theme": (lambda t: _edit_json(t / "GasRisk.Report/definition/report.json", _drop_base_theme),
                      "themeCollection has no baseTheme"),
    "missing_theme_file": (lambda t: (t / "GasRisk.Report/StaticResources/SharedResources/BaseThemes/CY24SU06.json").unlink(),
                           "resource SharedResources/BaseThemes/CY24SU06.json not found"),
    "active_page_unknown": (lambda t: _edit_json(t / PAGES / "pages.json", _set("activePageName", "overview")),
                            "activePageName 'overview' matches no page folder"),
    "page_order_case": (lambda t: _edit_json(t / PAGES / "pages.json",
                                             _set("pageOrder", ["Risk_overview", "backtesting", "stress_margin", "pnl_attribution"])),
                        "pageOrder entry 'Risk_overview' matches no page folder"),
    "page_order_missing": (lambda t: _edit_json(t / PAGES / "pages.json",
                                                _set("pageOrder", ["risk_overview", "backtesting", "stress_margin"])),
                           "page folder 'pnl_attribution' missing from pageOrder"),
    "page_name_differs": (lambda t: _edit_json(t / PAGES / "backtesting/page.json", _set("name", "backtest")),
                          "backtesting/page.json: name 'backtest' differs from its folder name"),
    "visual_name_differs": (lambda t: _edit_json(t / PAGES / "backtesting/visuals/backtesting_tests/visual.json",
                                                 _set("name", "tests")),
                            "backtesting_tests/visual.json: name 'tests' differs from its folder name"),
    "page_folder_bad_chars": (lambda t: _rename_folder(t, f"{PAGES}/backtesting", "back testing"),
                              "page folder 'back testing' has characters other than"),
    "pbir_version_3": (lambda t: _edit_json(t / "GasRisk.Report/definition.pbir", _set("version", "3.0")),
                       "definition.pbir: version 3.0 (PBIR needs 4.0)"),
    "schema_too_new": (lambda t: _edit_json(t / PAGES / "backtesting/page.json", _set(
        "$schema", "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/9.0.0/schema.json")),
                       "$schema version 9.0.0 is newer than 2.1.0"),
    "schema_wrong_kind": (lambda t: _edit_json(t / PAGES / "pages.json", _set(
        "$schema", "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/1.0.0/schema.json")),
                          "expected item/report/definition/pagesMetadata"),
    "legacy_report_json": (lambda t: (t / "GasRisk.Report/report.json").write_text("{}", encoding="utf-8"),
                           "legacy report.json next to the PBIR definition folder"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_report_structure_rules_catch_broken_versions(tmp_path, case) -> None:
    v = _load("validate_pbip")
    _copy_project(tmp_path, v)
    breaker, message = CASES[case]
    assert v.report_structure_errors(tmp_path / "GasRisk.Report") == []
    breaker(tmp_path)
    errs = v.report_structure_errors(tmp_path / "GasRisk.Report")
    assert any(message in e for e in errs), errs


def test_minimal_report_is_one_page_on_the_same_model() -> None:
    v = _load("validate_pbip")
    rep = ROOT / "powerbi" / "GasRiskMinimal.Report"
    assert v.report_structure_errors(rep) == []
    meta = json.loads((rep / "definition/pages/pages.json").read_text(encoding="utf-8"))
    assert meta["pageOrder"] == [meta["activePageName"]]
    pbir = json.loads((rep / "definition.pbir").read_text(encoding="utf-8"))
    assert pbir["datasetReference"]["byPath"]["path"] == "../GasRisk.SemanticModel"
