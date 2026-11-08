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


# --------------------------------------------------------------------------- presentation rules


def _visual(page, name):
    return ROOT / "powerbi" / "GasRisk.Report/definition/pages" / page / "visuals" / name / "visual.json"


def test_presentation_rules_catch_the_first_review_findings(tmp_path) -> None:
    v = _load("validate_pbip")
    tables = {"DimScenario": {}}
    data = _data("DimScenario", "FactStress")
    tests = json.loads(_visual("backtesting", "backtesting_tests").read_text(encoding="utf-8"))
    card = json.loads(_visual("risk_overview", "risk_overview_var_card").read_text(encoding="utf-8"))
    bar = json.loads(_visual("stress_margin", "stress_margin_stress_bar").read_text(encoding="utf-8"))
    for vis in (tests, card, bar):
        assert v.presentation_errors(vis, tables, data, "x", rels=_rels()) == []
    del tests["visual"]["objects"]["total"]
    del card["visual"]["objects"]["categoryLabels"]
    bar["visual"]["query"]["queryState"]["Category"]["projections"][0]["field"]["Column"]["Property"] = "scenario"
    assert "total row" in v.presentation_errors(tests, tables, data, "x")[0]
    assert "category label" in v.presentation_errors(card, tables, data, "x")[0]
    assert any("labels up to" in e for e in v.presentation_errors(bar, tables, data, "x", rels=_rels()))


def _rels():
    import re

    rel = (ROOT / "powerbi/GasRisk.SemanticModel/definition/relationships.tmdl").read_text(encoding="utf-8")
    out = []
    for frm, to in re.findall(r"fromColumn: (\S+)\n\ttoColumn: (\S+)", rel):
        (ft, fc), (tt, tc) = frm.split(".", 1), to.split(".", 1)
        out.append((ft, fc, tt, tc))
    return out


def _data(*names):
    import pandas as pd

    files = {"DimScenario": "dim_scenario", "FactStress": "fact_stress", "FactMarginStress": "fact_margin_stress",
             "DimLimit": "dim_limit", "FactLimitUtilisation": "fact_limit_utilisation"}
    return {n: pd.read_csv(ROOT / "exports" / f"{files[n]}.csv") for n in names}


def test_scenario_charts_show_every_scenario_without_scrolling() -> None:
    # Desktop's PDF export of the first layout hid scenarios behind a scrollbar, including the largest margin call.
    v = _load("validate_pbip")
    data = _data("DimScenario", "FactStress", "FactMarginStress")
    rels = _rels()
    assert v.category_count(data, rels, "DimScenario", "scenario_short", "FactMarginStress") == 17
    assert v.category_count(data, rels, "DimScenario", "scenario_short", "FactStress") == 14
    for name in ("stress_margin_mstress_bar", "stress_margin_stress_bar"):
        vis = json.loads(_visual("stress_margin", name).read_text(encoding="utf-8"))
        assert v.presentation_errors(vis, {}, data, name, rels=rels, page=(1280, 920)) == []
        assert vis["visual"]["query"]["sortDefinition"]["sort"][0]["direction"] == "Descending"
        vis["position"]["height"] = 280  # the first layout
        errs = v.presentation_errors(vis, {}, data, name, rels=rels, page=(1280, 920))
        assert any("the chart would scroll" in e for e in errs), errs


def test_limit_labels_fit_and_axis_titles_are_not_raw_column_names() -> None:
    v = _load("validate_pbip")
    data = _data("DimLimit", "FactLimitUtilisation")
    vis = json.loads(_visual("risk_overview", "risk_overview_util_bar").read_text(encoding="utf-8"))
    assert v.presentation_errors(vis, {}, data, "u", rels=_rels(), page=(1280, 720)) == []
    proj = vis["visual"]["query"]["queryState"]["Category"]["projections"][0]
    proj["field"]["Column"]["Property"] = "limit_name"  # long names, as in the first export
    vis["position"]["width"] = 400
    errs = v.presentation_errors(vis, {}, data, "u", rels=_rels(), page=(1280, 720))
    assert any("labels up to 20 characters, a 400 px bar chart shows about 16" in e for e in errs), errs
    del proj["displayName"]
    del vis["visual"]["objects"]["categoryAxis"]
    errs = v.presentation_errors(vis, {}, data, "u", rels=_rels(), page=(1280, 720))
    assert any("raw column name" in e for e in errs), errs


def test_visual_outside_the_page_is_rejected() -> None:
    v = _load("validate_pbip")
    vis = json.loads(_visual("stress_margin", "stress_margin_callshare_card").read_text(encoding="utf-8"))
    assert v.presentation_errors(vis, {}, {}, "c", page=(1280, 920)) == []
    assert "beyond the 1280 x 720 page" in v.presentation_errors(vis, {}, {}, "c", page=(1280, 720))[0]


def test_label_card_has_no_header() -> None:
    vis = json.loads(_visual("risk_overview", "risk_overview_label").read_text(encoding="utf-8"))
    assert vis["visual"]["visualContainerObjects"]["title"][0]["properties"]["show"]["expr"]["Literal"]["Value"] == "false"


def test_money_cards_show_millions_with_two_decimals() -> None:
    card = json.loads(_visual("stress_margin", "stress_margin_im_card").read_text(encoding="utf-8"))
    lab = card["visual"]["objects"]["labels"][0]["properties"]
    assert lab["labelDisplayUnits"]["expr"]["Literal"]["Value"] == "1000000D"
    assert lab["labelPrecision"]["expr"]["Literal"]["Value"] == "2L"


def test_backtest_line_plots_var_as_a_negative_threshold() -> None:
    import re

    tmdl = (ROOT / "powerbi/GasRisk.SemanticModel/definition/tables/FactVaR.tmdl").read_text(encoding="utf-8")
    assert re.search(r"measure 'Official VaR 99 Loss Threshold' = - \[Official VaR 99\]", tmdl)
    line = json.loads(_visual("backtesting", "backtesting_pnl_line").read_text(encoding="utf-8"))
    ys = [p["field"]["Measure"]["Property"] for p in line["visual"]["query"]["queryState"]["Y"]["projections"]]
    assert ys == ["Realised P&L", "Official VaR 99 Loss Threshold"]


def test_short_scenario_labels_are_unique_and_short() -> None:
    import pandas as pd

    from gasrisk.export import SHORT_SCENARIO_MAX, short_scenario

    d = pd.read_csv(ROOT / "exports" / "dim_scenario.csv")
    assert d["scenario_short"].is_unique
    assert d["scenario_short"].str.len().max() <= SHORT_SCENARIO_MAX
    assert short_scenario("worst 5-day crisis move") == "Crisis worst 5d"
    assert short_scenario("x" * 40) == "x" * (SHORT_SCENARIO_MAX - 1) + "."
