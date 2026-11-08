"""Validate the Power BI Project without Power BI Desktop.

Usage (from the repository root)::

    python powerbi/validate_pbip.py            # all checks; JSON schemas are downloaded once into powerbi/.schema_cache
    python powerbi/validate_pbip.py --offline  # skip the JSON-schema check if the cache is empty

Checks:

1. Every JSON file (``.pbip``, ``.pbir``, ``.pbism``, report JSON) parses and validates against the Microsoft schema
   named in its ``$schema`` (microsoft/json-schemas, resolved with all nested references).
2. TMDL structure: indentation uses tabs, every table in ``model.tmdl`` has a file, every table has a partition, the
   partition reads an existing CSV and every ``sourceColumn`` exists in that CSV's header.
3. Names and DAX: measure names are unique across the model and match no column name in any table (case-insensitive,
   as in Power BI); every ``Table[Column]`` and ``[Measure]`` reference exists.
4. Relationships: both columns exist, the one side is unique in the CSV data and every many-side key exists on it.
5. Report structure, for every report a ``.pbip`` opens (rules added after Desktop 2.159 failed on page activation):
   ``pages.json`` ``activePageName`` and every ``pageOrder`` entry name an existing page folder exactly, with no
   duplicates and no folder left out; every ``page.json`` and ``visual.json`` ``name`` equals its folder name and uses
   only word characters or hyphens; ``definition.pbir`` is version 4.0 with a ``byPath`` model that exists;
   ``version.json`` is the PBIR version Desktop writes (2.0.0); each ``$schema`` names the right file kind at a version
   published by October 2026 (Desktop 2.159); ``report.json`` has a base theme and every resource-package file exists;
   no legacy ``report.json`` sits next to the ``definition`` folder.
6. Report content: every field a visual uses exists in the model as a column or measure; every page of the main report
   shows the synthetic-data label.
7. Presentation (from the first review in Desktop): no table shows a total row; every card hides the category label
   under the value; text on a horizontal bar chart axis is at most ``MAX_CATEGORY_CHARS`` characters in the data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PBI = Path(__file__).resolve().parent
ROOT = PBI.parent
EXPORTS = ROOT / "exports"
NAME = "GasRisk"
CACHE = PBI / ".schema_cache"
PREFIX = "https://developer.microsoft.com/json-schemas/"
RAW = "https://raw.githubusercontent.com/microsoft/json-schemas/main/"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # PBIR object and folder names: word characters or hyphens
MAX_CATEGORY_CHARS = 24  # longest category label that fits the stress charts without truncation
PBIR_VERSION = "2.0.0"  # definition/version.json as written by Desktop since 2025; 1.0.0 is the 2024 preview format
# Latest schema version per PBIR file kind published by October 2026 (Desktop 2.159, the version the project was opened
# in), from the CHANGELOGs in microsoft/json-schemas. A newer version than Desktop knows makes the file unreadable.
SCHEMA_MAX = {
    "pbip/pbipProperties": "1.0.0",
    "item/report/definitionProperties": "2.0.0",
    "item/semanticModel/definitionProperties": "1.0.0",
    "item/report/definition/versionMetadata": "1.0.0",
    "item/report/definition/report": "3.3.0",
    "item/report/definition/pagesMetadata": "1.1.0",
    "item/report/definition/page": "2.1.0",
    "item/report/definition/visualContainer": "2.13.0",
}


# --------------------------------------------------------------------------- TMDL parsing


def parse_tmdl_table(text: str) -> dict:
    """Columns (name -> sourceColumn), measures (name -> expression) and the partition source of a table file."""
    lines = text.split("\n")
    m = re.match(r"table (.+)$", lines[0])
    assert m, "first line must declare the table"
    table = m.group(1).strip("'")
    cols, measures, partition = {}, {}, []
    i = 1
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("\tcolumn "):
            name = ln[len("\tcolumn "):].strip().strip("'")
            src = None
            i += 1
            while i < len(lines) and (lines[i].startswith("\t\t") or lines[i] == ""):
                s = lines[i].strip()
                if s.startswith("sourceColumn:"):
                    src = s.split(":", 1)[1].strip()
                i += 1
            cols[name] = src
            continue
        if ln.startswith("\tmeasure "):
            head = ln[len("\tmeasure "):]
            name, _, expr = head.partition(" = ") if " = " in head else head.partition(" =")
            name = name.strip().strip("'")
            body = [expr.strip()] if expr.strip() else []
            i += 1
            while i < len(lines) and lines[i].startswith("\t\t\t"):
                body.append(lines[i].strip())
                i += 1
            while i < len(lines) and (lines[i].startswith("\t\t") or lines[i] == ""):
                i += 1
            measures[name] = "\n".join(body)
            continue
        if ln.startswith("\tpartition "):
            i += 1
            while i < len(lines) and (lines[i].startswith("\t\t") or lines[i] == ""):
                partition.append(lines[i])
                i += 1
            continue
        i += 1
    return {"table": table, "columns": cols, "measures": measures, "partition": "\n".join(partition)}


def dax_refs(expr: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Column references Table[Column] and bare measure references [Measure] in a DAX expression."""
    expr = re.sub(r'"[^"]*"', '""', expr)
    cols = [(t.strip("'"), c) for t, c in re.findall(r"('[^']+'|[A-Za-z_][A-Za-z0-9_]*)\[([^\]]+)\]", expr)]
    bare = [m for m in re.findall(r"(?<![A-Za-z0-9_'\]])\[([^\]]+)\]", expr)]
    return cols, bare


# --------------------------------------------------------------------------- JSON schema


def _retrieve(uri: str):
    from referencing import Resource

    if not uri.startswith(PREFIX):
        raise ValueError(f"unexpected schema uri {uri}")
    rel = uri[len(PREFIX):]
    path = CACHE / rel
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + rel, timeout=60) as r:  # noqa: S310 - fixed GitHub host
            path.write_bytes(r.read())
    return Resource.from_contents(json.loads(path.read_text(encoding="utf-8")))


def schema_errors(doc: dict) -> list[str]:
    from jsonschema import Draft7Validator
    from referencing import Registry

    reg = Registry(retrieve=_retrieve)
    schema = _retrieve(doc["$schema"]).contents
    v = Draft7Validator(schema, registry=reg)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:160]}" for e in v.iter_errors(doc)]


# --------------------------------------------------------------------------- checks


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _schema_kind_errors(path: Path, kind: str, rel: str) -> list[str]:
    """The file's $schema must be the expected kind at a version Desktop 2.159 knows."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - reported by the JSON check
        return []
    m = re.fullmatch(re.escape(PREFIX) + r"fabric/(.+)/(\d+\.\d+\.\d+)/schema\.json", str(doc.get("$schema", "")))
    if not m:
        return [f"{rel}: $schema {doc.get('$schema')!r} is not a Fabric schema URL"]
    if m.group(1) != kind:
        return [f"{rel}: $schema is {m.group(1)}, expected {kind}"]
    if _ver(m.group(2)) > _ver(SCHEMA_MAX[kind]):
        return [f"{rel}: $schema version {m.group(2)} is newer than {SCHEMA_MAX[kind]}, the latest Desktop 2.159 knows"]
    return []


def _prop(v: dict, obj: str, name: str):
    """Literal value of visual.objects[obj][0].properties[name], or None."""
    try:
        return v["visual"]["objects"][obj][0]["properties"][name]["expr"]["Literal"]["Value"]
    except (KeyError, IndexError):
        return None


def presentation_errors(v: dict, tables: dict, data: dict, where: str) -> list[str]:
    errors = []
    vt = v["visual"]["visualType"]
    if vt in ("tableEx", "pivotTable") and _prop(v, "total", "totals") != "false":
        errors.append(f"{where}: table shows a total row (set total.totals to false)")
    if vt == "card" and _prop(v, "categoryLabels", "show") != "false":
        errors.append(f"{where}: card shows the category label under the value (clipped; the title names the figure)")
    if vt in ("barChart", "clusteredBarChart"):  # horizontal bars truncate axis labels; column charts wrap them
        for p in v["visual"]["query"]["queryState"].get("Category", {}).get("projections", []):
            spec = p["field"].get("Column")
            if not spec:
                continue
            t, c = spec["Expression"]["SourceRef"]["Entity"], spec["Property"]
            if t in data and c in data[t] and not pd.api.types.is_numeric_dtype(data[t][c]):
                longest = data[t][c].astype(str).str.len().max()
                if longest > MAX_CATEGORY_CHARS:
                    errors.append(f"{where}: category {t}[{c}] has labels up to {longest} characters "
                                  f"(max {MAX_CATEGORY_CHARS}, truncated on the axis)")
    return errors


def report_structure_errors(rep: Path) -> list[str]:
    """PBIR structure rules that decide whether Desktop can find and open the pages of a report folder."""
    errors: list[str] = []
    r = rep.name
    d = rep / "definition"
    if (rep / "report.json").exists():
        errors.append(f"{r}: legacy report.json next to the PBIR definition folder (Desktop reads one format, not both)")
    if not d.is_dir():
        return errors + [f"{r}: no definition/ folder (PBIR)"]

    # definition.pbir, version.json and schema versions
    pbir = rep / "definition.pbir"
    if not pbir.exists():
        errors.append(f"{r}: definition.pbir missing")
    else:
        doc = json.loads(pbir.read_text(encoding="utf-8"))
        errors += _schema_kind_errors(pbir, "item/report/definitionProperties", f"{r}/definition.pbir")
        if _ver(str(doc.get("version", "0"))) < (4, 0):
            errors.append(f"{r}/definition.pbir: version {doc.get('version')} (PBIR needs 4.0)")
        target = doc.get("datasetReference", {}).get("byPath", {}) or {}
        if not target.get("path") or not (rep / target["path"] / "definition.pbism").exists():
            errors.append(f"{r}/definition.pbir: byPath semantic model {target.get('path')!r} not found")
    vf = d / "version.json"
    if not vf.exists():
        errors.append(f"{r}: definition/version.json missing")
    else:
        errors += _schema_kind_errors(vf, "item/report/definition/versionMetadata", f"{r}/version.json")
        v = json.loads(vf.read_text(encoding="utf-8")).get("version")
        if v != PBIR_VERSION:
            errors.append(f"{r}/version.json: version {v}, Desktop writes and expects {PBIR_VERSION}")

    # report.json: base theme and resource files
    rj = d / "report.json"
    if not rj.exists():
        errors.append(f"{r}: definition/report.json missing")
    else:
        errors += _schema_kind_errors(rj, "item/report/definition/report", f"{r}/report.json")
        doc = json.loads(rj.read_text(encoding="utf-8"))
        themes = doc.get("themeCollection", {})
        if "baseTheme" not in themes:
            errors.append(f"{r}/report.json: themeCollection has no baseTheme")
        items = {}
        for pkg in doc.get("resourcePackages", []):
            for it in pkg.get("items", []):
                items[it["name"]] = it
                if not (rep / "StaticResources" / pkg["name"] / it["path"]).exists():
                    errors.append(f"{r}/report.json: resource {pkg['name']}/{it['path']} not found under StaticResources")
        for kind, t in themes.items():
            if t.get("name") not in items:
                errors.append(f"{r}/report.json: {kind} {t.get('name')!r} is not in resourcePackages")

    # pages.json against page folders
    pd_ = d / "pages"
    folders = sorted(p.name for p in pd_.iterdir() if p.is_dir()) if pd_.is_dir() else []
    if not folders:
        return errors + [f"{r}: no page folders"]
    pj = pd_ / "pages.json"
    if not pj.exists():
        errors.append(f"{r}: pages/pages.json missing")
    else:
        errors += _schema_kind_errors(pj, "item/report/definition/pagesMetadata", f"{r}/pages.json")
        meta = json.loads(pj.read_text(encoding="utf-8"))
        order = meta.get("pageOrder", [])
        if len(order) != len(set(order)):
            errors.append(f"{r}/pages.json: duplicate entries in pageOrder {order}")
        for name in order:
            if name not in folders:
                errors.append(f"{r}/pages.json: pageOrder entry {name!r} matches no page folder {folders}")
        for name in folders:
            if name not in order:
                errors.append(f"{r}/pages.json: page folder {name!r} missing from pageOrder")
        active = meta.get("activePageName")
        if active not in folders:
            errors.append(f"{r}/pages.json: activePageName {active!r} matches no page folder {folders}")

    # page and visual names
    for page in folders:
        if not NAME_RE.match(page):
            errors.append(f"{r}: page folder {page!r} has characters other than letters, digits, _ or -")
        pf = pd_ / page / "page.json"
        if not pf.exists():
            errors.append(f"{r}/{page}: page.json missing")
            continue
        errors += _schema_kind_errors(pf, "item/report/definition/page", f"{r}/{page}/page.json")
        pname = json.loads(pf.read_text(encoding="utf-8")).get("name")
        if pname != page:
            errors.append(f"{r}/{page}/page.json: name {pname!r} differs from its folder name")
        vdir = pd_ / page / "visuals"
        for vd in sorted(x for x in vdir.iterdir() if x.is_dir()) if vdir.is_dir() else []:
            vj = vd / "visual.json"
            if not NAME_RE.match(vd.name):
                errors.append(f"{r}/{page}: visual folder {vd.name!r} has characters other than letters, digits, _ or -")
            if not vj.exists():
                errors.append(f"{r}/{page}/{vd.name}: visual.json missing")
                continue
            errors += _schema_kind_errors(vj, "item/report/definition/visualContainer", f"{r}/{page}/{vd.name}/visual.json")
            vname = json.loads(vj.read_text(encoding="utf-8")).get("name")
            if vname != vd.name:
                errors.append(f"{r}/{page}/{vd.name}/visual.json: name {vname!r} differs from its folder name")
    return errors


def validate(offline: bool = False) -> list[str]:
    errors: list[str] = []
    sm = PBI / f"{NAME}.SemanticModel" / "definition"
    reports = []
    for pb in sorted(PBI.glob("*.pbip")):
        for a in json.loads(pb.read_text(encoding="utf-8")).get("artifacts", []):
            reports.append(PBI / a["report"]["path"])

    # 1. JSON
    json_files = [*sorted(PBI.glob("*.pbip")), PBI / f"{NAME}.SemanticModel" / "definition.pbism"]
    for rep in reports:
        json_files += [rep / "definition.pbir", *sorted((rep / "definition").rglob("*.json"))]
    n_schema = 0
    for f in json_files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{f.relative_to(PBI)}: invalid JSON ({exc})")
            continue
        if "$schema" not in doc:
            errors.append(f"{f.relative_to(PBI)}: no $schema")
            continue
        try:
            errs = schema_errors(doc)
            n_schema += 1
        except Exception as exc:  # noqa: BLE001
            if offline:
                continue
            errors.append(f"{f.relative_to(PBI)}: schema not available ({exc})")
            continue
        errors += [f"{f.relative_to(PBI)}: {e}" for e in errs]

    # 2. TMDL
    for f in sm.rglob("*.tmdl"):
        for n, ln in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if re.match(r"^ +\S", ln):
                errors.append(f"{f.name}:{n}: indented with spaces (TMDL uses tabs)")
    model = (sm / "model.tmdl").read_text(encoding="utf-8")
    refs = re.findall(r"^ref table (.+)$", model, flags=re.M)
    tables = {}
    for t in refs:
        tf = sm / "tables" / f"{t.strip(chr(39))}.tmdl"
        if not tf.exists():
            errors.append(f"model.tmdl references missing table file {tf.name}")
            continue
        tables[t] = parse_tmdl_table(tf.read_text(encoding="utf-8"))
    data = {}
    for t, info in tables.items():
        m = re.search(r'fnLoadCsv\("([^"]+)"\)', info["partition"])
        if not m:
            errors.append(f"{t}: partition does not load a CSV")
            continue
        csv = EXPORTS / f"{m.group(1)}.csv"
        if not csv.exists():
            errors.append(f"{t}: {csv.name} not found in exports/")
            continue
        df = pd.read_csv(csv)
        data[t] = df
        for c, src in info["columns"].items():
            if src not in df.columns:
                errors.append(f"{t}[{c}]: sourceColumn {src} not in {csv.name}")
        for c in re.findall(r'\{"([^"]+)", ', info["partition"]):
            if c not in df.columns:
                errors.append(f"{t}: type transform on missing column {c}")
    expr = (sm / "expressions.tmdl").read_text(encoding="utf-8")
    if "IsParameterQuery=true" not in expr or "fnLoadCsv" not in expr:
        errors.append("expressions.tmdl: DataFolder parameter or fnLoadCsv missing")

    # 3. Names: Power BI compares object names case-insensitively across the whole model, so a measure may not share
    # its name with another measure or with any column of any table ("cannot be created because a column with the
    # same name already exists").
    all_measures = {}
    seen: dict[str, str] = {}
    columns_ci: dict[str, list[str]] = {}
    for t, info in tables.items():
        for c in info["columns"]:
            columns_ci.setdefault(c.lower(), []).append(f"{t}[{c}]")
    for t, info in tables.items():
        for mname in info["measures"]:
            k = mname.lower()
            if k in seen:
                errors.append(f"duplicate measure name [{mname}] in {t} and {seen[k]}")
            if k in columns_ci:
                errors.append(f"measure {t}[{mname}] has the same name as column {', '.join(columns_ci[k])}")
            seen[k] = t
            all_measures[mname] = t

    # 3b. DAX references
    for t, info in tables.items():
        for mname, e in info["measures"].items():
            cols, bare = dax_refs(e)
            for tt, cc in cols:
                if tt not in tables or cc not in tables[tt]["columns"]:
                    errors.append(f"measure [{mname}]: unknown column {tt}[{cc}]")
            for b in bare:
                if b not in all_measures:
                    errors.append(f"measure [{mname}]: unknown measure [{b}]")

    # 4. Relationships
    rel = (sm / "relationships.tmdl").read_text(encoding="utf-8")
    n_rel = 0
    for frm, to in re.findall(r"fromColumn: (\S+)\n\ttoColumn: (\S+)", rel):
        n_rel += 1
        (ft, fc), (tt, tc) = frm.split(".", 1), to.split(".", 1)
        fc, tc = fc.strip("'"), tc.strip("'")
        if ft not in tables or fc not in tables[ft]["columns"] or tt not in tables or tc not in tables[tt]["columns"]:
            errors.append(f"relationship {frm} -> {to}: unknown column")
            continue
        one, many = data[tt][tc], data[ft][fc].dropna()
        if not one.is_unique:
            errors.append(f"relationship {frm} -> {to}: {tt}[{tc}] is not unique")
        missing = set(many) - set(one)
        if missing:
            errors.append(f"relationship {frm} -> {to}: {len(missing)} keys missing on the one side")

    # 5. Report structure
    for rep in reports:
        errors += report_structure_errors(rep)

    # 6. Report content
    n_vis, n_fields, n_pages = 0, 0, 0
    for rep in reports:
        pages_dir = rep / "definition" / "pages"
        folders = sorted(p.name for p in pages_dir.iterdir() if p.is_dir()) if pages_dir.is_dir() else []
        n_pages += len(folders)
        for page in folders:
            has_label = False
            for vf in (pages_dir / page / "visuals").glob("*/visual.json"):
                n_vis += 1
                v = json.loads(vf.read_text(encoding="utf-8"))
                for role in v["visual"]["query"]["queryState"].values():
                    for p in role["projections"]:
                        kind, spec = next(iter(p["field"].items()))
                        ent, prop = spec["Expression"]["SourceRef"]["Entity"], spec["Property"]
                        n_fields += 1
                        if kind == "Column" and (ent not in tables or prop not in tables[ent]["columns"]):
                            errors.append(f"{rep.name}/{page}/{v['name']}: unknown column {ent}[{prop}]")
                        if kind == "Measure" and (ent not in tables or prop not in tables[ent]["measures"]):
                            errors.append(f"{rep.name}/{page}/{v['name']}: unknown measure {ent}[{prop}]")
                        if kind == "Measure" and prop == "Data Label":
                            has_label = True
            if not has_label:
                errors.append(f"{rep.name} page {page}: no synthetic data label")

    # 7. Presentation
    for rep in reports:
        for vf in sorted((rep / "definition" / "pages").glob("*/visuals/*/visual.json")):
            errors += presentation_errors(json.loads(vf.read_text(encoding="utf-8")), tables, data, f"{rep.name}/{vf.parent.name}")
    print(f"JSON files: {len(json_files)} ({n_schema} schema-validated); tables: {len(tables)}; measures: {len(all_measures)}; "
          f"relationships: {n_rel}; reports: {len(reports)}; pages: {n_pages}; visuals: {n_vis}; visual fields: {n_fields}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    errs = validate(ap.parse_args().offline)
    for e in errs:
        print("ERROR", e)
    print("OK" if not errs else f"{len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
