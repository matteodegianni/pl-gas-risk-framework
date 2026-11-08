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
5. Report: page folders match ``pages.json``; every field a visual uses exists in the model as a column or measure;
   every page shows the synthetic-data label.
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


def validate(offline: bool = False) -> list[str]:
    errors: list[str] = []
    sm = PBI / f"{NAME}.SemanticModel" / "definition"
    rep = PBI / f"{NAME}.Report"

    # 1. JSON
    json_files = [PBI / f"{NAME}.pbip", rep / "definition.pbir", PBI / f"{NAME}.SemanticModel" / "definition.pbism",
                  *sorted((rep / "definition").rglob("*.json"))]
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

    # 5. Report
    pages_meta = json.loads((rep / "definition" / "pages" / "pages.json").read_text(encoding="utf-8"))
    folders = sorted(p.name for p in (rep / "definition" / "pages").iterdir() if p.is_dir())
    if sorted(pages_meta["pageOrder"]) != folders:
        errors.append(f"pages.json order {pages_meta['pageOrder']} does not match folders {folders}")
    n_vis, n_fields = 0, 0
    for page in folders:
        has_label = False
        for vf in (rep / "definition" / "pages" / page / "visuals").glob("*/visual.json"):
            n_vis += 1
            v = json.loads(vf.read_text(encoding="utf-8"))
            if v["name"] != vf.parent.name:
                errors.append(f"{page}/{vf.parent.name}: name differs from folder")
            for role in v["visual"]["query"]["queryState"].values():
                for p in role["projections"]:
                    kind, spec = next(iter(p["field"].items()))
                    ent, prop = spec["Expression"]["SourceRef"]["Entity"], spec["Property"]
                    n_fields += 1
                    if kind == "Column" and (ent not in tables or prop not in tables[ent]["columns"]):
                        errors.append(f"{page}/{v['name']}: unknown column {ent}[{prop}]")
                    if kind == "Measure" and (ent not in tables or prop not in tables[ent]["measures"]):
                        errors.append(f"{page}/{v['name']}: unknown measure {ent}[{prop}]")
                    if kind == "Measure" and prop == "Data Label":
                        has_label = True
        if not has_label:
            errors.append(f"page {page}: no synthetic data label")
    print(f"JSON files: {len(json_files)} ({n_schema} schema-validated); tables: {len(tables)}; measures: {len(all_measures)}; "
          f"relationships: {n_rel}; pages: {len(folders)}; visuals: {n_vis}; visual fields: {n_fields}")
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
