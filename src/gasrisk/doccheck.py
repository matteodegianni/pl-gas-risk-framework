"""Check that every number in a document comes from code output.

Usage::

    python -m gasrisk.doccheck docs/risk_framework.md docs/management_memo.md ...

A number token in the document (``1,234,567``, ``0.7329``, ``52.8%``, ``14.35``) passes if the same string, or the same
number with thousands separators removed, appears in the source corpus: the generated reports and tables in
``outputs/`` and ``config/config.yaml``. Integers up to 31 (stage numbers, horizons, day counts) and four-digit years
are allowed without a source. The check is deliberately literal: a hand-rounded number fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from gasrisk.config import Config, load_config

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?%?(?![\w])")


def corpus(cfg: Config, extra: list[Path] | None = None) -> str:
    out = cfg.path("outputs_dir")
    parts = [p.read_text(encoding="utf-8") for p in sorted(out.glob("*.md"))]
    parts += [p.read_text(encoding="utf-8") for p in sorted(out.glob("*.csv"))]
    parts.append((cfg.root / "config" / "config.yaml").read_text(encoding="utf-8"))
    for p in extra or []:
        parts.append(Path(p).read_text(encoding="utf-8"))
    text = "\n".join(parts)
    return text + "\n" + text.replace(",", "")


def _allowed(tok: str) -> bool:
    t = tok.strip("+-").rstrip("%").replace(",", "")
    if re.fullmatch(r"\d+", t):
        v = int(t)
        return v <= 31 or 1990 <= v <= 2035
    return False


def strip_non_prose(text: str) -> str:
    """Drop links targets, code spans, dates (YYYY-MM-DD) and identifiers such as CP04 or Stage labels."""
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    text = re.sub(r"\b\d{4}-\d{2}\b", "", text)
    return text


def check(doc: Path, text_corpus: str) -> list[str]:
    text = strip_non_prose(Path(doc).read_text(encoding="utf-8"))
    missing = []
    for m in NUM.finditer(text):
        tok = m.group(0).rstrip(",").lstrip("+")
        if not tok or _allowed(tok):
            continue
        bare = tok.replace(",", "")
        if tok in text_corpus or bare in text_corpus:
            continue
        missing.append(tok)
    return missing


def main() -> None:
    cfg = load_config()
    c = corpus(cfg)
    bad = 0
    for d in sys.argv[1:]:
        miss = check(Path(d), c)
        print(f"{d}: {'OK' if not miss else 'NOT FOUND ' + ', '.join(sorted(set(miss)))}")
        bad += len(miss)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
