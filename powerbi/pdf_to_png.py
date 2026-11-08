"""Convert a PDF exported from Power BI Desktop into one PNG per report page, for the README screenshots.

One-off step, run after exporting the report from Desktop (File > Export > Export to PDF)::

    pip install -r requirements-dev.txt
    python powerbi/pdf_to_png.py path/to/GasRisk.pdf        # writes docs/img/powerbi/<page>.png

Pages are named after the report pages in order, so the file names stay stable between exports. Needs PyMuPDF,
which is a development dependency only (``requirements-dev.txt``); the pipeline does not use it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "powerbi"
PAGES = ["1_risk_overview", "2_backtesting", "3_stress_margin", "4_pnl_attribution"]


def convert(pdf: Path, out: Path = OUT, dpi: int = 144) -> list[Path]:
    import pymupdf

    doc = pymupdf.open(pdf)
    if doc.page_count != len(PAGES):
        raise SystemExit(f"{pdf.name} has {doc.page_count} pages, the report has {len(PAGES)}")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for page, name in zip(doc, PAGES):
        path = out / f"{name}.png"
        page.get_pixmap(dpi=dpi).save(path)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--dpi", type=int, default=144)
    args = ap.parse_args()
    for p in convert(args.pdf, dpi=args.dpi):
        print(p.relative_to(ROOT), f"{p.stat().st_size // 1024} KB")
    sys.exit(0)


if __name__ == "__main__":
    main()
