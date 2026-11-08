"""Point the Power BI model at a data folder.

Usage (from the repository root)::

    python powerbi/set_data_folder.py            # use the local exports/ folder of this clone (absolute path)
    python powerbi/set_data_folder.py --github   # back to the public repository's exports/ (default after a clone)

Power Query cannot resolve paths relative to the .pbip file, so the DataFolder parameter holds either the GitHub raw
URL of exports/ or the absolute path of the local exports/ folder. Close Power BI Desktop before running this.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PBI = Path(__file__).resolve().parent
EXPR = PBI / "GasRisk.SemanticModel" / "definition" / "expressions.tmdl"
GITHUB = "https://raw.githubusercontent.com/matteodegianni/pl-gas-risk-framework/master/exports/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", action="store_true")
    a = ap.parse_args()
    folder = GITHUB if a.github else str((PBI.parent / "exports").resolve()) + ("\\" if "\\" in str(PBI) else "/")
    text = EXPR.read_text(encoding="utf-8")
    new = re.sub(r'expression DataFolder = "[^"]*"', f'expression DataFolder = "{folder}"', text, count=1)
    EXPR.write_text(new, encoding="utf-8")
    print(f"DataFolder = {folder}")


if __name__ == "__main__":
    main()
