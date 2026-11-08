"""Every number in the written documents must come from code output (generated reports, tables or config)."""

from pathlib import Path

import pytest

from gasrisk.config import load_config
from gasrisk.doccheck import check, corpus

CFG = load_config()
DOCS = [p for p in [CFG.root / "README.md", CFG.root / "docs" / "risk_framework.md", CFG.root / "docs" / "management_memo.md",
                    CFG.root / "docs" / "interview_notes.md", *sorted((CFG.root / "docs" / "model_cards").glob("*.md")),
                    *sorted((CFG.root / "handover").glob("*.md"))] if p.exists()]


@pytest.fixture(scope="module")
def text_corpus() -> str:
    return corpus(CFG)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_numbers_come_from_outputs(doc: Path, text_corpus: str) -> None:
    assert check(doc, text_corpus) == []


def test_checker_rejects_a_hand_typed_number(tmp_path: Path, text_corpus: str) -> None:
    p = tmp_path / "x.md"
    p.write_text("VaR was 12,345,678.9 PLN and 0.123456789 of the limit, in Stage 7.")
    assert set(check(p, text_corpus)) == {"12,345,678.9", "0.123456789"}
