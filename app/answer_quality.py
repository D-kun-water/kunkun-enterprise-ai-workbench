"""Answer-quality regression for spoken-language questions.

Unlike the retrieval evaluation (Hit@k / MRR), this module checks the final
employee-facing answers produced by the assistant, so that the corpus adaptation
layer's effect on answer quality is reproducible instead of anecdotal.

Cases live in ``data/answer_quality/answer_quality_cases.json`` (a separate
directory so the fixed evaluation loader never mistakes them for retrieval
cases). Each case declares the expected behavior with and without the corpus
adaptation layer:

- ``expected == "answer"``: the reply must contain real policy evidence and,
  when ``keywords`` is given, every mandatory keyword.
- ``expected == "refusal"``: the reply must be the "依据不足" refusal.
- ``may_vary``: only the with-adaptation side is asserted; the other side is
  recorded for comparison but not scored.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.agent import INSUFFICIENT_ANSWER
from app.core import BASE_DIR, RuntimeContainer
from app.corpus_adaptations import DISABLE_ENV_VAR

ANSWER_QUALITY_DIR = BASE_DIR / "data" / "answer_quality"


@contextmanager
def _adaptation_mode(enabled: bool) -> Iterator[None]:
    """Make the public evaluation flag authoritative for this run."""
    previous = os.environ.get(DISABLE_ENV_VAR)
    try:
        if enabled:
            os.environ.pop(DISABLE_ENV_VAR, None)
        else:
            os.environ[DISABLE_ENV_VAR] = "1"
        yield
    finally:
        if previous is None:
            os.environ.pop(DISABLE_ENV_VAR, None)
        else:
            os.environ[DISABLE_ENV_VAR] = previous


@dataclass(frozen=True)
class AnswerQualityCase:
    case_id: str
    question: str
    with_adaptation: dict[str, Any]
    without_adaptation: dict[str, Any]
    notes: str = ""


def load_answer_quality_cases(directory: Path | None = None) -> list[AnswerQualityCase]:
    path = Path(directory) if directory else ANSWER_QUALITY_DIR
    path = path / "answer_quality_cases.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[AnswerQualityCase] = []
    for row in payload:
        cases.append(
            AnswerQualityCase(
                case_id=str(row["case_id"]),
                question=str(row["question"]),
                with_adaptation=dict(row["with_adaptation"]),
                without_adaptation=dict(row["without_adaptation"]),
                notes=str(row.get("notes", "")),
            )
        )
    return cases


def _judge(answer: str, expectation: dict[str, Any]) -> bool:
    if expectation.get("expected") == "refusal":
        return INSUFFICIENT_ANSWER.split("，")[0] in answer or "依据不足" in answer
    if "依据不足" in answer:
        return False
    keywords = expectation.get("keywords") or []
    if keywords:
        return all(keyword in answer for keyword in keywords)
    return True


def evaluate_answer_quality(
    container: RuntimeContainer,
    cases: list[AnswerQualityCase],
    *,
    adaptation_enabled: bool,
) -> dict[str, Any]:
    """Score each case against the declared expectation for the given mode."""
    rows: list[dict[str, Any]] = []
    with _adaptation_mode(adaptation_enabled):
        for case in cases:
            expectation = case.with_adaptation if adaptation_enabled else case.without_adaptation
            answer = container.ask(case.question).get("answer", "")
            scored = not bool(expectation.get("may_vary")) or adaptation_enabled
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "adaptation_enabled": adaptation_enabled,
                    "expected": expectation.get("expected"),
                    "may_vary": bool(expectation.get("may_vary")) and not adaptation_enabled,
                    "passed": _judge(answer, expectation) if scored else None,
                    "answer_preview": answer.replace("\n", " ")[:80],
                    "notes": case.notes,
                }
            )
    scored_rows = [row for row in rows if row["passed"] is not None]
    valid_answers = sum(
        bool(row["passed"]) for row in scored_rows if row["expected"] == "answer"
    )
    expected_refusals = sum(row["expected"] == "refusal" for row in scored_rows)
    refusal_passed = sum(
        bool(row["passed"]) for row in scored_rows if row["expected"] == "refusal"
    )
    return {
        "mode": "adaptation_on" if adaptation_enabled else "adaptation_off",
        "passed": sum(bool(row["passed"]) for row in scored_rows),
        "failed": sum(not bool(row["passed"]) for row in scored_rows),
        "recorded": len(rows) - len(scored_rows),
        "valid_answers": valid_answers,
        "expected_refusals": expected_refusals,
        "refusal_passed": refusal_passed,
        "rows": rows,
    }
