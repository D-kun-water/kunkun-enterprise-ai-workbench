"""Tests for the spoken-language answer-quality regression."""

from __future__ import annotations

import re

import pytest

from app.answer_quality import (
    evaluate_answer_quality,
    load_answer_quality_cases,
)
from app.corpus_adaptations import DISABLE_ENV_VAR
from app.core import RuntimeContainer

def test_answer_quality_cases_load_from_dedicated_directory() -> None:
    cases = load_answer_quality_cases()
    assert len(cases) >= 7
    assert all(case.case_id.startswith("AQ-") for case in cases)


def test_answer_quality_regression_passes_with_adaptation_on(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    cases = load_answer_quality_cases()
    result = evaluate_answer_quality(runtime, cases, adaptation_enabled=True)
    # The adaptation-on side scores every case and all of them must pass.
    assert result["recorded"] == 0
    assert result["failed"] == 0, [
        row["case_id"] for row in result["rows"] if row["passed"] is False
    ]
    assert result["passed"] == len(cases)


def test_answer_quality_regression_captures_off_switch_differences(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    cases = load_answer_quality_cases()
    on = evaluate_answer_quality(runtime, cases, adaptation_enabled=True)
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    off = evaluate_answer_quality(runtime, cases, adaptation_enabled=False)

    # Refusal-declared cases must degrade to "依据不足" when the layer is off.
    for case in cases:
        if case.without_adaptation.get("expected") == "refusal":
            row = next(row for row in off["rows"] if row["case_id"] == case.case_id)
            assert row["passed"] is True, case.case_id

    assert on["failed"] == 0
    assert off["failed"] == 0


def test_answer_quality_mode_argument_controls_adaptation_without_env_override(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    cases = load_answer_quality_cases()

    result = evaluate_answer_quality(runtime, cases, adaptation_enabled=False)

    assert result["mode"] == "adaptation_off"
    assert result["expected_refusals"] == 1
    expected_valid_answers = sum(
        case.without_adaptation.get("expected") == "answer"
        and not case.without_adaptation.get("may_vary")
        for case in cases
    )
    assert result["valid_answers"] == expected_valid_answers
    assert result["failed"] == 0


def test_spoken_short_form_is_answered_in_both_modes(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    cases = load_answer_quality_cases()
    short_form = next(case for case in cases if case.case_id == "AQ-001")
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    on = evaluate_answer_quality(runtime, [short_form], adaptation_enabled=True)
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    off = evaluate_answer_quality(runtime, [short_form], adaptation_enabled=False)
    assert on["failed"] == 0
    assert off["failed"] == 0


@pytest.mark.parametrize(
    "question",
    [
        "事假怎么申请？",
        "婚假原则上多久休完？",
        "合同正式签署前需要经过什么审批？",
    ],
)
def test_api_answers_strip_policy_item_numbers(
    deterministic_runtime: RuntimeContainer, question: str
) -> None:
    """Summary answers must not leak raw policy numbering like ``- 1.``."""
    runtime = deterministic_runtime
    answer = runtime.ask(question)["answer"]
    assert not re.search(r"^- \d+[.、]", answer, flags=re.MULTILINE), answer
