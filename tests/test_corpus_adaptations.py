"""Tests for the switchable corpus adaptation layer."""

from __future__ import annotations

from app.corpus_adaptations import ADAPTATIONS, DISABLE_ENV_VAR, corpus_adaptations_enabled
from app.core import RuntimeContainer


def test_adaptation_layer_is_on_by_default() -> None:
    assert corpus_adaptations_enabled() is True


def test_adaptation_inventory_is_documented() -> None:
    assert len(ADAPTATIONS) >= 15
    assert all(isinstance(item, str) and item for item in ADAPTATIONS)

def test_adaptation_layer_preserves_answer_with_current_corpus(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    question = "事假怎么申请？"
    with_enabled = runtime.assistant.ask(question)["answer"]

    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    with_disabled = runtime.assistant.ask(question)["answer"]

    assert "依据不足" not in with_enabled
    assert "依据不足" not in with_disabled
    assert "提前1个工作日" in with_enabled
    assert "提前1个工作日" in with_disabled


def test_fixed_evaluation_is_stable_across_adaptation_switch(
    deterministic_runtime: RuntimeContainer, monkeypatch
) -> None:
    runtime = deterministic_runtime
    from app.evaluation import run_evaluation

    cases = runtime.evaluation_cases
    available_sources = {
        document.source_path for document in runtime.documents if document.status == "ready"
    }
    retriever = runtime.retriever

    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    on = run_evaluation(retriever, cases, available_sources)

    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    off = run_evaluation(retriever, cases, available_sources)

    assert on["passed_count"] == off["passed_count"]
    assert on["refusal_accuracy"] == off["refusal_accuracy"]
    assert on["hit_at_1"] == off["hit_at_1"]
