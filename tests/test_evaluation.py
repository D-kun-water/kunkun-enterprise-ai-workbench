import json
from pathlib import Path

import pytest

from app.evaluation import evaluate_ranking, load_evaluation_cases, run_evaluation
from app.ingestion import TextChunk
from app.rag import SearchResult


def test_ranking_metrics_for_second_place_hit() -> None:
    metrics = evaluate_ranking(["other.md", "expected.md"], "expected.md")

    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_3 == 1
    assert metrics.reciprocal_rank == 0.5
    assert metrics.failure_type == "passed"


def test_ranking_failure_types() -> None:
    low = evaluate_ranking(["a.md", "b.md", "c.md", "expected.md"], "expected.md")
    absent = evaluate_ranking(["a.md"], "expected.md", {"a.md", "expected.md"})
    missing = evaluate_ranking(["a.md"], "expected.md", {"a.md"})

    assert low.failure_type == "ranked_too_low"
    assert absent.failure_type == "not_retrieved"
    assert missing.failure_type == "missing_document"


def test_case_loader_combines_files_and_rejects_duplicates(tmp_path: Path) -> None:
    first = [{"case_id": "b-001", "question": "问题 B", "expected_source": "b.md", "expected_keywords": []}]
    second = [{"case_id": "a-001", "question": "问题 A", "expected_source": "a.md", "expected_keywords": ["规则"]}]
    (tmp_path / "b.json").write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(second, ensure_ascii=False), encoding="utf-8")

    cases = load_evaluation_cases(tmp_path)

    assert [case.case_id for case in cases] == ["a-001", "b-001"]

    (tmp_path / "duplicate.json").write_text(json.dumps(second, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_evaluation_cases(tmp_path)


class FakeRetriever:
    def retrieve(self, question: str, top_k: int = 10):
        source_path = "expected.md" if "正确" in question else "other.md"
        chunk = TextChunk("chunk-0", "doc", "制度", "分类", "章节", source_path, "正确规则", 0)
        return {"final_results": [SearchResult(chunk, 1.0, "hybrid", 1)]}


class EmptyRetriever:
    def retrieve(self, question: str, top_k: int = 10):
        return {"final_results": []}


def test_batch_report_exposes_metrics_and_failed_cases(tmp_path: Path) -> None:
    cases = [
        {"case_id": "ok", "question": "正确问题", "expected_source": "expected.md", "expected_keywords": ["规则"]},
        {"case_id": "bad", "question": "其他问题", "expected_source": "expected.md", "expected_keywords": []},
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    report = run_evaluation(
        FakeRetriever(),
        load_evaluation_cases(tmp_path),
        available_sources={"expected.md", "other.md"},
    )

    assert report["total_cases"] == 2
    assert report["hit_at_1"] == 0.5
    assert report["failed_count"] == 1
    assert report["cases"][1]["failure_type"] == "passed"


def test_evaluation_supports_refusal_and_product_metrics(tmp_path: Path) -> None:
    cases = [
        {"case_id": "answer", "question": "正确问题", "expected_source": "expected.md", "expected_keywords": ["规则"]},
        {"case_id": "refuse", "question": "无答案问题", "expected_source": "", "expected_keywords": [], "should_refuse": True},
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    class MixedRetriever:
        def retrieve(self, question: str, top_k: int = 10):
            return EmptyRetriever().retrieve(question) if "无答案" in question else FakeRetriever().retrieve(question)

    report = run_evaluation(MixedRetriever(), load_evaluation_cases(tmp_path), {"expected.md"})

    assert report["refusal_accuracy"] == 1.0
    assert report["refusal_baseline_accuracy"] == 1.0
    assert report["citation_coverage"] == 1.0
    assert report["document_coverage"] == 1.0
    assert report["latency_p50_ms"] >= 0
    assert report["latency_p95_ms"] >= report["latency_p50_ms"]


def test_refusal_accuracy_uses_evidence_gate_and_keeps_raw_retrieval_baseline(tmp_path: Path) -> None:
    cases = [
        {
            "case_id": "refuse",
            "question": "无答案问题",
            "expected_source": "",
            "expected_keywords": [],
            "should_refuse": True,
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    report = run_evaluation(
        FakeRetriever(),
        load_evaluation_cases(tmp_path),
        available_sources={"expected.md", "other.md"},
    )

    assert report["refusal_accuracy"] == 1.0
    assert report["refusal_baseline_accuracy"] == 0.0
    assert report["cases"][0]["refusal_correct"] is True
    assert report["cases"][0]["retrieval_refusal_correct"] is False
    assert report["cases"][0]["failure_type"] == "passed"


def test_refusal_accuracy_flags_a_false_positive_evidence_match(tmp_path: Path) -> None:
    cases = [
        {
            "case_id": "refuse",
            "question": "公司提供宠物医疗报销吗？",
            "expected_source": "",
            "expected_keywords": [],
            "should_refuse": True,
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    class FalsePositiveRetriever:
        def retrieve(self, question: str, top_k: int = 10):
            chunk = TextChunk(
                "bad-0", "bad", "无关制度", "分类", "章节", "bad.md",
                "公司提供宠物医疗报销。", 0,
            )
            return {"final_results": [SearchResult(chunk, 1.0, "hybrid", 1)]}

    report = run_evaluation(
        FalsePositiveRetriever(),
        load_evaluation_cases(tmp_path),
        available_sources={"bad.md"},
    )

    assert report["refusal_accuracy"] == 0.0
    assert report["refusal_baseline_accuracy"] == 0.0
    assert report["cases"][0]["failure_type"] == "false_positive_evidence"
