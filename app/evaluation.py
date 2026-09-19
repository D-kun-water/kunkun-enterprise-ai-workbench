"""Deterministic retrieval evaluation for the fixed employee question set."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.agent import has_sufficient_evidence
from app.rag import HybridRetriever, SearchResult


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_source: str
    expected_keywords: list[str]
    notes: str = ""
    should_refuse: bool = False


@dataclass(frozen=True)
class RankingMetrics:
    rank: int
    hit_at_1: int
    hit_at_3: int
    reciprocal_rank: float
    failure_type: str


def evaluate_ranking(
    ranked_sources: list[str],
    expected_source: str,
    available_sources: set[str] | None = None,
) -> RankingMetrics:
    """Evaluate the first occurrence of the expected document in a ranking."""

    try:
        rank = ranked_sources.index(expected_source) + 1
    except ValueError:
        rank = 0

    if 0 < rank <= 3:
        failure_type = "passed"
    elif rank > 3:
        failure_type = "ranked_too_low"
    elif available_sources is not None and expected_source not in available_sources:
        failure_type = "missing_document"
    else:
        failure_type = "not_retrieved"

    return RankingMetrics(
        rank=rank,
        hit_at_1=int(rank == 1),
        hit_at_3=int(0 < rank <= 3),
        reciprocal_rank=1.0 / rank if rank else 0.0,
        failure_type=failure_type,
    )


def load_evaluation_cases(directory: Path) -> list[EvaluationCase]:
    directory = Path(directory)
    cases: list[EvaluationCase] = []
    case_ids: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"评测文件必须是数组：{path.name}")
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"评测案例必须是对象：{path.name}")
            missing = [key for key in ("case_id", "question") if not row.get(key)]
            if not row.get("expected_source") and not row.get("should_refuse", False):
                missing.append("expected_source")
            if missing:
                raise ValueError(f"评测案例缺少字段 {', '.join(missing)}：{path.name}")
            case_id = str(row["case_id"])
            if case_id in case_ids:
                raise ValueError(f"发现重复 case_id：{case_id}")
            case_ids.add(case_id)
            keywords = row.get("expected_keywords", [])
            if not isinstance(keywords, list):
                raise ValueError(f"expected_keywords 必须是数组：{case_id}")
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    question=str(row["question"]),
                    expected_source=str(row.get("expected_source", "")),
                    expected_keywords=[str(keyword) for keyword in keywords],
                    notes=str(row.get("notes", "")),
                    should_refuse=bool(row.get("should_refuse", False)),
                )
            )
    return sorted(cases, key=lambda case: case.case_id)


def run_evaluation(
    retriever: HybridRetriever,
    cases: Iterable[EvaluationCase],
    available_sources: set[str],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        response = retriever.retrieve(case.question, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        results: list[SearchResult] = response.get("final_results", [])
        ranked_sources = _unique_sources(results)
        if case.should_refuse:
            # The answer path uses the same deterministic evidence gate before
            # composing a response. Keep the stricter no-source retrieval rate
            # as a separate diagnostic baseline.
            refusal_correct = not has_sufficient_evidence(case.question, results)
            retrieval_refusal_correct = not ranked_sources
            metrics = RankingMetrics(
                0,
                0,
                0,
                0.0,
                "passed" if refusal_correct else "false_positive_evidence",
            )
        else:
            metrics = evaluate_ranking(ranked_sources, case.expected_source, available_sources)
            refusal_correct = None
            retrieval_refusal_correct = None
        expected_content = "\n".join(
            result.chunk.content for result in results if result.chunk.source_path == case.expected_source
        )
        keyword_hits = [keyword for keyword in case.expected_keywords if keyword.lower() in expected_content.lower()]
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_source": case.expected_source,
                "expected_keywords": case.expected_keywords,
                "keyword_hits": keyword_hits,
                "notes": case.notes,
                "should_refuse": case.should_refuse,
                "refusal_correct": refusal_correct,
                "retrieval_refusal_correct": retrieval_refusal_correct,
                "ranked_sources": ranked_sources,
                **asdict(metrics),
            }
        )

    total = len(rows)
    answer_rows = [row for row in rows if not row["should_refuse"]]
    hit_at_1_count = sum(row["hit_at_1"] for row in answer_rows)
    hit_at_3_count = sum(row["hit_at_3"] for row in answer_rows)
    reciprocal_rank_sum = sum(row["reciprocal_rank"] for row in answer_rows)
    passed_count = sum(row["failure_type"] == "passed" for row in rows)
    refusal_rows = [row for row in rows if row["should_refuse"]]
    citation_hits = sum(row["expected_source"] in row["ranked_sources"] for row in answer_rows)
    expected_documents = {row["expected_source"] for row in answer_rows if row["expected_source"]}
    refusal_correct_count = sum(bool(row["refusal_correct"]) for row in refusal_rows)
    retrieval_refusal_correct_count = sum(
        bool(row["retrieval_refusal_correct"]) for row in refusal_rows
    )
    return {
        "total_cases": total,
        "hit_at_1": round(hit_at_1_count / len(answer_rows), 4) if answer_rows else 0.0,
        "hit_at_3": round(hit_at_3_count / len(answer_rows), 4) if answer_rows else 0.0,
        "mrr": round(reciprocal_rank_sum / len(answer_rows), 4) if answer_rows else 0.0,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        # This is deterministic answer-gate accuracy, not subjective LLM
        # quality. The raw no-source rate remains visible as a baseline.
        "refusal_accuracy": round(refusal_correct_count / len(refusal_rows), 4) if refusal_rows else 0.0,
        "refusal_baseline_accuracy": round(
            retrieval_refusal_correct_count / len(refusal_rows), 4
        ) if refusal_rows else 0.0,
        "citation_coverage": round(citation_hits / len(answer_rows), 4) if answer_rows else 0.0,
        "document_coverage": round(sum(source in available_sources for source in expected_documents) / len(expected_documents), 4) if expected_documents else 0.0,
        "latency_p50_ms": round(_percentile(latencies, 50), 2),
        "latency_p95_ms": round(_percentile(latencies, 95), 2),
        "cases": rows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _unique_sources(results: list[SearchResult]) -> list[str]:
    sources: list[str] = []
    for result in results:
        source = result.chunk.source_path
        if source not in sources:
            sources.append(source)
    return sources
