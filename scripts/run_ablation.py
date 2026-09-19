"""Ablation comparison: corpus adaptation layer on vs off.

Runs the fixed evaluation twice — with the corpus-specific adaptation
layer enabled and disabled — and prints the difference. Retrieval metrics
(Hit@k / MRR / coverage) are unaffected by the switch; the evidence-gate
metrics (refusal_accuracy, passed/failed) show how much of the answer
quality comes from corpus-tuned rules versus the generic evidence layer.

The second section is an answer-quality regression over spoken-language
questions (``data/answer_quality/answer_quality_cases.json``): each case declares
the expected answer/refusal behavior with and without the adaptation layer,
making the "回答质量" difference reproducible instead of anecdotal.

Usage:
    python scripts/run_ablation.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jieba  # noqa: E402
import logging  # noqa: E402

jieba.setLogLevel(logging.ERROR)  # keep PowerShell stderr clean

from app.corpus_adaptations import DISABLE_ENV_VAR  # noqa: E402
from app.agent import compose_local_answer  # noqa: E402
from app.core import RuntimeContainer, Settings  # noqa: E402
from app.evaluation import run_evaluation  # noqa: E402
from app.answer_quality import evaluate_answer_quality, load_answer_quality_cases  # noqa: E402

_METRIC_KEYS = (
    "total_cases",
    "hit_at_1",
    "hit_at_3",
    "mrr",
    "passed_count",
    "failed_count",
    "refusal_accuracy",
    "refusal_baseline_accuracy",
    "citation_coverage",
    "document_coverage",
)


def _format(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _run(container: RuntimeContainer) -> None:
    cases = container.evaluation_cases
    available_sources = {
        document.source_path for document in container.documents if document.status == "ready"
    }
    retriever = container.retriever
    if retriever is None:
        raise RuntimeError("检索器未初始化，请先运行 scripts/rebuild_index.py")

    os.environ.pop(DISABLE_ENV_VAR, None)
    on = run_evaluation(retriever, cases, available_sources)

    os.environ[DISABLE_ENV_VAR] = "1"
    off = run_evaluation(retriever, cases, available_sources)
    os.environ.pop(DISABLE_ENV_VAR, None)

    print("指标对比（开启适配层 vs 关闭适配层）")
    print(f"{'指标':<28}{'开启':>12}{'关闭':>12}")
    for key in _METRIC_KEYS:
        print(f"{key:<28}{_format(on[key]):>12}{_format(off[key]):>12}")

    on_rows = {row["case_id"]: row for row in on["cases"]}
    off_rows = {row["case_id"]: row for row in off["cases"]}
    changed = [
        case_id
        for case_id in on_rows
        if on_rows[case_id].get("failure_type") != off_rows[case_id].get("failure_type")
        or on_rows[case_id].get("refusal_correct") != off_rows[case_id].get("refusal_correct")
    ]
    print(f"\n行为变化的案例（{len(changed)} 个）")
    for case_id in sorted(changed):
        row_on = on_rows[case_id]
        row_off = off_rows[case_id]
        print(
            f"- {case_id}: {row_on['failure_type']} (开启) → "
            f"{row_off['failure_type']} (关闭)"
        )

    quality_cases = load_answer_quality_cases()
    if quality_cases:
        # 回答质量消融验证的是证据选择与语料适配层，不应受本机模型措辞波动影响。
        # 产品问答仍使用 .env 配置的模型；这里只固定离线回归的回答组织方式。
        if container.assistant is None:
            raise RuntimeError("问答助手未初始化")
        container.assistant._compose_llm_answer = compose_local_answer
        os.environ.pop(DISABLE_ENV_VAR, None)
        quality_on = evaluate_answer_quality(container, quality_cases, adaptation_enabled=True)
        os.environ[DISABLE_ENV_VAR] = "1"
        quality_off = evaluate_answer_quality(container, quality_cases, adaptation_enabled=False)
        os.environ.pop(DISABLE_ENV_VAR, None)

        print("\n回答质量回归（口语问法，开启 vs 关闭适配层）")
        print(
            f"开启适配层：{quality_on['passed']}/{quality_on['passed'] + quality_on['failed']} 符合预期"
            f"（有效回答 {quality_on['valid_answers']} · 预期拒答 {quality_on['expected_refusals']}"
            f" · 拒答通过 {quality_on['refusal_passed']}）"
        )
        print(
            f"关闭适配层：{quality_off['passed']}/{quality_off['passed'] + quality_off['failed']} 符合预期"
            f"（有效回答 {quality_off['valid_answers']} · 预期拒答 {quality_off['expected_refusals']}"
            f" · 拒答通过 {quality_off['refusal_passed']}"
            f" · 仅记录 {quality_off['recorded']}）"
        )
        for row in quality_on["rows"]:
            counterpart = next(r for r in quality_off["rows"] if r["case_id"] == row["case_id"])
            status = "PASS" if row["passed"] else "FAIL"
            counterpart_status = "PASS" if counterpart["passed"] else ("REC" if counterpart["passed"] is None else "FAIL")
            print(f"- [{status}] {row['question']}")
            print(f"    开启: {row['answer_preview']}")
            print(f"    关闭[{counterpart_status}]: {counterpart['answer_preview']}")


def main() -> None:
    previous = os.environ.get(DISABLE_ENV_VAR)
    try:
        with tempfile.TemporaryDirectory(prefix="knowledgeops_ablation_") as directory:
            temporary = Path(directory)
            settings = Settings()
            settings.processed_dir = temporary / "processed"
            settings.faiss_dir = temporary / "faiss"
            settings.contract_data_dir = temporary / "contracts"
            container = RuntimeContainer(settings)
            container.bootstrap()
            _run(container)
    finally:
        if previous is None:
            os.environ.pop(DISABLE_ENV_VAR, None)
        else:
            os.environ[DISABLE_ENV_VAR] = previous


if __name__ == "__main__":
    main()
