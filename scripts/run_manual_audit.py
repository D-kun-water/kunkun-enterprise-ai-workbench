"""采集真实模型的十题回答；结果需人工逐题判定，不自动判为通过。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import RuntimeContainer, Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/acceptance/policy_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/manual-audit.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    # Keep feedback, contract access metadata and indexes out of the live state.
    with tempfile.TemporaryDirectory(prefix="knowledgeops_audit_") as directory:
        settings = Settings()
        temporary = Path(directory)
        settings.processed_dir = temporary / "processed"
        settings.faiss_dir = temporary / "faiss"
        settings.contract_data_dir = temporary / "contracts"
        runtime = RuntimeContainer(settings)
        runtime.bootstrap()
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_provider": runtime.effective_embedding_provider,
            "document_count": len(runtime.documents),
            "chunk_count": len(runtime.chunks),
            "retrieval_evaluation": runtime.evaluate(),
            "judgement": "待人工复核，不能作为自动通过率",
            "cases": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for case in cases:
            start = time.perf_counter()
            result = runtime.ask(case["question"])
            row = {**case, **result, "elapsed_seconds": round(time.perf_counter() - start, 2)}
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"id": case["id"], "answer": result["answer"], "seconds": row["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
