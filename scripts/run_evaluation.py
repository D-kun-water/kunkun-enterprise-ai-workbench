"""Run the fixed retrieval evaluation from the command line."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import RuntimeContainer, Settings


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="knowledgeops_evaluation_") as directory:
        temporary = Path(directory)
        settings = Settings()
        settings.processed_dir = temporary / "processed"
        settings.faiss_dir = temporary / "faiss"
        settings.contract_data_dir = temporary / "contracts"
        runtime = RuntimeContainer(settings)
        runtime.bootstrap()
        report = runtime.evaluate()
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = [row for row in report["cases"] if row["failure_type"] != "passed"]
    if failed:
        print("\n失败案例：")
        print(json.dumps(failed, ensure_ascii=False, indent=2))
