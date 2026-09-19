"""启动鲲坤科技企业 AI 协作工作台前端（端口 8501）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT_DIR / "frontend" / "streamlit_app.py"),
                "--server.address=127.0.0.1",
                "--server.port=8501",
                "--browser.gatherUsageStats=false",
            ],
            cwd=ROOT_DIR,
        )
    )
