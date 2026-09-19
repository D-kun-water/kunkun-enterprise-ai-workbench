"""启动鲲坤科技企业 AI 协作工作台 API（端口 8000）。"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
