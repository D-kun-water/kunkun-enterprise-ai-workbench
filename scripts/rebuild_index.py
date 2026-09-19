"""Rebuild the local knowledge index and print its status."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core import get_runtime


if __name__ == "__main__":
    print(json.dumps(get_runtime().bootstrap(force_rebuild=True), ensure_ascii=False, indent=2))
