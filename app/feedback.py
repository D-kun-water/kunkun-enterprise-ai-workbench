"""Small anonymous feedback store for the current MVP."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.utils import ensure_directory

ALLOWED_RATINGS = {"up", "down"}
ALLOWED_REASONS = {"答非所问", "没有依据", "制度过期", "未解决"}

# The local service runs in one process. Share locks across store instances so
# concurrent API threads cannot overwrite each other's read-modify-write cycle.
# Multi-worker deployment requires a transactional database, not this store.
_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(self.path, threading.RLock())

    def record(self, question: str, answer: str, rating: str, reason: str = "") -> dict[str, str]:
        if rating not in ALLOWED_RATINGS:
            raise ValueError("rating 必须是 up 或 down")
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            raise ValueError("问题和回答不能为空")
        if rating == "up":
            reason = ""
        elif reason and reason not in ALLOWED_REASONS:
            raise ValueError("reason 不是支持的点踩原因")
        record = {
            "feedback_id": uuid4().hex,
            "question": question.strip(),
            "answer": answer.strip(),
            "rating": rating,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            rows = self._load()
            rows.append(record)
            ensure_directory(self.path.parent)
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return record

    def summary(self) -> dict[str, object]:
        with self._lock:
            rows = self._load()
        reasons: dict[str, int] = {}
        for row in rows:
            reason = str(row.get("reason", ""))
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "total": len(rows),
            "up": sum(row.get("rating") == "up" for row in rows),
            "down": sum(row.get("rating") == "down" for row in rows),
            "reasons": reasons,
            "unresolved": sum(row.get("reason") == "未解决" for row in rows),
        }

    def _load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError("反馈文件结构无效，已保留原文件，请先核查或恢复备份")
        return payload
