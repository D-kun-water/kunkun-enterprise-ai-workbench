from app.feedback import FeedbackStore
from concurrent.futures import ThreadPoolExecutor
import json
import pytest


def test_concurrent_feedback_across_store_instances_is_not_lost(tmp_path):
    path = tmp_path / "feedback.json"
    stores = [FeedbackStore(path), FeedbackStore(path)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: stores[i % 2].record(f"问题{i}", "回答", "up"), range(40)))
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 40
    assert len({row["feedback_id"] for row in rows}) == 40


def test_invalid_feedback_file_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text('{"unexpected": "data"}', encoding="utf-8")
    with pytest.raises(ValueError, match="反馈文件"):
        FeedbackStore(path).record("问题", "回答", "up")
    assert json.loads(path.read_text(encoding="utf-8")) == {"unexpected": "data"}


def test_failed_feedback_publish_preserves_previous_file(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "feedback.json"
    store = FeedbackStore(path)
    store.record("问题1", "回答", "up")
    previous = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("publish failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        store.record("问题2", "回答", "up")
    assert path.read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == ["feedback.json"]


def test_feedback_store_records_and_summarizes_ratings(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")

    store.record("忘记打卡怎么补卡？", "请提交补卡申请。", "up")
    store.record("股票代码？", "依据不足", "down", "未解决")

    summary = store.summary()

    assert summary["total"] == 2
    assert summary["up"] == 1
    assert summary["down"] == 1
    assert summary["reasons"] == {"未解决": 1}
    assert summary["unresolved"] == 1


def test_feedback_store_rejects_invalid_rating(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")

    try:
        store.record("问题", "回答", "maybe")
    except ValueError as exc:
        assert "rating" in str(exc)
    else:
        raise AssertionError("invalid rating should fail")


def test_feedback_store_rejects_blank_question_or_answer(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")

    for question, answer in (("   ", "回答"), ("问题", "\n\t")):
        try:
            store.record(question, answer, "up")
        except ValueError as exc:
            assert "不能为空" in str(exc)
        else:
            raise AssertionError("blank feedback fields should fail")
