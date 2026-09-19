"""离线回归使用独立临时目录，不读取私有合同、不改写运行数据。"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.agent import compose_local_answer
from app.core import RuntimeContainer, Settings

_ROOT = Path(__file__).resolve().parent
_TAG_PREFIX = ".pytest_tmp_session_"
_TEMP_ROOT = Path(tempfile.gettempdir()) / "kunkun_pytest_sessions"


def pytest_configure(config) -> None:
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    fresh = _TEMP_ROOT / f"{_TAG_PREFIX}{uuid.uuid4().hex[:12]}"
    if not config.option.basetemp:
        config.option.basetemp = str(fresh)


@pytest.fixture(scope="session")
def deterministic_runtime(tmp_path_factory) -> RuntimeContainer:
    """Build one runtime whose answer assertions do not depend on a live LLM.

    Product runtime still uses the model configured in ``.env``.  Tests that
    assert exact policy facts use the deterministic, evidence-only composer so
    a local model's wording variance cannot make the regression suite flaky.
    """

    temporary = tmp_path_factory.mktemp("runtime")
    container = RuntimeContainer(Settings(
        _env_file=None,
        raw_data_dir=_ROOT / "data" / "raw",
        evaluation_dir=_ROOT / "data" / "evaluation",
        processed_dir=temporary / "processed",
        faiss_dir=temporary / "faiss",
        contract_data_dir=temporary / "contracts",
        embedding_provider="local",
        llm_provider="ollama",
        enable_reranker=False,
    ))
    container.bootstrap()
    if container.assistant is None:
        raise RuntimeError("问答助手初始化失败")
    container.assistant._compose_llm_answer = compose_local_answer
    return container
