import json
from pathlib import Path

import pytest

from app.agent import KnowledgeAssistant
from app.core import RuntimeContainer, Settings, _DisabledEmbeddings
from app.ingestion import DocumentLoader, IngestionStore, KnowledgeDocument, TextChunk, TextChunker
from app.models import LocalHashEmbeddings
from app.rag import HybridRetriever


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        evaluation_dir=tmp_path / "evaluation",
        processed_dir=tmp_path / "processed",
        faiss_dir=tmp_path / "faiss",
        contract_data_dir=tmp_path / "contracts",
        embedding_provider="ollama",
        llm_provider="ollama",
    )


@pytest.mark.parametrize(
    ("effective_provider", "persisted_embeddings", "expected_backend"),
    [
        ("local_fallback", LocalHashEmbeddings(), "faiss"),
        ("bm25_only", _DisabledEmbeddings(), "disabled"),
    ],
)
def test_bootstrap_reuses_effective_embedding_for_persisted_index(
    tmp_path: Path,
    monkeypatch,
    effective_provider: str,
    persisted_embeddings,
    expected_backend: str,
) -> None:
    settings = _settings(tmp_path)
    settings.raw_data_dir.mkdir(parents=True)
    settings.evaluation_dir.mkdir(parents=True)
    (settings.raw_data_dir / "policy.md").write_text(
        "# 员工制度\n\n## 补卡规则\n忘记打卡应在三个工作日内提交补卡。",
        encoding="utf-8",
    )

    runtime = RuntimeContainer(settings)
    documents = DocumentLoader().load_directory(settings.raw_data_dir)
    chunks = TextChunker(settings.chunk_size, settings.chunk_overlap).split_documents(documents)
    retriever = HybridRetriever.from_chunks(chunks, persisted_embeddings, prefer_faiss=False)
    if effective_provider == "bm25_only":
        retriever.vector_index.backend = "disabled"
    IngestionStore(settings.processed_dir).save(
        documents,
        chunks,
        {
            "fingerprint": runtime._corpus_fingerprint(),
            "built_at": "2026-08-06T00:00:00+00:00",
            "effective_embedding_provider": effective_provider,
        },
    )
    retriever.vector_index.save(
        settings.faiss_dir,
        corpus_fingerprint=runtime._corpus_fingerprint(),
    )

    provider_calls = 0

    def fail_if_external_provider_is_built(_settings: Settings):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("持久化索引为降级模式时不应创建外部 Embedding")

    monkeypatch.setattr("app.core.build_embedding_model", fail_if_external_provider_is_built)

    status = runtime.bootstrap()

    assert provider_calls == 0
    assert status["effective_embedding_provider"] == effective_provider
    if expected_backend == "faiss":
        assert status["vector_backend"] in {"faiss", "numpy"}
    else:
        assert status["vector_backend"] == expected_backend
    assert status["errors"] == {}
    assert runtime.retriever is not None
    assert runtime.retriever.retrieve("忘记打卡", top_k=1)["final_results"]


class FailingChatClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("Ollama connection failed")


def _runtime_with_travel_policy() -> RuntimeContainer:
    chunk = TextChunk(
        "travel-0",
        "travel",
        "差旅报销制度",
        "财务",
        "市内交通",
        "travel.md",
        "出差期间因公务产生的出租车费用可以报销，但需要提供发票。",
        0,
    )
    document = KnowledgeDocument(
        "travel",
        "差旅报销制度",
        "财务",
        "travel.md",
        chunk.content,
        ["市内交通"],
    )
    retriever = HybridRetriever.from_chunks([chunk], LocalHashEmbeddings(), prefer_faiss=False)
    runtime = RuntimeContainer(Settings(llm_provider="ollama"))
    runtime.documents = [document]
    runtime.chunks = [chunk]
    runtime.retriever = retriever
    runtime.assistant = KnowledgeAssistant(
        retriever,
        [document],
        llm_provider="ollama",
        llm_client=FailingChatClient(),
    )
    return runtime


def test_status_reports_effective_llm_fallback_after_runtime_failure() -> None:
    runtime = _runtime_with_travel_policy()

    runtime.ask("出差打车可以报销吗？")
    status = runtime.status()

    assert status["status"] == "degraded"
    assert status["effective_llm_provider"] == "unavailable"
    assert "Ollama connection failed" in status["errors"]["llm_runtime"]


def test_status_reports_reranker_runtime_failure() -> None:
    runtime = _runtime_with_travel_policy()
    assert runtime.retriever is not None
    runtime.settings.enable_reranker = True
    runtime.retriever.reranker.enabled = True
    runtime.retriever.reranker.error = "reranker model unavailable"

    status = runtime.status()

    assert status["status"] == "degraded"
    assert status["reranker_effective"] is False
    assert "reranker model unavailable" in status["errors"]["reranker_runtime"]


def test_status_does_not_report_reranker_effective_before_model_load() -> None:
    runtime = _runtime_with_travel_policy()
    assert runtime.retriever is not None
    runtime.settings.enable_reranker = True
    runtime.retriever.reranker.enabled = True

    status = runtime.status()

    assert status["reranker_enabled"] is True
    assert status["reranker_effective"] is False


def test_embedding_fallback_does_not_report_reranker_as_enabled(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings.enable_reranker = True
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "policy.md").write_text(
        "# 制度\n\n## 规则\n员工应通过 IT-SERVICE 提交申请。",
        encoding="utf-8",
    )

    def fail_embedding_model(_settings: Settings):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr("app.core.build_embedding_model", fail_embedding_model)

    status = RuntimeContainer(settings).bootstrap(force_rebuild=True)

    assert status["effective_embedding_provider"] == "local_fallback"
    assert status["reranker_enabled"] is False
    assert status["reranker_effective"] is False


def test_failed_rebuild_preserves_previous_runtime_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.raw_data_dir.mkdir(parents=True)
    policy_path = settings.raw_data_dir / "policy.md"
    policy_path.write_text(
        "# 员工制度\n\n## 补卡规则\n忘记打卡后应在三个工作日内补卡。",
        encoding="utf-8",
    )

    runtime = RuntimeContainer(settings)
    runtime.bootstrap(force_rebuild=True)
    previous_documents = runtime.documents
    previous_chunks = runtime.chunks
    previous_retriever = runtime.retriever
    previous_assistant = runtime.assistant

    policy_path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="没有可用于构建索引的文档内容"):
        runtime.bootstrap(force_rebuild=True)

    assert runtime.documents == previous_documents
    assert runtime.chunks == previous_chunks
    assert runtime.retriever is previous_retriever
    assert runtime.assistant is previous_assistant
    status = runtime.status()
    assert status["status"] == "degraded"
    assert "rebuild" in status["errors"]


def test_empty_corpus_bootstraps_in_degraded_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = RuntimeContainer(settings)

    status = runtime.bootstrap()

    assert status["status"] == "degraded"
    assert status["document_count"] == 0
    assert status["chunk_count"] == 0
    assert "没有可用于构建索引的文档内容" in status["errors"]["index"]
    assert runtime.retriever is None
    assert runtime.assistant is None


def test_empty_corpus_question_returns_an_explained_refusal(tmp_path: Path) -> None:
    runtime = RuntimeContainer(_settings(tmp_path))

    response = runtime.ask("公司的股票代码是多少？")

    assert "依据不足" in response["answer"]
    assert response["sources"] == []
    assert response["debug"]["route"] == "unavailable"


def test_document_parse_errors_mark_runtime_degraded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.embedding_provider = "local"
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "good.md").write_text(
        "# 有效制度\n\n## 规则\n有效内容。",
        encoding="utf-8",
    )
    (settings.raw_data_dir / "bad.md").write_text("", encoding="utf-8")

    status = RuntimeContainer(settings).bootstrap(force_rebuild=True)

    assert status["status"] == "degraded"
    assert "documents" in status["errors"]
    assert "bad.md" in status["errors"]["documents"]


def test_bootstrap_derives_bm25_from_loaded_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.embedding_provider = "local"
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "policy.md").write_text(
        "# 制度\n\n## 规则\n第一条规则。\n\n## 流程\n第二条流程。",
        encoding="utf-8",
    )

    RuntimeContainer(settings).bootstrap(force_rebuild=True)
    assert not (settings.processed_dir / "bm25_corpus.json").exists()

    runtime = RuntimeContainer(settings)
    runtime.bootstrap()

    assert runtime.retriever is not None
    assert len(runtime.retriever.bm25_index.corpus) == len(runtime.chunks)


def test_successful_rebuild_clears_recoverable_index_load_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.embedding_provider = "local"
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "policy.md").write_text(
        "# 制度\n\n## 规则\n第一条规则。",
        encoding="utf-8",
    )
    RuntimeContainer(settings).bootstrap(force_rebuild=True)

    metadata_path = settings.faiss_dir / "vector_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["count"] = metadata["count"] + 1
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    status = RuntimeContainer(settings).bootstrap()

    assert status["status"] == "ok"
    assert "index_load" not in status["errors"]


def test_corpus_fingerprint_includes_index_schema_version(tmp_path: Path, monkeypatch) -> None:
    from app import core

    runtime = RuntimeContainer(_settings(tmp_path))
    first = runtime._corpus_fingerprint()
    monkeypatch.setattr(core, "INDEX_SCHEMA_VERSION", "next")

    assert runtime._corpus_fingerprint() != first


def test_corpus_fingerprint_streams_files_instead_of_reading_them_whole(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "large-policy.pdf").write_bytes(b"policy" * 1024)
    runtime = RuntimeContainer(settings)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("语料指纹不应一次性读取整份文件")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert runtime._corpus_fingerprint()


def test_contract_upload_is_separate_from_policy_documents_and_admin_can_list(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()

    created = runtime.upload_contract(
        "采购合同.md",
        "# 采购合同\n\n## 付款\n付款节点：验收后30日内付款。",
        ["legal"],
        "legal",
    )

    assert created["knowledge_base_id"] == "contracts"
    assert runtime.document_summaries() == []
    assert runtime.list_contracts("admin")[0]["title"] == "采购合同"
    assert runtime.list_contracts("employee") == []


def test_contract_upload_accepts_extractable_pdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Purchase Contract\nPayment due within 30 days after acceptance.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    created = runtime.upload_contract(
        "采购合同.pdf",
        "",
        ["legal", "business"],
        "legal",
        file_bytes=pdf_bytes,
    )

    assert created["knowledge_base_id"] == "contracts"
    assert runtime.list_contracts("business")[0]["title"]
    reviewed = runtime.review_contract(created["document_id"], "legal")
    assert "Payment due within 30 days" in reviewed["sources"][0]["content_preview"]


def test_contract_review_hides_unauthorized_document_details(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()
    created = runtime.upload_contract(
        "敏感合同.md",
        "# 敏感合同\n\n付款节点：支付100000元。",
        ["legal"],
        "legal",
    )

    with pytest.raises(KeyError):
        runtime.review_contract(created["document_id"], "employee")

    reviewed = runtime.review_contract(created["document_id"], "legal")
    assert reviewed["sources"][0]["content_preview"]
    assert "100000" in reviewed["sources"][0]["content_preview"]


def test_contract_review_without_query_limits_source_previews_but_checks_full_document(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    settings.answer_top_k = 2
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()
    long_contract = "# 长合同\n\n合同期限：2026年1月1日至2026年12月31日。\n" + "\n\n".join(
        f"## 条款{i}\n付款节点{i}：验收后{i + 1}日内付款。" for i in range(20)
    )
    created = runtime.upload_contract("长合同.md", long_contract, ["legal"], "legal")

    reviewed = runtime.review_contract(created["document_id"], "legal")

    assert len(reviewed["sources"]) <= settings.answer_top_k
    assert reviewed["review_counts"]["checked"] == 12
    assert reviewed["fields"][0]["evidence"]


def test_contract_upload_rejects_employee_actor(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()

    with pytest.raises(PermissionError):
        runtime.upload_contract("合同.md", "# 合同\n\n付款节点：验收后付款。", ["employee"], "employee")


def test_contract_upload_rejects_duplicate_name_without_overwriting_original(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()
    original = "# 原合同\n\n付款节点：验收后30日内付款。"
    runtime.upload_contract("采购合同.md", original, ["legal"], "legal")

    with pytest.raises(ValueError, match="已存在同名合同"):
        runtime.upload_contract(
            "采购合同.md",
            "# 替换内容\n\n付款节点：立即付款。",
            ["employee"],
            "legal",
        )

    assert (settings.contract_data_dir / "采购合同.md").read_text(encoding="utf-8") == original
    assert runtime.list_contracts("legal")
    assert runtime.list_contracts("employee") == []


def test_invalid_contract_upload_leaves_no_file_permission_or_staging_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.contract_data_dir = tmp_path / "contracts"
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()

    with pytest.raises(ValueError, match="文档内容为空"):
        runtime.upload_contract("空合同.txt", "   ", ["legal"], "legal", file_bytes=b"   ")

    assert list(settings.contract_data_dir.iterdir()) == []
    permissions_path = settings.processed_dir / "contract_permissions.json"
    assert not permissions_path.exists() or "空合同.txt" not in permissions_path.read_text(encoding="utf-8")


def test_contract_review_includes_related_policy_sources_without_mixing_contract_access(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.embedding_provider = "local"
    settings.contract_data_dir = tmp_path / "contracts"
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "policy.md").write_text(
        "# 采购管理制度\n\n## 验收\n采购交付后应留存验收记录和付款审批依据。",
        encoding="utf-8",
    )
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()
    created = runtime.upload_contract(
        "采购合同.md",
        "# 采购合同\n\n交付与验收：供应商交付后十日内验收。",
        ["legal"],
        "legal",
    )

    result = runtime.review_contract(created["document_id"], "legal", query="验收记录")

    assert result["sources"][0]["source_type"] == "contract"
    assert result["related_policy_sources"][0]["title"] == "采购管理制度"
    assert runtime._related_policy_sources("咖啡机维修可以报销吗") == []


@pytest.mark.parametrize("payload", ['{"合同.md": null}', '{"合同.md": "legal"}', '["legal"]', 'invalid'])
def test_invalid_contract_permissions_fail_closed_without_crashing_policy_runtime(tmp_path, payload):
    settings = _settings(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "contract_permissions.json").write_text(payload, encoding="utf-8")
    runtime = RuntimeContainer(settings)
    status = runtime.bootstrap()
    assert "contracts" in status["errors"]
    assert runtime.list_contracts("admin") == []
    with pytest.raises(KeyError):
        runtime.review_contract("合同", "admin")


def test_contract_with_unreadable_content_cannot_generate_review(tmp_path):
    settings = _settings(tmp_path)
    settings.contract_data_dir.mkdir(parents=True)
    (settings.contract_data_dir / "empty.txt").write_text("", encoding="utf-8")
    runtime = RuntimeContainer(settings)
    runtime.bootstrap()
    with pytest.raises(KeyError):
        runtime.review_contract("empty", "legal")
