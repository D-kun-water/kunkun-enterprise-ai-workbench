from pathlib import Path

import numpy as np
import pytest

from app.ingestion import TextChunk
from app.models import LocalHashEmbeddings
from app.rag import FaissIndexWrapper, HybridRetriever, OptionalReranker, SearchResult


def _sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            "vpn-0",
            "vpn",
            "VPN 指南",
            "IT",
            "故障处理",
            "vpn.md",
            "VPN-403 表示账号未完成 MFA，请联系 IT-SERVICE。",
            0,
        ),
        TextChunk(
            "leave-0",
            "leave",
            "休假制度",
            "HR",
            "年假",
            "leave.md",
            "员工转正后可以按剩余额度申请年假，连续休假需要提前审批。",
            0,
        ),
        TextChunk(
            "invoice-0",
            "invoice",
            "发票制度",
            "财务",
            "报销",
            "invoice.md",
            "费用报销需要提交合规发票和支付记录。",
            0,
        ),
    ]


def test_hybrid_retrieval_handles_codes_and_natural_language() -> None:
    retriever = HybridRetriever.from_chunks(_sample_chunks(), LocalHashEmbeddings(dim=128))

    code_result = retriever.retrieve("VPN-403 怎么处理？")
    leave_result = retriever.retrieve("转正以后能申请休假吗？")

    assert code_result["final_results"][0].chunk.document_id == "vpn"
    assert leave_result["final_results"][0].chunk.document_id == "leave"


def test_numpy_vector_index_round_trip(tmp_path: Path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FaissIndexWrapper.from_vectors(vectors, prefer_faiss=False)

    index.save(tmp_path)
    loaded = FaissIndexWrapper.load(tmp_path, prefer_faiss=False)

    assert loaded.search([1.0, 0.0], top_k=1)[0][0] == 0
    assert loaded.backend == "numpy"


def test_empty_query_returns_no_results() -> None:
    retriever = HybridRetriever.from_chunks(_sample_chunks(), LocalHashEmbeddings(dim=64))

    response = retriever.retrieve("   ")

    assert response["final_results"] == []


def test_query_embedding_failure_retains_bm25_and_reports_degradation(monkeypatch):
    retriever = HybridRetriever.from_chunks(_sample_chunks(), LocalHashEmbeddings(dim=64))

    def fail_query(_query):
        raise ConnectionError("embedding unavailable")

    monkeypatch.setattr(retriever.embeddings, "embed_query", fail_query)
    result = retriever.retrieve("VPN-403 怎么处理？")
    assert result["final_results"][0].chunk.document_id == "vpn"
    assert result["vector_results"] == []
    assert result["vector_error"]


def test_vector_load_requires_metadata_when_validating_fingerprint(tmp_path):
    np.save(tmp_path / "vectors.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="元数据"):
        FaissIndexWrapper.load(tmp_path, corpus_fingerprint="expected")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_vector_index_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="有限"):
        FaissIndexWrapper.from_vectors([[value, 1.0]], prefer_faiss=False)


def test_numpy_save_removes_stale_faiss_artifact(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    stale_index = faiss.IndexFlatIP(2)
    stale_index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    faiss.write_index(stale_index, str(tmp_path / "index.faiss"))
    assert (tmp_path / "index.faiss").exists()

    numpy_index = FaissIndexWrapper.from_vectors(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        prefer_faiss=False,
    )
    numpy_index.save(tmp_path)

    loaded = FaissIndexWrapper.load(tmp_path, prefer_faiss=True)

    assert loaded.backend == "faiss"
    assert not (tmp_path / "index.faiss").exists()


def test_vector_load_rebuilds_faiss_from_vectors_not_stale_file(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FaissIndexWrapper.from_vectors(vectors, prefer_faiss=True)
    index.save(tmp_path, corpus_fingerprint="current-corpus")

    stale_index = faiss.IndexFlatIP(2)
    stale_index.add(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    faiss.write_index(stale_index, str(tmp_path / "index.faiss"))

    loaded = FaissIndexWrapper.load(
        tmp_path,
        prefer_faiss=True,
        corpus_fingerprint="current-corpus",
    )

    assert loaded.backend == "faiss"
    assert loaded.search([1.0, 0.0], top_k=1)[0][0] == 0


def test_reranker_clears_previous_error_after_recovery() -> None:
    class WorkingModel:
        def predict(self, pairs):
            return np.asarray([0.0 for _ in pairs])

    reranker = OptionalReranker(enabled=True)
    reranker._model = WorkingModel()
    reranker.error = "previous failure"

    reranker.rerank("问题", [SearchResult(_sample_chunks()[0], 1.0, "hybrid", 1)])

    assert reranker.error == ""


def test_reranker_keeps_zero_scores_in_normal_order() -> None:
    class WorkingModel:
        def predict(self, pairs):
            return np.asarray([0.0, -1.0])

    reranker = OptionalReranker(enabled=True)
    reranker._model = WorkingModel()
    results = [
        SearchResult(_sample_chunks()[0], 1.0, "hybrid", 1),
        SearchResult(_sample_chunks()[1], 0.5, "hybrid", 2),
    ]

    ordered = reranker.rerank("问题", results)

    assert ordered[0].rerank_score == 0.0


def test_reranker_does_not_retry_after_a_model_load_failure(monkeypatch) -> None:
    import builtins

    import_calls = 0
    original_import = builtins.__import__

    def fail_reranker_import(name, *args, **kwargs):
        nonlocal import_calls
        if name == "sentence_transformers":
            import_calls += 1
            raise ImportError("model unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_reranker_import)
    reranker = OptionalReranker(enabled=True)
    results = [SearchResult(_sample_chunks()[0], 1.0, "hybrid", 1)]

    reranker.rerank("VPN-403", results)
    reranker.rerank("VPN-403", results)

    assert import_calls == 1
    assert "model unavailable" in reranker.error


def test_vector_load_rejects_metadata_count_mismatch(tmp_path: Path) -> None:
    index = FaissIndexWrapper.from_vectors(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        prefer_faiss=False,
    )
    index.save(tmp_path)
    metadata_path = tmp_path / "vector_meta.json"
    metadata = metadata_path.read_text(encoding="utf-8").replace('"count": 1', '"count": 2')
    metadata_path.write_text(metadata, encoding="utf-8")

    with pytest.raises(ValueError, match="向量文件与元数据不一致"):
        FaissIndexWrapper.load(tmp_path, prefer_faiss=False)


def test_vector_load_rejects_different_corpus_fingerprint(tmp_path: Path) -> None:
    index = FaissIndexWrapper.from_vectors(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        prefer_faiss=False,
    )
    index.save(tmp_path, corpus_fingerprint="original-corpus")

    with pytest.raises(ValueError, match="语料指纹不一致"):
        FaissIndexWrapper.load(
            tmp_path,
            prefer_faiss=False,
            corpus_fingerprint="replacement-corpus",
        )
