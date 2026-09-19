"""Hybrid vector/BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from app.ingestion import TextChunk
from app.utils import ensure_directory, tokenize_zh


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float
    source: str
    rank: int
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked result IDs without requiring comparable raw score scales."""

    if k < 0:
        raise ValueError("RRF 参数 k 不能小于 0")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class FaissIndexWrapper:
    """Inner-product vector index using FAISS when available, NumPy otherwise."""

    def __init__(self, vectors: np.ndarray, backend: str, index: Any = None) -> None:
        self.vectors = self._normalize_rows(vectors)
        self.backend = backend
        self.index = index
        self.dimension = int(self.vectors.shape[1]) if self.vectors.ndim == 2 and self.vectors.size else 0

    @classmethod
    def from_vectors(cls, vectors: Sequence[Sequence[float]], prefer_faiss: bool = True) -> "FaissIndexWrapper":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("向量必须是二维矩阵")
        matrix = cls._normalize_rows(matrix)

        if prefer_faiss and matrix.shape[0] > 0:
            try:
                import faiss

                index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                return cls(matrix, "faiss", index)
            except (ImportError, OSError):
                pass
        return cls(matrix, "numpy")

    def search(self, query_vector: Sequence[float], top_k: int = 10) -> list[tuple[int, float]]:
        if top_k <= 0 or self.vectors.shape[0] == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(f"查询向量维度 {query.shape[1]} 与索引维度 {self.dimension} 不一致")
        query = self._normalize_rows(query)
        if not np.any(query):
            return []

        limit = min(top_k, self.vectors.shape[0])
        if self.backend == "faiss" and self.index is not None:
            scores, indices = self.index.search(query, limit)
            return [
                (int(index), float(score))
                for index, score in zip(indices[0].tolist(), scores[0].tolist())
                if index >= 0
            ]

        scores = (self.vectors @ query[0]).astype(float)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(index), float(scores[index])) for index in order]

    def save(self, directory: Path, *, corpus_fingerprint: str = "") -> None:
        directory = ensure_directory(Path(directory))
        np.save(directory / "vectors.npy", self.vectors)
        faiss_path = directory / "index.faiss"
        metadata = {
            "backend": self.backend,
            "dimension": self.dimension,
            "count": int(self.vectors.shape[0]),
            "corpus_fingerprint": corpus_fingerprint,
        }
        (directory / "vector_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        faiss_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        directory: Path,
        prefer_faiss: bool = True,
        *,
        corpus_fingerprint: str | None = None,
    ) -> "FaissIndexWrapper":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy").astype(np.float32)
        metadata_path = directory / "vector_meta.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_count = int(metadata.get("count", -1))
            expected_dimension = int(metadata.get("dimension", -1))
            actual_dimension = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0
            if expected_count != int(vectors.shape[0]) or expected_dimension != actual_dimension:
                raise ValueError("向量文件与元数据不一致")
            if corpus_fingerprint is not None and metadata.get("corpus_fingerprint") != corpus_fingerprint:
                raise ValueError("向量文件与语料指纹不一致")
        elif corpus_fingerprint is not None:
            raise ValueError("向量元数据缺失，无法校验语料指纹")
        return cls.from_vectors(vectors, prefer_faiss=prefer_faiss)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("向量必须全部为有限数值")
        if matrix.ndim != 2:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


class BM25Index:
    """Compact BM25 implementation with no external runtime service."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in tokenized_corpus]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(document) for document in tokenized_corpus]
        document_frequencies: Counter[str] = Counter()
        for document in tokenized_corpus:
            document_frequencies.update(set(document))
        count = len(tokenized_corpus)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        if not query_tokens or top_k <= 0 or not self.corpus:
            return []
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.lengths[index]
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / (self.average_length or 1.0)
                )
                scores[index] += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator

        positive = [index for index in np.argsort(-scores, kind="stable") if scores[index] > 0]
        return [(int(index), float(scores[index])) for index in positive[:top_k]]


class OptionalReranker:
    """Lazy cross-encoder reranker that never blocks the base retrieval path."""

    def __init__(self, enabled: bool = False, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.error = ""
        self._model: Any = None

    @property
    def effective(self) -> bool:
        return self.enabled and self._model is not None and not self.error

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not self.enabled or not results:
            return results
        if self._model is None and self.error:
            return results
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict([(query, result.chunk.content) for result in results])
            self.error = ""
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
            ordered = sorted(
                results,
                key=lambda result: (
                    result.rerank_score if result.rerank_score is not None else float("-inf")
                ),
                reverse=True,
            )
            for rank, result in enumerate(ordered, start=1):
                result.rank = rank
                result.score = float(result.rerank_score) if result.rerank_score is not None else 0.0
                result.source = "reranker"
            return ordered
        except Exception as exc:  # Optional provider failure must preserve RRF results.
            self.error = str(exc)
            return results


class HybridRetriever:
    """Search chunks using vector similarity and BM25, then combine via RRF."""

    def __init__(
        self,
        chunks: list[TextChunk],
        embeddings: EmbeddingModel,
        vector_index: FaissIndexWrapper,
        bm25_index: BM25Index,
        *,
        retrieval_top_k: int = 10,
        rrf_k: int = 60,
        reranker: OptionalReranker | None = None,
    ) -> None:
        if vector_index.vectors.shape[0] != len(chunks):
            raise ValueError("向量数量与文本块数量不一致")
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.retrieval_top_k = retrieval_top_k
        self.rrf_k = rrf_k
        self.reranker = reranker or OptionalReranker(False)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        embeddings: EmbeddingModel,
        *,
        retrieval_top_k: int = 10,
        rrf_k: int = 60,
        enable_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        prefer_faiss: bool = True,
    ) -> "HybridRetriever":
        vectors = embeddings.embed_documents([chunk.content for chunk in chunks])
        vector_index = FaissIndexWrapper.from_vectors(vectors, prefer_faiss=prefer_faiss)
        bm25_index = BM25Index([tokenize_zh(chunk.content) for chunk in chunks])
        return cls(
            chunks,
            embeddings,
            vector_index,
            bm25_index,
            retrieval_top_k=retrieval_top_k,
            rrf_k=rrf_k,
            reranker=OptionalReranker(enable_reranker, reranker_model),
        )

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query or top_k <= 0:
            return self._response([], [], [])

        candidate_limit = max(top_k, self.retrieval_top_k)
        vector_error = ""
        try:
            vector_pairs = self.vector_index.search(self.embeddings.embed_query(query), candidate_limit)
        except Exception as exc:
            # Keep the independent lexical path available on a provider outage.
            # Evidence gating still applies; degraded recall is not an answer.
            vector_pairs = []
            vector_error = f"向量检索暂不可用（{type(exc).__name__}），本次仅使用关键词检索"
        bm25_pairs = self.bm25_index.search(tokenize_zh(query), candidate_limit)

        vector_results = self._raw_results(vector_pairs, "vector")
        bm25_results = self._raw_results(bm25_pairs, "bm25")
        fused = reciprocal_rank_fusion(
            [
                [result.chunk.chunk_id for result in vector_results],
                [result.chunk.chunk_id for result in bm25_results],
            ],
            self.rrf_k,
        )

        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        vector_scores = {result.chunk.chunk_id: result.score for result in vector_results}
        bm25_scores = {result.chunk.chunk_id: result.score for result in bm25_results}
        final_results: list[SearchResult] = []
        for rank, (chunk_id, score) in enumerate(fused[:candidate_limit], start=1):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            final_results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    source="hybrid",
                    rank=rank,
                    vector_score=vector_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    rrf_score=float(score),
                )
            )

        final_results = self.reranker.rerank(query, final_results)[:top_k]
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank
        response = self._response(final_results, vector_results, bm25_results)
        response["vector_error"] = vector_error
        return response

    def _raw_results(self, pairs: list[tuple[int, float]], source: str) -> list[SearchResult]:
        return [
            SearchResult(self.chunks[index], score, source, rank)
            for rank, (index, score) in enumerate(pairs, start=1)
            if 0 <= index < len(self.chunks)
        ]

    def _response(
        self,
        final_results: list[SearchResult],
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> dict[str, Any]:
        return {
            "final_results": final_results,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "reranker_error": self.reranker.error,
            "vector_backend": self.vector_index.backend,
        }
