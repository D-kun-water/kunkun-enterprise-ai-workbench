"""Application settings, index bootstrap, and shared runtime state."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.agent import INSUFFICIENT_ANSWER, KnowledgeAssistant, has_sufficient_evidence
from app.contracts import review_contract
from app.evaluation import EvaluationCase, load_evaluation_cases, run_evaluation
from app.feedback import FeedbackStore
from app.ingestion import DocumentLoader, IngestionStore, KnowledgeDocument, TextChunk, TextChunker
from app.models import LocalHashEmbeddings, build_chat_client, build_embedding_model
from app.rag import BM25Index, FaissIndexWrapper, HybridRetriever, OptionalReranker
from app.utils import ensure_directory, tokenize_zh


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_SCHEMA_VERSION = "3"


class Settings(BaseSettings):
    """Environment-driven settings for a grounded, model-generated assistant."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "鲲坤科技企业 AI 协作工作台"
    app_version: str = "1.0.0"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    evaluation_dir: Path = BASE_DIR / "data" / "evaluation"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    faiss_dir: Path = BASE_DIR / "storage" / "faiss"
    contract_data_dir: Path = BASE_DIR / "data" / "contracts"

    chunk_size: int = 500
    chunk_overlap: int = 80
    retrieval_top_k: int = 10
    answer_top_k: int = 5
    rrf_k: int = 60

    embedding_provider: str = "local"
    embedding_model: str = "nomic-embed-text"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5"

    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class _DisabledEmbeddings:
    """One-dimensional zero vectors used only when all embeddings fail."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class RuntimeContainer:
    """Own the small in-process runtime used by FastAPI and scripts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.documents: list[KnowledgeDocument] = []
        self.chunks: list[TextChunk] = []
        self.evaluation_cases: list[EvaluationCase] = []
        self.retriever: HybridRetriever | None = None
        self.assistant: KnowledgeAssistant | None = None
        self.built_at = ""
        self.effective_embedding_provider = self.settings.embedding_provider
        self.errors: dict[str, str] = {}
        self.feedback_store = FeedbackStore(self.settings.processed_dir / "feedback.json")
        self.contract_documents: list[KnowledgeDocument] = []
        self.contract_chunks: list[TextChunk] = []
        self._lock = threading.RLock()

    def bootstrap(self, force_rebuild: bool = False) -> dict[str, Any]:
        with self._lock:
            self.errors = {}
            for directory in (
                self.settings.raw_data_dir,
                self.settings.evaluation_dir,
                self.settings.processed_dir,
                self.settings.faiss_dir,
                self.settings.contract_data_dir,
            ):
                ensure_directory(directory)

            fingerprint = self._corpus_fingerprint()
            store = IngestionStore(self.settings.processed_dir)
            loaded = False
            if not force_rebuild and store.exists() and (self.settings.faiss_dir / "vectors.npy").exists():
                try:
                    documents, chunks, metadata = store.load()
                    if metadata.get("fingerprint") == fingerprint:
                        effective_provider = str(
                            metadata.get("effective_embedding_provider", self.settings.embedding_provider)
                        )
                        if effective_provider == "local_fallback":
                            embeddings = LocalHashEmbeddings()
                        elif effective_provider == "bm25_only":
                            embeddings = _DisabledEmbeddings()
                        else:
                            embeddings = build_embedding_model(self.settings)
                        vector_index = FaissIndexWrapper.load(
                            self.settings.faiss_dir,
                            corpus_fingerprint=fingerprint,
                        )
                        if effective_provider == "bm25_only":
                            vector_index.backend = "disabled"
                        retriever = HybridRetriever(
                            chunks,
                            embeddings,
                            vector_index,
                            BM25Index([tokenize_zh(chunk.content) for chunk in chunks]),
                            retrieval_top_k=self.settings.retrieval_top_k,
                            rrf_k=self.settings.rrf_k,
                            reranker=OptionalReranker(
                                self.settings.enable_reranker,
                                self.settings.reranker_model,
                            ),
                        )
                        self.documents = documents
                        self.chunks = chunks
                        self.retriever = retriever
                        self.built_at = str(metadata.get("built_at", ""))
                        self.effective_embedding_provider = effective_provider
                        loaded = True
                except Exception as exc:
                    self.errors["index_load"] = str(exc)

            if not loaded:
                try:
                    self._rebuild_index(store, fingerprint)
                except Exception as exc:
                    self.errors["rebuild"] = str(exc)
                    raise
                self.errors.pop("index_load", None)

            try:
                self.evaluation_cases = load_evaluation_cases(self.settings.evaluation_dir)
            except Exception as exc:
                self.evaluation_cases = []
                self.errors["evaluation"] = str(exc)

            self._load_contracts()

            llm_client = None
            try:
                llm_client = build_chat_client(self.settings)
            except Exception as exc:
                self.errors["llm"] = str(exc)

            if self.retriever is None:
                if self.errors.get("index"):
                    return self.status()
                raise RuntimeError("检索器初始化失败")
            self.assistant = KnowledgeAssistant(
                self.retriever,
                self.documents,
                llm_provider=self.settings.llm_provider,
                llm_client=llm_client,
                answer_top_k=self.settings.answer_top_k,
            )
            return self.status()

    def _rebuild_index(self, store: IngestionStore, fingerprint: str) -> None:
        documents = DocumentLoader().load_directory(self.settings.raw_data_dir)
        chunks = TextChunker(
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        ).split_documents(documents)
        if not chunks:
            if self.documents or self.retriever is not None:
                raise RuntimeError("没有可用于构建索引的文档内容")
            self.documents = documents
            self.chunks = []
            self.errors["index"] = "没有可用于构建索引的文档内容"
            return

        retriever: HybridRetriever
        effective_embedding_provider: str
        try:
            embeddings = build_embedding_model(self.settings)
            retriever = HybridRetriever.from_chunks(
                chunks,
                embeddings,
                retrieval_top_k=self.settings.retrieval_top_k,
                rrf_k=self.settings.rrf_k,
                enable_reranker=self.settings.enable_reranker,
                reranker_model=self.settings.reranker_model,
            )
            effective_embedding_provider = self.settings.embedding_provider
        except Exception as exc:
            self.errors["embedding"] = str(exc)
            try:
                embeddings = LocalHashEmbeddings()
                retriever = HybridRetriever.from_chunks(
                    chunks,
                    embeddings,
                    retrieval_top_k=self.settings.retrieval_top_k,
                    rrf_k=self.settings.rrf_k,
                    enable_reranker=False,
                )
                effective_embedding_provider = "local_fallback"
            except Exception as fallback_exc:  # pragma: no cover - defensive final fallback
                self.errors["embedding_fallback"] = str(fallback_exc)
                embeddings = _DisabledEmbeddings()
                retriever = HybridRetriever.from_chunks(
                    chunks,
                    embeddings,
                    retrieval_top_k=self.settings.retrieval_top_k,
                    rrf_k=self.settings.rrf_k,
                    enable_reranker=False,
                    prefer_faiss=False,
                )
                retriever.vector_index.backend = "disabled"
                effective_embedding_provider = "bm25_only"

        built_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "fingerprint": fingerprint,
            "built_at": built_at,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "effective_embedding_provider": effective_embedding_provider,
        }
        store.save(documents, chunks, metadata)
        retriever.vector_index.save(self.settings.faiss_dir, corpus_fingerprint=fingerprint)

        self.documents = documents
        self.chunks = chunks
        self.retriever = retriever
        self.built_at = built_at
        self.effective_embedding_provider = effective_embedding_provider

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.assistant is None:
            self.bootstrap()
        if self.assistant is None:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "sources": [],
                "debug": {
                    "route": "unavailable",
                    "llm_provider": self.settings.llm_provider,
                    "reason": self.errors.get("index", "知识库当前不可用"),
                },
            }
        return self.assistant.ask(question, history=history)

    def evaluate(self) -> dict[str, Any]:
        if self.retriever is None:
            self.bootstrap()
        if self.retriever is None:
            raise RuntimeError("检索器尚未初始化")
        return run_evaluation(
            self.retriever,
            self.evaluation_cases,
            {document.source_path for document in self.documents if document.status == "ready"},
        )

    def record_feedback(self, question: str, answer: str, rating: str, reason: str = "") -> dict[str, str]:
        return self.feedback_store.record(question, answer, rating, reason)

    def feedback_summary(self) -> dict[str, object]:
        return self.feedback_store.summary()

    def _load_contracts(self) -> None:
        try:
            permissions = self._load_contract_permissions()
        except (ValueError, TypeError):
            # Fail closed for contract access without disabling public policy QA.
            self.contract_documents = []
            self.contract_chunks = []
            self.errors["contracts"] = "合同权限配置无效，合同访问已暂停；请核查配置文件"
            return
        self.errors.pop("contracts", None)
        documents = DocumentLoader().load_directory(self.settings.contract_data_dir)
        for document in documents:
            document.knowledge_base_id = "contracts"
            document.allowed_roles = permissions.get(document.source_path, ["legal"])
        chunks = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap).split_documents(documents)
        self.contract_documents = documents
        self.contract_chunks = chunks

    def _load_contract_permissions(self) -> dict[str, list[str]]:
        path = self.settings.processed_dir / "contract_permissions.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or any(
            not isinstance(name, str) or not isinstance(roles, list)
            or any(role not in {"admin", "legal", "business", "employee"} for role in roles)
            for name, roles in data.items()
        ):
            raise ValueError("合同权限配置格式无效")
        return data

    def _save_contract_permissions(self, permissions: dict[str, list[str]]) -> None:
        ensure_directory(self.settings.processed_dir)
        path = self.settings.processed_dir / "contract_permissions.json"
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(permissions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_role(role: str) -> str:
        if role not in {"admin", "legal", "business", "employee"}:
            raise ValueError("角色必须是 admin、legal、business 或 employee")
        return role

    def upload_contract(
        self,
        filename: str,
        content: str,
        allowed_roles: list[str],
        actor_role: str,
        *,
        file_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            actor_role = self._validate_role(actor_role)
            if actor_role not in {"admin", "legal"}:
                raise PermissionError("只有 admin 或 legal 可以上传合同")
            if not filename or (file_bytes is None and not content.strip()):
                raise ValueError("合同文件名和内容不能为空")
            normalized_roles = sorted({self._validate_role(role) for role in allowed_roles})
            if not normalized_roles:
                raise ValueError("至少需要一个允许角色")
            safe_name = Path(filename).name
            suffix = Path(safe_name).suffix.lower()
            text_suffixes = {".md", ".markdown", ".txt"}
            binary_suffixes = {".pdf", ".docx"}
            if suffix not in text_suffixes | binary_suffixes:
                raise ValueError("合同上传仅支持 .pdf、.docx、.md、.markdown 或 .txt")

            ensure_directory(self.settings.contract_data_dir)
            path = self.settings.contract_data_dir / safe_name
            if path.exists():
                raise ValueError("已存在同名合同；为防止覆盖原件，请重命名后重新上传")

            staged_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.upload{suffix}")
            permissions = self._load_contract_permissions()
            original_permissions = {name: list(roles) for name, roles in permissions.items()}
            file_installed = False
            permissions_saved = False
            try:
                if suffix in binary_suffixes:
                    if not file_bytes:
                        raise ValueError("PDF 或 Word 合同需要上传原始文件内容")
                    staged_path.write_bytes(file_bytes)
                else:
                    staged_path.write_text(content, encoding="utf-8")

                # Validate extractability before the file becomes visible to the
                # contract list. This also prevents a failed replacement from
                # deleting an earlier contract with the same business name.
                DocumentLoader().load(staged_path)
                staged_path.replace(path)
                file_installed = True

                permissions[safe_name] = normalized_roles
                self._save_contract_permissions(permissions)
                permissions_saved = True
                self._load_contracts()
                document = next(
                    (item for item in self.contract_documents if item.source_path == safe_name),
                    None,
                )
                if document is None or document.status != "ready":
                    error = document.error if document is not None else "未能读取已上传文件"
                    raise ValueError(f"合同解析失败：{error or '未能读取有效文本'}")
            except Exception as exc:
                staged_path.unlink(missing_ok=True)
                if file_installed:
                    path.unlink(missing_ok=True)
                if permissions_saved:
                    self._save_contract_permissions(original_permissions)
                self._load_contracts()
                if isinstance(exc, (PermissionError, ValueError)):
                    raise
                raise ValueError(f"合同上传失败：{exc}") from exc
            finally:
                staged_path.unlink(missing_ok=True)

            return {
                "document_id": document.document_id,
                "title": document.title,
                "knowledge_base_id": "contracts",
                "allowed_roles": normalized_roles,
                "status": document.status,
            }

    def list_contracts(self, role: str) -> list[dict[str, Any]]:
        role = self._validate_role(role)
        return [
            {
                "document_id": document.document_id,
                "title": document.title,
                "knowledge_base_id": "contracts",
                "status": document.status,
                "allowed_roles": document.allowed_roles,
                "document_version": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
            }
            for document in self.contract_documents
            if role == "admin" or role in document.allowed_roles
        ]

    def review_contract(self, document_id: str, role: str, query: str = "") -> dict[str, Any]:
        role = self._validate_role(role)
        document = next(
            (
                item
                for item in self.contract_documents
                if item.document_id == document_id and item.status == "ready"
                and (role == "admin" or role in item.allowed_roles)
            ),
            None,
        )
        if document is None:
            raise KeyError("合同不可用")
        chunks = [chunk for chunk in self.contract_chunks if chunk.document_id == document_id]
        if query and chunks:
            # Build from already-authorized document chunks, before query execution.
            authorized_retriever = HybridRetriever.from_chunks(
                chunks,
                LocalHashEmbeddings(),
                retrieval_top_k=self.settings.retrieval_top_k,
                rrf_k=self.settings.rrf_k,
                enable_reranker=False,
                prefer_faiss=False,
            )
            chunks = [result.chunk for result in authorized_retriever.retrieve(query, top_k=self.settings.answer_top_k)["final_results"]]
        elif chunks:
            # Full-document field extraction uses ``document.content`` below;
            # returning every chunk only bloats the API response and exposes
            # more authorized-but-unrequested contract text than necessary.
            chunks = chunks[: self.settings.answer_top_k]
        result = review_contract(document, chunks, query=query)
        result["related_policy_sources"] = self._related_policy_sources(query)
        return result

    def _related_policy_sources(self, query: str) -> list[dict[str, Any]]:
        """Retrieve public policy references separately from protected contract evidence."""
        if not query.strip() or self.retriever is None:
            return []
        retrieval = self.retriever.retrieve(query, top_k=self.settings.answer_top_k)
        if not has_sufficient_evidence(query, retrieval["final_results"]):
            return []
        return [
            {
                "title": item.chunk.title,
                "section": item.chunk.section,
                "source_path": item.chunk.source_path,
                "source_type": Path(item.chunk.source_path).suffix.lstrip(".") or "text",
                "rank": item.rank,
                "content_preview": item.chunk.content,
            }
            for item in retrieval["final_results"]
            if item.chunk.knowledge_base_id == "policy"
        ]

    def document_summaries(self) -> list[dict[str, Any]]:
        chunk_counts: dict[str, int] = {}
        for chunk in self.chunks:
            chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
        return [
            {
                "document_id": document.document_id,
                "title": document.title,
                "category": document.category,
                "source_path": document.source_path,
                "status": document.status,
                "error": document.error,
                "version": document.version,
                "effective_date": document.effective_date,
                "department": document.department,
                "owner": document.owner,
                "section_count": len(document.sections),
                "chunk_count": chunk_counts.get(document.document_id, 0),
            }
            for document in self.documents
        ]

    def status(self) -> dict[str, Any]:
        ready_documents = sum(document.status == "ready" for document in self.documents)
        runtime_errors = dict(self.errors)
        document_errors = {
            document.source_path: document.error
            for document in self.documents
            if document.status != "ready" and document.error
        }
        if document_errors:
            runtime_errors["documents"] = document_errors
        effective_llm_provider = self.settings.llm_provider
        if self.assistant and self.assistant.last_llm_error:
            runtime_errors["llm_runtime"] = self.assistant.last_llm_error
            effective_llm_provider = "unavailable"

        reranker_effective = False
        if self.retriever:
            reranker_effective = self.retriever.reranker.effective
            if self.retriever.reranker.error:
                runtime_errors["reranker_runtime"] = self.retriever.reranker.error

        if runtime_errors:
            health_status = "degraded"
        elif self.retriever is None:
            health_status = "starting"
        else:
            health_status = "ok"

        return {
            "status": health_status,
            "app_name": self.settings.app_name,
            "version": self.settings.app_version,
            "document_count": len(self.documents),
            "ready_document_count": ready_documents,
            "chunk_count": len(self.chunks),
            "evaluation_case_count": len(self.evaluation_cases),
            "embedding_provider": self.settings.embedding_provider,
            "effective_embedding_provider": self.effective_embedding_provider,
            "llm_provider": self.settings.llm_provider,
            "effective_llm_provider": effective_llm_provider,
            "vector_backend": self.retriever.vector_index.backend if self.retriever else "uninitialized",
            "reranker_enabled": self.retriever.reranker.enabled if self.retriever else False,
            "reranker_effective": reranker_effective,
            "built_at": self.built_at,
            "errors": runtime_errors,
        }

    def _corpus_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"{INDEX_SCHEMA_VERSION}:{self.settings.chunk_size}:{self.settings.chunk_overlap}:"
            f"{self.settings.embedding_provider}:{self.settings.embedding_model}".encode("utf-8")
        )
        for path in sorted(self.settings.raw_data_dir.glob("*")):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                # Large policy PDFs must not be copied into memory only to
                # calculate a cache key. Streaming also keeps startup stable
                # when tests or local services initialize close together.
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
        return digest.hexdigest()


@lru_cache(maxsize=1)
def get_runtime() -> RuntimeContainer:
    return RuntimeContainer()
