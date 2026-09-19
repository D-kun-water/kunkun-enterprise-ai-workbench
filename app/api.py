"""FastAPI schemas and routes."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core import get_runtime


router = APIRouter()


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=8)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    debug: dict[str, Any]


class DocumentsResponse(BaseModel):
    documents: list[dict[str, Any]]


class RebuildResponse(BaseModel):
    message: str
    status: dict[str, Any]


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=5000)
    rating: str
    reason: str = Field(default="", max_length=20)

    @field_validator("question", "answer")
    @classmethod
    def validate_feedback_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题和回答不能为空")
        return value

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: str) -> str:
        if value not in {"up", "down"}:
            raise ValueError("rating 必须是 up 或 down")
        return value


class FeedbackResponse(BaseModel):
    feedback_id: str
    created_at: str


class ContractUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=200)
    content: str = Field(default="", max_length=100000)
    content_base64: str = Field(default="", max_length=25000000)
    allowed_roles: list[Literal["admin", "legal", "business", "employee"]] = Field(min_length=1)
    actor_role: Literal["admin", "legal", "business", "employee"]


class ContractReviewRequest(BaseModel):
    role: Literal["admin", "legal", "business", "employee"]
    query: str = Field(default="", max_length=1000)


class ContractsResponse(BaseModel):
    contracts: list[dict[str, Any]]


@router.get("/")
def root() -> dict[str, str]:
    return {"name": "鲲坤科技企业 AI 协作工作台", "message": "企业 AI 协作服务 API"}


@router.get("/health")
def health() -> dict[str, Any]:
    return get_runtime().status()


@router.get("/knowledge-base/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    return DocumentsResponse(documents=get_runtime().document_summaries())


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        history = [message.model_dump() for message in request.history]
        return ChatResponse(**get_runtime().ask(request.question, history=history))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答处理失败：{exc}") from exc


@router.post("/evaluation/run")
def evaluation() -> dict[str, Any]:
    try:
        return get_runtime().evaluate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"评测执行失败：{exc}") from exc


@router.post("/knowledge-base/rebuild", response_model=RebuildResponse)
def rebuild() -> RebuildResponse:
    try:
        status = get_runtime().bootstrap(force_rebuild=True)
        return RebuildResponse(message="知识库索引已重建", status=status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"索引重建失败：{exc}") from exc


@router.get("/contracts", response_model=ContractsResponse)
def contracts(role: Literal["admin", "legal", "business", "employee"]) -> ContractsResponse:
    try:
        return ContractsResponse(contracts=get_runtime().list_contracts(role))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/contracts")
def upload_contract(request: ContractUploadRequest) -> dict[str, Any]:
    try:
        file_bytes = None
        if request.content_base64:
            try:
                file_bytes = base64.b64decode(request.content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("合同文件内容编码无效") from exc
        return get_runtime().upload_contract(
            request.filename,
            request.content,
            list(request.allowed_roles),
            request.actor_role,
            file_bytes=file_bytes,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/contracts/{document_id}/review")
def review_contract(document_id: str, request: ContractReviewRequest) -> dict[str, Any]:
    try:
        return get_runtime().review_contract(document_id, request.role, request.query)
    except KeyError as exc:
        # Missing and unauthorized contracts share one response to avoid existence leakage.
        raise HTTPException(status_code=404, detail="合同不可用") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        record = get_runtime().record_feedback(
            request.question, request.answer, request.rating, request.reason
        )
        return FeedbackResponse(**record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/feedback/summary")
def feedback_summary() -> dict[str, object]:
    return get_runtime().feedback_summary()
