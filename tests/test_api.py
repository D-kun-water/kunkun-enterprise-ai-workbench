import base64
import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def isolate_api_runtime(monkeypatch, deterministic_runtime):
    monkeypatch.setattr("app.api.get_runtime", lambda: deterministic_runtime)
    monkeypatch.setattr("app.main.get_runtime", lambda: deterministic_runtime)
    # The fixture already bootstrapped the isolated runtime and installed its
    # deterministic answer composer. Lifespan must not replace it with a live LLM.
    monkeypatch.setattr(deterministic_runtime, "bootstrap", lambda **kwargs: deterministic_runtime.status())


def test_health_exposes_index_and_model_status() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_count"] == 3
    assert payload["chunk_count"] > 0
    assert "embedding_provider" in payload


def test_root_uses_chinese_product_name() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "鲲坤科技企业 AI 协作工作台"


def test_documents_chat_and_evaluation_routes() -> None:
    with TestClient(app) as client:
        documents = client.get("/knowledge-base/documents")
        chat = client.post("/chat", json={"question": "事假怎么申请？"})
        evaluation = client.post("/evaluation/run")

    assert documents.status_code == 200
    assert len(documents.json()["documents"]) == 3
    assert chat.status_code == 200
    assert chat.json()["sources"]
    assert evaluation.status_code == 200
    assert evaluation.json()["total_cases"] >= 15
    assert "hit_at_3" in evaluation.json()
    assert "refusal_accuracy" in evaluation.json()


def test_empty_chat_question_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_schema_does_not_advertise_unused_thread_memory() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    properties = schema["components"]["schemas"]["ChatRequest"]["properties"]
    assert "thread_id" not in properties
    assert "history" in properties


def test_chat_route_forwards_conversation_history(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RuntimeStub:
        def ask(self, question: str, history=None):
            captured["question"] = question
            captured["history"] = history
            return {"answer": "已记住上下文", "sources": [], "debug": {}}

    monkeypatch.setattr("app.api.get_runtime", lambda: RuntimeStub())
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "那怎么申请？",
                "history": [{"role": "user", "content": "新员工有几天年假？"}],
            },
        )

    assert response.status_code == 200
    assert captured == {
        "question": "那怎么申请？",
        "history": [{"role": "user", "content": "新员工有几天年假？"}],
    }


def test_feedback_routes_record_and_summarize() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/feedback",
            json={"question": "事假怎么申请？", "answer": "请提交事假申请。", "rating": "up"},
        )
        summary = client.get("/feedback/summary")

    assert created.status_code == 200
    assert created.json()["feedback_id"]
    assert summary.status_code == 200
    assert summary.json()["up"] >= 1


def test_feedback_rejects_unknown_rating() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json={"question": "问题", "answer": "回答", "rating": "maybe"},
        )

    assert response.status_code == 422


def test_feedback_rejects_blank_question_and_answer() -> None:
    with TestClient(app) as client:
        blank_question = client.post(
            "/feedback",
            json={"question": "   ", "answer": "回答", "rating": "up"},
        )
        blank_answer = client.post(
            "/feedback",
            json={"question": "问题", "answer": "\n\t", "rating": "up"},
        )

    assert blank_question.status_code == 422
    assert blank_answer.status_code == 422


def test_contract_routes_expose_only_authorized_review_data(monkeypatch) -> None:
    class RuntimeStub:
        def list_contracts(self, role: str):
            return [{"document_id": "c-1", "title": "采购合同", "knowledge_base_id": "contracts"}]

        def review_contract(self, document_id: str, role: str, query: str = ""):
            if role == "employee":
                raise KeyError("合同不可用")
            return {
                "document_id": document_id,
                "knowledge_base_id": "contracts",
                "summary": "审阅提示：请确认付款节点。",
                "fields": [],
                "risk_prompts": [],
                "clauses": [],
                "sources": [{"content_preview": "付款100000元"}],
            }

    monkeypatch.setattr("app.api.get_runtime", lambda: RuntimeStub())
    with TestClient(app) as client:
        visible = client.get("/contracts?role=legal")
        hidden = client.post("/contracts/c-1/review", json={"role": "employee"})
        allowed = client.post("/contracts/c-1/review", json={"role": "legal"})

    assert visible.status_code == 200
    assert visible.json()["contracts"][0]["title"] == "采购合同"
    assert hidden.status_code == 404
    assert "采购合同" not in hidden.text
    assert "100000" not in hidden.text
    assert allowed.status_code == 200
    assert allowed.json()["sources"][0]["content_preview"] == "付款100000元"


def test_contract_upload_maps_actor_permission_to_forbidden(monkeypatch) -> None:
    class RuntimeStub:
        def upload_contract(self, filename, content, allowed_roles, actor_role, *, file_bytes=None):
            raise PermissionError("只有 admin 或 legal 可以上传合同")

    monkeypatch.setattr("app.api.get_runtime", lambda: RuntimeStub())
    with TestClient(app) as client:
        response = client.post(
            "/contracts",
            json={
                "filename": "采购合同.md",
                "content": "# 采购合同\n付款节点：验收后付款。",
                "allowed_roles": ["legal"],
                "actor_role": "employee",
            },
        )

    assert response.status_code == 403


def test_contract_upload_decodes_binary_content(monkeypatch) -> None:
    captured = {}

    class RuntimeStub:
        def upload_contract(self, filename, content, allowed_roles, actor_role, *, file_bytes=None):
            captured.update(
                {
                    "filename": filename,
                    "content": content,
                    "allowed_roles": allowed_roles,
                    "actor_role": actor_role,
                    "file_bytes": file_bytes,
                }
            )
            return {
                "document_id": "contract-pdf",
                "title": "合同PDF",
                "knowledge_base_id": "contracts",
                "allowed_roles": allowed_roles,
                "status": "ready",
            }

    monkeypatch.setattr("app.api.get_runtime", lambda: RuntimeStub())
    with TestClient(app) as client:
        response = client.post(
            "/contracts",
            json={
                "filename": "合同.pdf",
                "content_base64": base64.b64encode(b"%PDF-1.7 fake").decode("ascii"),
                "allowed_roles": ["legal", "business"],
                "actor_role": "legal",
            },
        )

    assert response.status_code == 200
    assert captured["filename"] == "合同.pdf"
    assert captured["content"] == ""
    assert captured["allowed_roles"] == ["legal", "business"]
    assert captured["actor_role"] == "legal"
    assert captured["file_bytes"] == b"%PDF-1.7 fake"
