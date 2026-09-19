from streamlit.testing.v1 import AppTest

from app.contracts import review_contract
from app.ingestion import KnowledgeDocument
from frontend import streamlit_app


def _contract_page():
    from frontend.streamlit_app import render_contract_review_tab
    render_contract_review_tab("http://test.invalid", "legal")


def test_review_survives_note_edit_and_exports_versioned_record(monkeypatch):
    document = KnowledgeDocument("a#1", "合同", "合同", "a#1.md", "## 付款\n付款节点：验收后30日内付款。", [])
    result = review_contract(document, [])
    monkeypatch.setattr(streamlit_app, "fetch_contracts", lambda *args: [{
        "document_id": document.document_id, "title": document.title,
        "document_version": result["document_version"],
    }])
    requests = []

    def review_request(method, base, path, **kwargs):
        requests.append(path)
        return result

    monkeypatch.setattr(streamlit_app, "api_request", review_request)
    app = AppTest.from_function(_contract_page).run()
    assert not app.exception
    next(button for button in app.button if button.label == "开始审阅").click().run()
    assert not app.exception
    assert requests == ["/contracts/a%231/review"]
    app.text_area[0].input("待核实期限").run()
    assert not app.exception
    assert app.text_area[0].value == "待核实期限"
    assert len(app.get("download_button")) == 1
    assert len(requests) == 1


def test_identity_change_clears_contract_result_and_notes():
    streamlit_app.st.session_state.clear()
    streamlit_app.st.session_state.update({
        "active_role": "employee", "_assistant_role": "legal",
        "contract-review-result": {"result": "private"},
        "contract-decision-note": "private note",
    })
    streamlit_app._handle_role_change()
    assert not any(str(key).startswith("contract-") for key in streamlit_app.st.session_state)


def test_identical_contract_text_does_not_share_manual_decisions(monkeypatch):
    documents = [KnowledgeDocument(name, name, "合同", name + ".md",
                                   "## 付款\n付款节点：验收后30日内付款。", []) for name in ("first", "second")]
    reviews = {doc.document_id: review_contract(doc, []) for doc in documents}
    assert reviews["first"]["document_version"] == reviews["second"]["document_version"]
    monkeypatch.setattr(streamlit_app, "fetch_contracts", lambda *args: [
        {"document_id": doc.document_id, "title": doc.title,
         "document_version": reviews[doc.document_id]["document_version"]} for doc in documents
    ])
    monkeypatch.setattr(streamlit_app, "api_request", lambda method, base, path, **kwargs: reviews[path.split("/")[2]])
    app = AppTest.from_function(_contract_page).run()
    next(button for button in app.button if button.label == "开始审阅").click().run()
    app.text_area[0].input("仅属于第一份合同的复核意见").run()
    app.selectbox(key="contract-document").select("second").run()
    next(button for button in app.button if button.label == "开始审阅").click().run()
    assert not app.exception
    assert all(field.value == "" for field in app.text_area)


def test_contract_page_uses_real_api_upload_review_and_export(tmp_path, monkeypatch):
    from pathlib import Path
    from urllib.parse import urlsplit
    from fastapi.testclient import TestClient
    from app.main import app as api
    from app.core import RuntimeContainer, Settings
    from app.contracts import export_contract_review

    runtime = RuntimeContainer(Settings(
        _env_file=None, raw_data_dir=tmp_path / "raw", evaluation_dir=tmp_path / "evaluation",
        processed_dir=tmp_path / "processed", faiss_dir=tmp_path / "faiss",
        contract_data_dir=tmp_path / "contracts", embedding_provider="local",
    ))
    monkeypatch.setattr("app.api.get_runtime", lambda: runtime)
    monkeypatch.setattr("app.main.get_runtime", lambda: runtime)
    reports = []

    def capture_export(result, decisions):
        report = export_contract_review(result, decisions)
        reports.append(report)
        return report

    monkeypatch.setattr(streamlit_app, "export_contract_review", capture_export)
    with TestClient(api) as client:
        sample = Path(__file__).resolve().parent.parent / "data/samples/工程采购合同_验收样本.md"
        uploaded = client.post("/contracts", json={
            "filename": "验收#1.md", "content": sample.read_text(encoding="utf-8"),
            "actor_role": "legal", "allowed_roles": ["legal"],
        })
        assert uploaded.status_code == 200, uploaded.text
        assert client.get("/contracts", params={"role": "employee"}).json()["contracts"] == []

        def transport(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, urlsplit(url).path, **kwargs)

        monkeypatch.setattr(streamlit_app.requests, "request", transport)
        app = AppTest.from_function(_contract_page).run()
        next(button for button in app.button if button.label == "开始审阅").click().run()
        assert not app.exception
        assert len(app.text_area) == 12
        app.text_area[0].input("由业务负责人核对工期").run()
        assert not app.exception
        assert "由业务负责人核对工期" in reports[-1]
        assert "人工复核：待复核" in reports[-1]
        assert len(app.get("download_button")) == 1
