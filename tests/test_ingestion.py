import zipfile
from pathlib import Path

from app.ingestion import is_catalog_text


def test_pdf_catalog_page_is_not_indexed_but_body_page_is_kept():
    from app.ingestion import TextChunker

    catalog = "目录\n第一章 总则\n3\n第二章 合同谈判与起草\n4\n第三章 合同履行\n8"
    content = f"## 第 2 页\n{catalog}\n\n## 第 4 页\n签署合同前应调查相对方的主体资格和资信状况。"
    blocks = TextChunker._section_blocks(content)
    assert [section for section, body in blocks] == ["第 4 页"]
    assert is_catalog_text(catalog)
    assert not is_catalog_text(catalog + "\n正文规定：签署合同前应调查相对方的主体资格。")

from app.ingestion import (
    DocumentLoader,
    IngestionStore,
    KnowledgeDocument,
    TextChunk,
    TextChunker,
)


def _write_docx(path: Path, paragraphs: list[tuple[str, str]]) -> None:
    body = "".join(
        "<w:p><w:pPr><w:pStyle w:val=\"%s\"/></w:pPr><w:r><w:t>%s</w:t></w:r></w:p>"
        % (style, text)
        for style, text in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


def test_docx_loader_extracts_title_and_heading_sections(tmp_path: Path) -> None:
    path = tmp_path / "attendance_policy.docx"
    _write_docx(
        path,
        [
            ("Title", "考勤与异常修正制度"),
            ("Heading1", "一、适用范围"),
            ("Normal", "本制度适用于公司全体正式员工、试用期员工及经批准的外包驻场人员。"),
            ("Heading1", "二、补卡申请"),
            ("Normal", "员工应在异常发生后三个工作日内提交补卡申请，并附准确的日期、原因和证明材料。"),
        ],
    )

    document = DocumentLoader().load(path)

    assert document.title == "考勤与异常修正制度"
    assert document.category == "人力资源"
    assert document.sections == ["一、适用范围", "二、补卡申请"]
    assert "三个工作日内提交补卡申请" in document.content


def test_pdf_loader_prefers_document_name_metadata(tmp_path: Path) -> None:
    import fitz

    path = tmp_path / "vpn_policy.pdf"
    pdf = fitz.open()
    pdf.set_metadata({"title": "远程办公与 VPN 使用管理制度"})
    page = pdf.new_page()
    page.insert_text((72, 72), "VPN access requires MFA.")
    pdf.save(path)
    pdf.close()

    document = DocumentLoader().load(path)

    assert document.title == "远程办公与 VPN 使用管理制度"


def test_markdown_loader_extracts_title_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        "# 考勤制度\n\n- 知识分类：人力资源\n\n## 补卡\n员工应及时补卡。",
        encoding="utf-8",
    )

    document = DocumentLoader().load(path)

    assert document.title == "考勤制度"
    assert document.category == "人力资源"
    assert document.sections == ["补卡"]
    assert "员工应及时补卡" in document.content


def test_txt_loader_uses_filename_as_title(tmp_path: Path) -> None:
    path = tmp_path / "访客须知.txt"
    path.write_text("访客进入办公区前需要登记。", encoding="utf-8")

    document = DocumentLoader().load(path)

    assert document.title == "访客须知"
    assert document.document_id == "访客须知"


def test_markdown_loader_prefers_document_name_metadata(tmp_path: Path) -> None:
    path = tmp_path / "vpn.md"
    path.write_text(
        "# 【样例制度】不代表真实企业\n\n- 文档名称：远程办公与 VPN 使用制度\n\n## 规则\n必须启用 MFA。",
        encoding="utf-8",
    )

    document = DocumentLoader().load(path)

    assert document.title == "远程办公与 VPN 使用制度"


def test_loader_extracts_policy_metadata(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        "# 考勤制度\n\n- 版本：V2.1\n- 生效日期：2026-01-15\n- 发布部门：人力资源部\n- 负责人：张三\n\n## 规则\n员工应及时补卡。",
        encoding="utf-8",
    )

    document = DocumentLoader().load(path)

    assert document.version == "V2.1"
    assert document.effective_date == "2026-01-15"
    assert document.department == "人力资源部"
    assert document.owner == "张三"


def test_loader_does_not_parse_department_owner_sentence_as_owner(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        "# 试用期制度\n\n"
        "## 文档信息\n"
        "- 发布部门：人力资源部\n"
        "- 部门负责人：确认岗位资源与评价结论。\n\n"
        "## 规则\n员工应及时完成试用期目标确认。",
        encoding="utf-8",
    )

    document = DocumentLoader().load(path)

    assert document.owner == ""


def test_chunker_preserves_overlap() -> None:
    chunks = TextChunker(chunk_size=20, chunk_overlap=5).split_text(
        "制度内容" * 12,
        document_id="policy",
        title="制度",
    )

    assert len(chunks) > 1
    assert chunks[0].content[-5:] == chunks[1].content[:5]


def test_chunker_keeps_heading_metadata() -> None:
    document = KnowledgeDocument(
        document_id="leave",
        title="休假制度",
        category="人力资源",
        source_path="leave.md",
        content="# 休假制度\n\n## 年假申请\n员工需提前提交申请。",
        sections=["年假申请"],
    )

    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_document(document)

    assert chunks[0].section == "年假申请"
    assert chunks[0].document_id == "leave"


def test_chunker_excludes_document_catalog_section() -> None:
    document = KnowledgeDocument(
        document_id="leave",
        title="休假制度",
        category="人力资源",
        source_path="leave.md",
        content=(
            "# 样例声明\n\n## 文档信息\n- 文档名称：休假制度\n\n"
            "## 年假申请\n员工需提前提交申请。"
        ),
        sections=["文档信息", "年假申请"],
    )

    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_document(document)

    assert [chunk.section for chunk in chunks] == ["年假申请"]


def test_ingestion_store_round_trip(tmp_path: Path) -> None:
    document = KnowledgeDocument("vpn", "VPN 指南", "IT", "vpn.md", "连接说明", ["连接"])
    chunk = TextChunk("vpn-0", "vpn", "VPN 指南", "IT", "连接", "vpn.md", "VPN-403", 0)
    store = IngestionStore(tmp_path)

    store.save([document], [chunk], {"embedding_provider": "local"})
    documents, chunks, metadata = store.load()

    assert documents == [document]
    assert chunks == [chunk]
    assert metadata["embedding_provider"] == "local"
    assert not (tmp_path / "bm25_corpus.json").exists()


def test_ingestion_store_loads_legacy_documents_without_policy_metadata(tmp_path: Path) -> None:
    (tmp_path / "documents.json").write_text(
        '[{"document_id":"legacy","title":"旧制度","category":"HR","source_path":"legacy.md","content":"内容","sections":[],"status":"ready","error":""}]',
        encoding="utf-8",
    )
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    (tmp_path / "index_meta.json").write_text("{}", encoding="utf-8")

    documents, _, _ = IngestionStore(tmp_path).load()

    assert documents[0].version == ""


def test_loader_disambiguates_documents_with_same_stem(tmp_path: Path) -> None:
    (tmp_path / "policy.md").write_text("# Markdown\n\n## 规则\nMarkdown 内容。", encoding="utf-8")
    (tmp_path / "policy.txt").write_text("TXT 内容。", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert len({document.document_id for document in documents}) == 2
    chunks = TextChunker(100, 10).split_documents(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
