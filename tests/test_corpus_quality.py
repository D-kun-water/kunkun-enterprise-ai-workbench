import re
from pathlib import Path

from app.ingestion import DocumentLoader


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_employee_facing_policy_corpus_has_actionable_rules() -> None:
    documents = DocumentLoader().load_directory(RAW_DIR)
    by_source = {document.source_path: document for document in documents}

    files = sorted(path for path in RAW_DIR.iterdir() if path.is_file())
    assert len(files) == 3
    assert all(path.suffix.lower() == ".pdf" for path in files)

    anchors = {
        "鲲坤科技-员工管理制度汇编-2025融合版.pdf": ("新员工入职办理流程", "探亲假", "晋升与调薪评审"),
        "鲲坤科技-合同管理制度-2025版.pdf": ("合同谈判与起草", "合同审核与审批"),
        "鲲坤科技-供应商管理制度-2025版.pdf": ("供应商管理制度", "采购活动"),
    }
    expected_titles = {
        "鲲坤科技-员工管理制度汇编-2025融合版.pdf": "鲲坤科技员工管理制度汇编",
        "鲲坤科技-合同管理制度-2025版.pdf": "鲲坤科技合同管理制度",
        "鲲坤科技-供应商管理制度-2025版.pdf": "鲲坤科技供应商管理制度",
    }

    assert set(anchors) == set(by_source)
    for source_path, required_terms in anchors.items():
        document = by_source[source_path]
        assert document.status == "ready", source_path
        assert document.title == expected_titles[source_path], source_path
        assert "鲲坤科技" in document.content, f"{source_path} 缺少当前样例品牌"
        assert "星桥科技" not in document.content, f"{source_path} 仍包含旧样例品牌"
        assert "\x00" not in document.content, f"{source_path} 包含 NUL 字符"
        assert '“\n”' not in document.content, f"{source_path} 包含品牌替换残留占位"
        compact_content = re.sub(r"\s+", "", document.content)
        for term in required_terms:
            assert term.replace(" ", "") in compact_content, f"{source_path} 缺少员工办理规则：{term}"
