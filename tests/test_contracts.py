import pytest

from app.contracts import extract_contract_fields, export_contract_review, review_contract, run_contract_checks
from app.ingestion import KnowledgeDocument, TextChunk


@pytest.mark.parametrize(("key", "text", "status"), [
    ("term", "合同期限：2026年1月1日至2026年12月31日。", "found"),
    ("parties", "甲方：甲建设有限公司。乙方：乙安装有限公司。", "found"),
    ("amount", "合同金额（含税）：109000元；合同金额（不含税）：100000元；税率9%。", "found"),
    ("performance_bond", "履约保函金额为合同价的5%，签订后7日内提交，竣工验收后30日内返还。", "found"),
    ("payment", "付款节点：第3条，具体付款事宜另行约定。", "incomplete"),
    ("banking", "开户银行：____；账号：____；纳税人识别号：____。", "incomplete"),
    ("parties", "甲方名称：____；乙方名称：____。", "incomplete"),
    ("amount", "合同金额（含税）：____元；合同金额（不含税）：____元；第9条另行约定。", "incomplete"),
])
def test_contract_common_wordings_and_empty_templates(key, text, status):
    assert extract_contract_fields(text)[key]["status"] == status


def test_long_clause_keeps_the_matched_anchor():
    text = "合同说明：" + "其他事项" * 160 + "付款节点：验收后30日内支付90%。"
    assert "30日内支付90%" in extract_contract_fields(text)["payment"]["evidence"]


def test_extract_contract_fields_keeps_clause_evidence() -> None:
    text = (
        "合同期限：2026年1月1日至2026年12月31日。\n"
        "付款节点：验收合格后30日内支付合同价款的90%。\n"
        "承包人：甲方建筑公司；分包人：乙方装饰公司；\n"
        "签约合同价（含增值税）：人民币 1090000 元；签约合同价（不含增值税）：1000000 元。\n"
        "开户银行：中国银行；账号：123456789；纳税人识别号：9111。\n"
        "履约保证金：合同价的10%，供货完毕后返还。\n"
        "终止和续约：合同到期前30日书面通知是否续约。"
    )

    fields = extract_contract_fields(text)

    assert "2026年12月31日" in fields["term"]["value"]
    assert "30日内支付" in fields["payment"]["evidence"]
    assert fields["parties"]["evidence"]
    assert fields["amount"]["evidence"]
    assert fields["banking"]["evidence"]
    assert fields["performance_bond"]["evidence"]
    assert fields["termination_renewal"]["evidence"]
    assert "付款触发条件" in fields["payment"]["review_focus"]


def test_extract_contract_fields_prefers_operational_clause_over_early_keyword_noise() -> None:
    text = (
        "履约保证金在供货完毕且办理完总结算签字确认后返还。"
        "六、付款信息及支付比例。款项的支付：按月度计量值付款70%，"
        "总结算办理完后12个月内分批次付至97%，余款3%作为质保金。"
        "八、合同生、失效日期：本合同自合同签订之日起生效，有效期至合同履约完毕债权两清。"
        "7.5 违约责任：乙方违反合同约定，应赔偿甲方因此遭受的损失。"
        "验收要求：符合国家现行标准；不合格产品应退场。"
        "发票种类及税率：增值税专用发票，税率13%。"
    )

    fields = extract_contract_fields(text)

    assert "按月度计量值付款70%" in fields["payment"]["evidence"]
    assert "履约保证金" not in fields["payment"]["evidence"]
    assert fields["payment"]["status"] == "found"
    assert "有效期至合同履约完毕债权两清" in fields["term"]["evidence"]
    assert fields["term"]["status"] == "incomplete"
    assert "具体签订日期" in fields["term"]["issue"]
    assert fields["termination_renewal"]["status"] == "missing"


def test_extract_contract_fields_uses_project_schedule_dates_for_term_review() -> None:
    text = (
        "四、工期、质量、安全要求。计划开工日期：2024年9月20日；"
        "计划完工日期：2026年11月18日；计划竣工日期：2026年11月18日；合同工期总日历天数为790天。"
    )

    field = extract_contract_fields(text)["term"]

    assert field["status"] == "found"
    assert "2024年9月20日" in field["evidence"]
    assert "2026年11月18日" in field["evidence"]


def test_missing_and_vague_clauses_become_review_prompts_without_legal_conclusion() -> None:
    text = "合同期限：2026年1月1日至2026年12月31日。付款事宜另行约定。"
    fields = extract_contract_fields(text)

    prompts = run_contract_checks(text, fields)
    joined = " ".join(item["message"] for item in prompts)

    assert any(item["field"] == "insurance" for item in prompts)
    assert any(item["field"] == "payment" for item in prompts)
    assert "待确认" in joined
    assert "另行约定" in joined
    assert "合法" not in joined
    assert "违法" not in joined


def test_review_contract_returns_clause_locations_and_auxiliary_language() -> None:
    document = KnowledgeDocument(
        "contract-1",
        "示例采购合同",
        "合同",
        "contract-1.md",
        "付款节点：验收后30日内付款。\n承包人名称：甲方公司；分包人名称：乙方公司。",
        ["付款", "交付与验收"],
        knowledge_base_id="contracts",
        allowed_roles=["legal"],
    )
    chunks = [
        TextChunk(
            "contract-1-000",
            "contract-1",
            document.title,
            document.category,
            "付款",
            document.source_path,
            document.content,
            0,
            knowledge_base_id="contracts",
            allowed_roles=["legal"],
        )
    ]

    result = review_contract(document, chunks)

    assert result["knowledge_base_id"] == "contracts"
    assert result["clauses"]
    assert result["sources"][0]["content_preview"] == document.content
    assert "已定位" in result["summary"]
    assert result["review_counts"] == {"checked": 12, "located": 2, "incomplete": 0, "pending": 10}
    assert "公司名称" in result["fields"][2]["review_focus"]
    assert "法律结论" not in result["summary"]


def test_commercial_review_uses_sections_and_exports_human_decisions():
    content = (
        "# 采购合同\n\n## 变更及质保\n工程变更须在7日内提交书面通知。\n"
        "质保金扣留3%，验收满12个月后30日内返还。\n\n"
        "## 交付及责任\n验收标准：按技术附件核对并形成验收记录。\n"
        "违约责任：逾期按未交付金额每日0.05%支付违约金。\n"
    )
    document = KnowledgeDocument("c-1", "采购合同", "合同", "合同.md", content, [])
    result = review_contract(document, [])
    fields = {item["field"]: item for item in result["fields"]}
    for key in ("variation_claims", "retention_warranty", "acceptance", "liability"):
        assert fields[key]["status"] == "found"
        assert fields[key]["evidence_sections"]
    report = export_contract_review(result, {"variation_claims": {"status": "需补充", "note": "核实接收人"}})
    assert "人工复核：需补充" in report
    assert "核实接收人" in report
    assert result["document_version"] in report
    assert "原文位置：变更及质保" in report


def test_export_defaults_to_unreviewed_and_rejects_unknown_status():
    document = KnowledgeDocument("c", "合同", "合同", "c.md", "付款：30日内付款。", [])
    result = review_contract(document, [])
    assert "人工复核：待复核" in export_contract_review(result)
    with pytest.raises(ValueError):
        export_contract_review(result, {"payment": {"status": "自动通过"}})


def test_named_party_rows_outrank_later_mentions():
    text = "甲方：甲方工程有限公司。\n乙方：乙方设备有限公司。\n工程变更须在7日内向甲方提交书面通知及工作量记录。"
    field = extract_contract_fields(text)["parties"]
    assert field["status"] == "found"
    assert "甲方工程有限公司" in field["evidence"]
    assert "乙方设备有限公司" in field["evidence"]
    assert "工程变更" not in field["evidence"]
