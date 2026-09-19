"""Regression checks for the project-level policy Q&A smoke test."""

from __future__ import annotations

import pytest

from app.core import RuntimeContainer


@pytest.mark.parametrize(
    ("question", "expected_source", "expected_keywords"),
    [
        (
            "新员工报到前，谁负责发起账号开通申请，试用期目标由谁准备？",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            ("人力资源部", "账号开通申请", "用人部门", "试用期目标"),
        ),
        (
            "试用期届满转正申请最晚应提前几个工作日，审核涉及哪些部门？",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            ("届满前10个工作日", "部门负责人", "人力资源部"),
        ),
        (
            "事假要提前多久申请，全年原则上累计最多多少工作日？",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            ("提前1个工作日", "10个工作日"),
        ),
        (
            "员工级在上海出差，住宿每天最高能报销多少，报销时限是什么？",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            ("450元", "10个工作日"),
        ),
        (
            "差旅报销缺少行程单会怎样？发票抬头需要如何填写？",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            ("缺少任一关键材料", "财务部门可退回补充", "公司全称", "统一社会信用代码"),
        ),
        (
            "合同草案形成前，承办部门需要调查相对方哪些方面？",
            "鲲坤科技-合同管理制度-2025版.pdf",
            ("主体资格", "资信状况", "履行能力"),
        ),
        (
            "准备起草一份采购合同时，我们要先调查供应商哪几方面？",
            "鲲坤科技-合同管理制度-2025版.pdf",
            ("主体资格", "资信状况", "履行能力", "经营状况", "市场需求", "服务质量", "价格"),
        ),
        (
            "合同签署后对方不履行，承办人应采取什么处理步骤？",
            "鲲坤科技-合同管理制度-2025版.pdf",
            ("详细说明", "法律顾问", "书面方式", "提出异议"),
        ),
        (
            "驳船供应商进入供应商库，最低注册资金以及必备许可、保险是什么？",
            "鲲坤科技-供应商管理制度-2025版.pdf",
            ("200万元", "水路运输许可证", "船舶险", "承运人责任险"),
        ),
        (
            "新增驳船供应商申请时，应说明什么并同步提交哪些材料？",
            "鲲坤科技-供应商管理制度-2025版.pdf",
            (
                "采购需求、预算",
                "营业执照副本",
                "水路运输许可证",
                "船舶配置与服务航线",
                "船舶保单",
            ),
        ),
    ],
)
def test_policy_answers_cover_every_requested_fact(
    deterministic_runtime: RuntimeContainer,
    question: str,
    expected_source: str,
    expected_keywords: tuple[str, ...],
) -> None:
    runtime = deterministic_runtime
    response = runtime.ask(question)

    assert response["debug"]["evidence_sufficient"] is True
    normalized_answer = response["answer"].replace(" ", "")
    assert all(keyword.replace(" ", "") in normalized_answer for keyword in expected_keywords), response["answer"]
    assert response["sources"]
    assert {source["source_path"] for source in response["sources"]} == {expected_source}


def test_unknown_request_is_refused_without_sources(
    deterministic_runtime: RuntimeContainer,
) -> None:
    runtime = deterministic_runtime
    response = runtime.ask("办公室咖啡机维修如何申请？")

    assert "依据不足" in response["answer"]
    assert response["sources"] == []
    assert response["debug"]["evidence_sufficient"] is False
