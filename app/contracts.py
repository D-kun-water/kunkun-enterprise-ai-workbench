"""Deterministic contract clause extraction and review prompts."""

from __future__ import annotations

import re
import hashlib
from typing import Any

from app.ingestion import KnowledgeDocument, TextChunk


CONTRACT_FIELDS = (
    (
        "term",
        "合同期限/工期/到期日",
        ("合同期限", "有效期", "到期日", "履行期限"),
        "计划开工、完工/竣工日期，合同工期，以及到期后的处理和自动续期",
        (
            "计划开工日期",
            "计划完工日期",
            "计划竣工日期",
            "合同工期",
            "合同生、失效日期",
            "合同生效",
            "有效期至",
            "合同期限",
            "履行期限",
        ),
        ("生效", "有效期", "期限", "到期"),
    ),
    (
        "payment",
        "付款与结算",
        ("付款", "支付", "结算"),
        "付款触发条件、比例或金额、付款期限及结算凭证",
        (
            "工程款支付比例",
            "工程进度款支付",
            "工程款支付周期",
            "款项的支付",
            "支付比例",
            "付款信息及支付比例",
            "付款节点",
            "付款",
        ),
        ("支付", "付款", "结算"),
    ),
    (
        "parties",
        "甲乙方公司信息",
        ("承包人", "分包人", "甲方", "乙方"),
        "甲方/乙方或承包人/分包人的公司名称、统一社会信用代码及主体身份",
        ("承包人", "分包人", "采购方", "供应方", "甲方", "乙方"),
        ("名称", "统一社会信用代码", "承包人", "分包人", "甲方", "乙方"),
    ),
    (
        "amount",
        "合同金额（含税/不含税）",
        ("签约合同价", "合同价款", "合同金额", "含增值税", "不含增值税"),
        "含税金额、不含税金额、税额和税率是否写明且相互对应",
        ("签约合同价（含增值税）", "签约合同价（不含增值税）", "含增值税", "不含增值税", "含税", "不含税", "合同金额", "合同价款"),
        ("含税", "不含税", "增值税", "元", "万元"),
    ),
    (
        "banking",
        "银行账户及纳税信息",
        ("开户银行", "开户行", "账号", "帐号", "纳税人识别号"),
        "收款账户、开户行、账户名称、纳税人识别号、地址和电话",
        ("开户银行", "开户行", "账号", "帐号", "纳税人识别号", "纳税人身份"),
        ("开户", "账号", "帐号", "识别号", "地址", "电话"),
    ),
    (
        "insurance",
        "保险",
        ("保险", "保险费", "保险责任", "投保"),
        "是否投保、保险范围、保险期限、保险费承担方及相关凭证",
        ("保险责任", "保险费", "投保", "保险", "保险范围"),
        ("保险", "投保", "保单", "保险费"),
    ),
    (
        "performance_bond",
        "履约保证金",
        ("履约保证金", "履约保函", "保证金"),
        "保证金/保函金额、提交时间、扣除条件和返还时间",
        ("履约保证金", "履约保函", "保证金"),
        ("保证金", "保函", "扣除", "返还", "提交"),
    ),
    (
        "termination_renewal",
        "终止和续约",
        ("终止", "解除", "续约", "续签"),
        "解除条件、通知期限、交接安排和续约规则",
        ("终止", "解除", "续约", "续签"),
        ("终止", "解除", "续约", "续签"),
    ),
    (
        "variation_claims", "变更、签证与索赔",
        ("变更", "签证", "索赔"),
        "提出时限、通知对象、书面确认、计价方式及证明材料",
        ("变更签证", "工程变更", "现场签证", "索赔通知", "索赔期限"),
        ("确认", "审批", "通知", "提交"),
    ),
    (
        "retention_warranty", "质保金与保修",
        ("质保金", "质量保证金", "保修期"),
        "扣留比例、返还触发条件、返还期限及保修责任",
        ("质保金", "质量保证金", "保修期"),
        ("返还", "扣留", "保修", "退还"),
    ),
    (
        "acceptance", "交付与验收",
        ("验收", "交付"),
        "交付材料、验收标准、负责人员、时限及不合格处理",
        ("验收标准", "验收程序", "验收期限", "验收要求", "交付与验收"),
        ("标准", "合格", "验收记录", "日内", "天内"),
    ),
    (
        "liability", "违约与赔偿",
        ("违约", "赔偿"),
        "责任触发条件、计算基数、比例或上限及双方责任是否对应",
        ("违约责任", "违约金", "逾期赔偿", "赔偿责任"),
        ("赔偿", "损失", "承担", "支付"),
    ),
)

_VAGUE_TERMS = ("及时", "尽快", "另行约定", "适时", "合理期限", "按需")


def _compact_text(text: str) -> str:
    """Preserve clause punctuation while repairing PDF line wrapping."""
    text = re.sub(r"(?m)^#{1,6}[^\n]*(?:\n|$)", "\n", text)
    return re.sub(r"[^\S\n]+", " ", text).strip()


def _excerpt_around(text: str, start_at: int, *, limit: int = 420) -> str:
    """Return a readable clause-sized excerpt instead of a single hit sentence."""
    left_boundary = max(text.rfind(mark, 0, start_at) for mark in ("。", "；", "\n"))
    start = max(0, left_boundary + 1, start_at - 96)
    hard_end = min(len(text), start + limit)
    sentence_end = min(
        (position for mark in ("。", "；") if (position := text.find(mark, start_at)) >= 0),
        default=-1,
    )
    end = sentence_end + 1 if start_at < sentence_end <= hard_end else hard_end
    excerpt = text[start:end].strip()
    return excerpt if end == len(text) or excerpt.endswith(("。", "；")) else f"{excerpt}…"


def _field_evidence(text: str, anchors: tuple[str, ...], *, max_items: int = 1) -> str:
    """Choose the strongest clause contexts, not the first keyword occurrence.

    Contract PDFs commonly mention words such as “付款” and “验收” in a guarantee,
    price table or header long before the operative clause.  Prefer specific anchors
    and retain the strongest operative context rather than a broad page-level hit.
    """
    compact = _compact_text(text)
    candidates: list[tuple[int, int, str]] = []
    for priority, anchor in enumerate(anchors):
        offset = compact.find(anchor)
        while offset >= 0:
            excerpt = _excerpt_around(compact, offset)
            # Anchor specificity must dominate excerpt length.  Otherwise a broad
            # page heading can outrank an exact operative clause simply because
            # PDF extraction made the heading's surrounding text longer.
            score = 10_000 - priority * 1_000 + min(len(excerpt), 420)
            if "......" in excerpt or "……" in excerpt:
                score -= 2_000
            if any(term in excerpt for term in ("%", "元", "付款", "支付比例", "支付周期")):
                score += 300
            if any(term in excerpt for term in ("名称", "统一社会信用代码", "纳税人识别号", "开户银行", "开户行", "账号", "帐号")):
                score += 450
            candidates.append((score, offset, excerpt))
            offset = compact.find(anchor, offset + len(anchor))

    selected: list[str] = []
    for _score, _offset, excerpt in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if not any(excerpt in existing or existing in excerpt for existing in selected):
            selected.append(excerpt)
        if len(selected) == max_items:
            break
    return "\n\n".join(selected)


def _field_status(key: str, evidence: str, required_terms: tuple[str, ...]) -> str:
    if not evidence:
        return "missing"
    normalized = re.sub(r"\s+", "", evidence)
    if key == "term":
        has_calendar_date = bool(
            re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[./-]\d{1,2}[./-]\d{1,2}", evidence)
        )
        has_project_schedule = any(term in evidence for term in ("开工", "完工", "竣工", "工期"))
        has_effective_term = any(term in evidence for term in ("有效期", "期限", "到期"))
        complete = has_calendar_date and (has_project_schedule or has_effective_term)
    elif key == "payment":
        # Clause numbers are not amounts, deadlines or payment percentages.
        complete = any(term in evidence for term in required_terms) and bool(
            re.search(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百]+)\s*(?:%|％|万元|元|个?月|日|天)", evidence)
        )
    elif key == "parties":
        def named_party(role: str) -> bool:
            match = re.search(rf"{role}(?:名称|公司名称)?[：:]([\u4e00-\u9fffA-Za-z0-9（）()]+)", normalized)
            return bool(match and len(match.group(1)) >= 2 and match.group(1) not in {"待填", "待填写", "待定"})

        complete = any(named_party(left) and named_party(right) for left, right in (
            ("甲方", "乙方"), ("承包人", "分包人"), ("采购方", "供应方"),
        ))
    elif key == "amount":
        value = r"[）)]?[：:为]?(?:人民币)?[￥¥]?\d[\d,，]*(?:\.\d+)?(?:万元|元)"
        complete = bool(re.search(r"(?<!不)含(?:增值)?税" + value, normalized)) and bool(
            re.search(r"不含(?:增值)?税" + value, normalized)
        )
    elif key == "banking":
        complete = bool(re.search(r"开户(?:银行|行)[：:][\u4e00-\u9fffA-Za-z]{2,}", normalized)) and bool(
            re.search(r"(?:账号|帐号)[：:]\d{6,}", normalized)
        )
    elif key == "insurance":
        complete = any(term in evidence for term in ("投保", "保单", "保险责任", "保险范围"))
    elif key == "performance_bond":
        complete = any(term in evidence for term in ("履约保证金", "履约保函")) and bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:%|％|万元|元|个?月|日|天)", evidence)
        )
    elif key == "variation_claims":
        complete = any(term in evidence for term in required_terms) and bool(
            re.search(r"书面|\d+\s*(?:日|天)", evidence)
        )
    elif key == "retention_warranty":
        complete = any(term in evidence for term in required_terms) and bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:%|％|年|个?月|日|天)", evidence)
        )
    elif key == "acceptance":
        complete = any(term in evidence for term in required_terms)
    elif key == "liability":
        complete = any(term in evidence for term in required_terms) and bool(
            re.search(r"损失|\d+(?:\.\d+)?\s*(?:%|％|元)", evidence)
        )
    else:
        complete = any(term in evidence for term in required_terms)
    return "found" if complete else "incomplete"


def _field_issue(key: str, evidence: str, status: str) -> str:
    """Explain exactly why an otherwise related clause is not complete."""
    if status != "incomplete":
        return ""
    if key == "term" and evidence:
        return "已定位期限相关表述，但当前提取片段不足以确认具体签订日期、计划开工/竣工日期或明确到期日；请在签署页、补充协议或盖章日期处确认。"
    if key == "amount":
        return "已找到合同价款相关表述，但未能同时确认含税金额和不含税金额；请核对价款页及税率。"
    if key == "banking":
        return "已找到账户相关表述，但开户行、账户名称、账号或纳税信息不完整，请以双方基本信息页为准。"
    if key == "insurance":
        return "合同中出现保险相关表述，但未能确认投保范围、保险期限或保单要求。"
    if key == "performance_bond":
        return "已找到履约保证金相关表述，但未能确认金额、提交时间、扣除条件或返还时间。"
    return "已定位到相关表述，但用于本项核对的关键信息仍不完整，请结合完整合同确认。"


def extract_contract_fields(text: str) -> dict[str, dict[str, Any]]:
    """Locate the operative clause context and mark weak hits as incomplete."""
    fields: dict[str, dict[str, Any]] = {}
    for key, label, _terms, review_focus, anchors, required_terms in CONTRACT_FIELDS:
        # A schedule normally contains a start date and a completion date.  Keep
        # both rather than treating the first date alone as the whole term.
        evidence_limit = {
            "term": 4,
            "parties": 2,
            "amount": 2,
            "banking": 6,
            "insurance": 2,
            "performance_bond": 2,
        }.get(key, 1)
        evidence = _field_evidence(text, anchors, max_items=evidence_limit)
        if key == "parties":
            # Named identity rows outrank later clauses that merely mention
            # either party. Preserve both sides, including blank placeholders.
            compact = _compact_text(text)
            for left, right in (("甲方", "乙方"), ("承包人", "分包人"), ("采购方", "供应方")):
                rows = [re.search(rf"{role}(?:名称|公司名称)?\s*[:：]\s*[^\n。；]+", compact)
                        for role in (left, right)]
                if all(rows):
                    evidence = "\n\n".join(row.group().strip() for row in rows)
                    break
        if key in {"parties", "banking"}:
            # The two parties and their account/tax rows are normally a single
            # “双方基本信息” table. Keep that table together and avoid a random
            # legal-use mention of “承包人” or “账号” elsewhere in the contract.
            compact = _compact_text(text)
            table_start = compact.rfind("双方基本信息")
            table_match = None
            if table_start < 0:
                table_match = re.search(r"双[\s]*方[\s]*基[\s]*本[\s]*信[\s]*息", compact)
                table_start = table_match.start() if table_match else -1
            if table_start >= 0:
                heading_len = len(table_match.group()) if table_match else len("双方基本信息")
                next_heading = re.search(r"\s(?:14\.2|15\.|十五、|十六、|合同附件)", compact[table_start + heading_len :])
                table_end = table_start + 1800
                if next_heading:
                    table_end = min(table_end, table_start + heading_len + next_heading.start())
                table = compact[table_start:table_end].strip()
                table_normalized = re.sub(r"\s+", "", table)
                if key == "parties" and ("承包人" in table_normalized or "甲方" in table_normalized):
                    evidence = table
                elif key == "banking" and ("开户" in table_normalized or "账号" in table_normalized):
                    evidence = table
        if key == "amount":
            # Tax-inclusive and tax-exclusive prices are often separate rows;
            # retain both rows instead of showing only the strongest one.
            compact = _compact_text(text)
            amount_parts = []
            for marker in ("签约合同价（含增值税）", "签约合同价（不含增值税）", "含增值税", "不含增值税", "含税", "不含税"):
                offset = compact.find(marker)
                if offset >= 0:
                    excerpt = _excerpt_around(compact, offset, limit=280)
                    if not any(excerpt in part or part in excerpt for part in amount_parts):
                        amount_parts.append(excerpt)
            if len(amount_parts) >= 2:
                evidence = "\n\n".join(amount_parts[:4])
        if key == "term" and "计划开工日期" in evidence:
            # PDF text often puts the section title and the next safety heading
            # in the same block; show only the schedule rows as evidence. Read
            # through the total-calendar-days row so all dates stay together.
            compact = _compact_text(text)
            start = compact.find("计划开工日期")
            end = compact.find("合同工期", start)
            if end >= 0:
                duration = re.search(r"合同工期.{0,100}?\d+\s*天", compact[end:])
                evidence = compact[start : end + duration.end()] if duration else compact[start : end + 120]
            else:
                evidence = evidence[evidence.find("计划开工日期") :]
            for marker in ("2.质量标准", "2．质量标准", "五、签约合同价"):
                if marker in evidence:
                    evidence = evidence[: evidence.find(marker)].rstrip()
        status = _field_status(key, evidence, required_terms)
        fields[key] = {
            "label": label,
            "review_focus": review_focus,
            "value": evidence,
            "evidence": evidence,
            "status": status,
            "issue": _field_issue(key, evidence, status),
        }
    return fields


def _evidence_sections(text: str, evidence: str) -> list[str]:
    """Only report headings that contain a verified piece of the exact excerpt."""
    sections = re.split(r"(?m)^#{1,6}\s+([^\n]+)\n", text)
    found: list[str] = []
    needles = [re.sub(r"\s+", "", part).rstrip("…") for part in evidence.split("\n\n") if part]
    for index in range(1, len(sections), 2):
        title, body = sections[index], re.sub(r"\s+", "", sections[index + 1])
        if any(len(needle) >= 8 and (needle in body or needle[:60] in body) for needle in needles):
            if title not in found:
                found.append(title)
    return found


def run_contract_checks(text: str, fields: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Return review prompts for missing or ambiguous clauses."""
    prompts: list[dict[str, str]] = []
    for key, label, _terms, review_focus, _anchors, _required_terms in CONTRACT_FIELDS:
        evidence = str(fields.get(key, {}).get("evidence", ""))
        status = fields.get(key, {}).get("status", "missing")
        vague = [term for term in _VAGUE_TERMS if term in evidence]
        if status == "missing":
            prompts.append({"field": key, "severity": "medium", "message": f"待确认条款：未检出{label}。"})
            continue
        if status == "incomplete":
            issue = str(fields.get(key, {}).get("issue", ""))
            vague_suffix = f"；其中包含“{'、'.join(vague)}”等含糊表述" if vague else ""
            prompts.append(
                {
                    "field": key,
                    "severity": "medium",
                    "message": f"待确认条款：{issue or f'已找到与{label}相关的表述，但未能确认{review_focus}是否完整明确'}{vague_suffix}",
                }
            )
            continue
        if vague:
            prompts.append(
                {
                    "field": key,
                    "severity": "low",
                    "message": f"审阅提示：{label}包含表述“{'、'.join(vague)}”，建议确认具体条件或期限。",
                }
            )
    return prompts


def review_contract(
    document: KnowledgeDocument,
    chunks: list[TextChunk],
    query: str = "",
) -> dict[str, Any]:
    # Field extraction must always inspect the full authorized contract.  A user
    # query may narrow the evidence previews, but must not make the checklist
    # review silently inspect only the top retrieved chunks.
    text = document.content or "\n".join(chunk.content for chunk in chunks)
    fields = extract_contract_fields(text)
    prompts = run_contract_checks(text, fields)
    clauses = [
        {
            "field": key,
            "label": value["label"],
            "review_focus": value["review_focus"],
            "evidence": value["evidence"],
            "status": value["status"],
            "issue": value["issue"],
            "evidence_sections": _evidence_sections(text, value["evidence"]),
        }
        for key, value in fields.items()
    ]
    located_count = sum(item["status"] == "found" for item in clauses)
    incomplete_count = sum(item["status"] == "incomplete" for item in clauses)
    pending_count = len(clauses) - located_count
    sources = [
        {
            "title": chunk.title,
            "section": chunk.section,
            "source_path": chunk.source_path,
            "source_type": "contract",
            "rank": index,
            "content_preview": chunk.content,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    return {
        "document_id": document.document_id,
        "document_title": document.title,
        "document_version": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "knowledge_base_id": document.knowledge_base_id,
        "query": query,
        "summary": (
            f"本次按 {len(clauses)} 类常见条款进行定位：已定位 {located_count} 项，"
            f"信息不完整或待确认 {pending_count} 项。定位到原文不代表条款已经充分或无需人工复核。"
        ),
        "review_counts": {
            "checked": len(clauses),
            "located": located_count,
            "incomplete": incomplete_count,
            "pending": pending_count,
        },
        "fields": clauses,
        "risk_prompts": prompts,
        "clauses": clauses,
        "sources": sources,
    }


def export_contract_review(result: dict[str, Any], decisions: dict[str, dict[str, str]] | None = None) -> str:
    """Export a session review for human hand-off; no automatic approval implied."""
    decisions = decisions or {}
    lines = [
        "# 合同条款核对记录", "",
        f"合同：{result.get('document_title', result.get('document_id', ''))}", "",
        f"文本版本（SHA-256）：{result.get('document_version', '')}", "",
        "本记录用于人工复核与交接，不代表签署批准。人工结论由当前操作者填写，未进行身份签名。", "",
    ]
    status_names = {"found": "已定位", "incomplete": "信息不完整", "missing": "未定位"}
    for item in result.get("fields", []):
        decision = decisions.get(item["field"], {})
        status = decision.get("status", "待复核")
        if status not in {"待复核", "已确认", "需补充"}:
            raise ValueError("不支持的人工复核状态")
        lines.extend([
            f"## {item['label']}", "",
            f"系统定位：{status_names.get(item['status'], '待确认')}；人工复核：{status}", "",
            f"核对要点：{item.get('review_focus', '')}", "",
            "原文位置：" + "、".join(item.get("evidence_sections", [])) if item.get("evidence_sections") else "原文位置：未取得可靠页码或章节，需打开完整文件核对", "",
        ])
        for line in item.get("evidence", "未定位到相关文本").splitlines():
            lines.append(f"> {line}")
        if decision.get("note"):
            lines.extend(["", "人工备注：", *[f"> {line}" for line in decision["note"].splitlines()]])
        lines.append("")
    return "\n".join(lines)
