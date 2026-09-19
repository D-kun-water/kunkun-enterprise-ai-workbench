"""Evidence-grounded employee knowledge assistant workflow."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import jieba.posseg as posseg

from app.corpus_adaptations import corpus_adaptations_enabled
from app.ingestion import KnowledgeDocument, is_catalog_text
from app.models import ChatClient
from app.rag import HybridRetriever, SearchResult
from app.utils import split_sentences, tokenize_zh


INSUFFICIENT_ANSWER = "依据不足：现有制度中未检索到能确认该问题的条款，请联系对应责任部门核实。"
MODEL_UNAVAILABLE_ANSWER = "暂时无法生成服务答复，请稍后重试；系统不会在模型不可用时直接拼接制度原文。"
PROMPT_VERSION = "policy-answer-v2"
POLICY_PROMPT_PATH = Path(__file__).parent / "prompts" / "policy_answer.md"


class AnswerConditionError(ValueError):
    """A generated answer dropped a protected qualification from scoped facts."""

    def __init__(self, facts: str) -> None:
        super().__init__("生成答复改变了原文中的原则性限定")
        self.facts = facts

WEAK_QUERY_TOKENS = {
    "什么",
    "怎么",
    "怎么办",
    "如何",
    "多少",
    "多久",
    "几天",
    "时候",
    "为什么",
    "是否",
    "可以",
    "能否",
    "需要",
    "公司",
    "员工",
    "请问",
    "一下",
    "的",
    "了",
    "吗",
    "呢",
    "有",
    "要",
    "是",
    "内",
    "前",
    "后",
}

GENERIC_EVIDENCE_TOKENS = {
    "报销",
    "费用",
    "制度",
    "政策",
    "规定",
    "流程",
    "标准",
    "材料",
    "要求",
    "申请",
    "办理",
    "提交",
    "处理",
    "支持",
    "使用",
    "维修",
    "个人",
    "相关",
    "信息",
    "情况",
    "方式",
    "规则",
    "金额",
    "上限",
    "下限",
    "额度",
    "期限",
}

FOLLOW_UP_PREFIXES = (
    "那",
    "然后",
    "这个",
    "上面",
    "刚才",
    "它",
    "该",
    "还需要",
    "具体",
)

APPLICATION_INTENT_TERMS = (
    "怎么申请",
    "如何申请",
    "申请流程",
    "办理流程",
    "提交申请",
    "怎么办理",
    "怎么报销",
    "如何报销",
    "报销流程",
)

TRAVEL_QUERY_TERMS = ("差旅费", "差旅费用", "差旅报销")
TRAVEL_POLICY_TERMS = (
    "差旅",
    "出差",
    "返程",
    "行程",
    "住宿",
    "交通",
    "补助",
    "机票",
    "车票",
)
TRANSPORT_QUERY_TERMS = ("高铁", "动车", "动车组", "飞机", "轮船", "火车")
TRANSPORT_QUERY_ALIASES = {
    "高铁": ("动车组", "动车"),
    "动车": ("动车组", "高铁"),
    "动车组": ("动车", "高铁"),
}

POLICY_CODE_PATTERN = re.compile(r"\b(?:XQ|KK|XB)-[A-Z0-9]+(?:-[A-Z0-9]+)+\b")


def _current_question_text(question: str) -> str:
    """Return the latest user wording from a contextualized follow-up query."""
    marker = "当前追问："
    if marker in question:
        return question.rsplit(marker, 1)[-1].strip()
    return question


def _expanded_query_tokens(question: str) -> set[str]:
    """Expand scoped vocabulary without changing the displayed question."""
    tokens = set(tokenize_zh(question))
    if _is_contract_drafting_question(_current_question_text(question)):
        # A supplier is the counterparty in this workflow, not necessarily a
        # transport supplier entering a vendor pool. Keep stage and evidence
        # gating aligned with the retrieval rewrite.
        tokens.update(tokenize_zh("合同 起草 相对方 相关方 调查 实地调查"))
    if not corpus_adaptations_enabled():
        return tokens
    normalized = question.replace(" ", "")
    for term, aliases in TRANSPORT_QUERY_ALIASES.items():
        if term in normalized:
            for alias in aliases:
                tokens.update(tokenize_zh(alias))
    return tokens


def contextualize_question(
    question: str, history: Sequence[Mapping[str, Any]] | None = None
) -> str:
    """Attach the latest user topic to a short follow-up question."""
    clean_question = question.strip()
    if not history or not clean_question:
        return clean_question
    previous_question = next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(history)
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not previous_question or previous_question == clean_question:
        return clean_question
    starts_as_follow_up = clean_question.startswith(FOLLOW_UP_PREFIXES)
    # Time-only replies such as “刚入职 2 个月” normally supplement the
    # immediately preceding policy question.  They contain words, so the old
    # topic-free check treated them as a brand-new question and lost “年假”.
    is_employment_time_qualifier = bool(
        re.fullmatch(
            r"(?:(?:我)?(?:刚)?入职\s*)?\d+\s*(?:个)?(?:天|周|星期|月|个月|年|年半)",
            clean_question.replace("，", "").replace("。", ""),
        )
    )
    # A short question with its own business object (e.g. "发票怎么弄") is a
    # new topic, even when it follows an unrelated conversation. Only inherit
    # history for explicit follow-up wording or a topic-free shorthand.
    current_tokens = {
        token
        for token in tokenize_zh(clean_question)
        if token not in WEAK_QUERY_TOKENS and token not in GENERIC_EVIDENCE_TOKENS
    }
    is_topic_free_short_form = len(clean_question) <= 14 and not current_tokens
    if not (starts_as_follow_up or is_topic_free_short_form or is_employment_time_qualifier):
        return clean_question
    return f"上一问：{previous_question}\n当前追问：{clean_question}"


class AgentState(TypedDict, total=False):
    question: str
    route: str
    retrieval: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    debug: dict[str, Any]


def is_catalog_question(question: str) -> bool:
    compact = "".join(question.lower().split())
    patterns = (
        "知识库里有哪些",
        "知识库有哪些",
        "有哪些制度",
        "有哪些文档",
        "收录了什么",
        "收录多少",
        "文档数量",
    )
    return any(pattern in compact for pattern in patterns)


def is_project_meta_question(question: str) -> bool:
    """Keep software/project explanations out of the employee-policy corpus."""
    compact = "".join(question.lower().split())
    patterns = (
        "讲解项目",
        "介绍项目",
        "这个项目是做什么",
        "项目是做什么",
        "项目目标",
        "项目架构",
        "项目功能",
    )
    return any(pattern in compact for pattern in patterns)


def answer_document_catalog(documents: list[str] | list[KnowledgeDocument]) -> str:
    titles = [item.title if isinstance(item, KnowledgeDocument) else str(item) for item in documents]
    titles = list(dict.fromkeys(title for title in titles if title))
    if not titles:
        return "知识库当前还没有可用文档。"
    listing = "；".join(f"{index}. {title}" for index, title in enumerate(titles, start=1))
    return f"知识库当前收录 {len(titles)} 份文档：{listing}。"


def has_sufficient_evidence(question: str, results: list[SearchResult]) -> bool:
    """Require query terms to co-occur in one retrieved policy sentence."""
    return _best_evidence_document(question, results) is not None


@lru_cache(maxsize=256)
def _specific_subjects(question: str) -> tuple[str, ...]:
    """Bind concrete equipment names in maintenance/expense questions.

    A matching action cannot establish coverage for an unseen business object.
    Restrict the guard to equipment suffixes: Chinese segmentation often marks
    actions (such as taking a flight) as nouns. This is a conservative lexical
    safeguard, not a general semantic entailment check.
    """
    if not any(term in question for term in ("维修", "修理", "保养", "报销")):
        return ()
    return tuple(dict.fromkeys(
        item.word.lower() for item in posseg.cut(question)
        if item.flag.startswith("n") and len(item.word) >= 3
        and item.word.endswith(("机", "仪", "器"))
        and not item.word.startswith(("坐", "乘"))
        and item.word not in WEAK_QUERY_TOKENS | GENERIC_EVIDENCE_TOKENS
    ))


def _is_contract_drafting_question(question: str) -> bool:
    if "合同" not in question:
        return False
    if not any(term in question for term in ("调查", "了解", "核查", "考察", "哪些方面", "哪几方面")):
        return False
    return any(term in question for term in ("起草", "草案", "草拟")) or (
        any(term in question for term in ("签署前", "签约前"))
        and any(term in question for term in ("调查", "相对方", "相关方"))
    )


def _best_evidence_document(question: str, results: list[SearchResult]) -> str | None:
    """Find the retrieved document whose sentence best supports the query."""
    if not results:
        return None
    current_question = _current_question_text(question)
    candidate_text = re.sub(r"\s+", "", "\n".join(item.chunk.content for item in results)).lower()
    if any(subject not in candidate_text for subject in _specific_subjects(current_question)):
        return None
    normalized_current_question = current_question.replace(" ", "")
    normalized_full_question = question.replace(" ", "")
    personal_leave_follow_up = (
        "事假" in normalized_full_question
        and any(term in normalized_current_question for term in ("流程", "具体", "怎么", "如何", "申请", "办理"))
    )
    query_tokens = {
        token
        for token in _expanded_query_tokens(question)
        if token not in WEAK_QUERY_TOKENS and (token.isascii() or len(token) > 1)
    }
    has_policy_phrase = "年假" in question or "年休假" in question
    if not query_tokens and not has_policy_phrase:
        return None
    topic_tokens = query_tokens - GENERIC_EVIDENCE_TOKENS
    if not topic_tokens and not has_policy_phrase:
        return None

    threshold = 1 if (
        len(query_tokens) <= 1
        or "婚假" in normalized_current_question
        or personal_leave_follow_up
        or ("事假" in normalized_current_question and any(term in normalized_current_question for term in ("扣钱", "工资", "薪资", "薪水", "扣薪", "无薪")))
        or ("入职" in normalized_current_question and any(term in normalized_current_question for term in ("手续", "流程", "办理")))
    ) else 2
    # Marriage-leave rules are often phrased with only the policy term in the
    # source sentence while the question contains time wording ("多久休完").
    # Treat the explicit policy term as sufficient evidence instead of
    # requiring two lexical topic hits.
    required_topic_hits = 1 if (
        "婚假" in normalized_current_question
        or personal_leave_follow_up
        or ("事假" in normalized_current_question and any(term in normalized_current_question for term in ("扣钱", "工资", "薪资", "薪水", "扣薪", "无薪")))
        or ("入职" in normalized_current_question and any(term in normalized_current_question for term in ("手续", "流程", "办理")))
    ) else min(2, len(topic_tokens))
    best: tuple[int, int, str] | None = None
    for result in results[:10]:
        if is_catalog_text(result.chunk.content):
            continue
        if _is_contract_drafting_question(current_question) and not any(
            term in result.chunk.content for term in ("合同谈判", "合同起草", "起草合同", "合同草案", "签署合同前")
        ):
            continue
        if (
            "入职" in normalized_current_question
            and any(term in normalized_current_question for term in ("手续", "流程", "办理"))
            and not any(
                term in result.chunk.content
                for term in ("录用确认", "报到前准备", "报到材料", "入职登记", "劳动合同", "入职培训", "试用期", "转正")
            )
            and not (
                "报到" in result.chunk.content
                and "材料" in result.chunk.content
            )
        ):
            continue
        sentence_overlap = 0
        for sentence in split_sentences(result.chunk.content):
            sentence_tokens = set(tokenize_zh(sentence))
            lexical_overlap = len(query_tokens & sentence_tokens)
            phrase_bonus = _evidence_score(query_tokens, sentence, question) - lexical_overlap
            topic_score = len(topic_tokens & sentence_tokens) + max(0, phrase_bonus)
            if topic_score < required_topic_hits:
                continue
            sentence_overlap = max(sentence_overlap, lexical_overlap + phrase_bonus)
        if (
            "入职" in normalized_current_question
            and any(term in normalized_current_question for term in ("手续", "流程", "办理"))
            and "报到" in result.chunk.content
            and "材料" in result.chunk.content
        ):
            sentence_overlap = max(sentence_overlap, 1)
        normalized_content = result.chunk.content.replace(" ", "")
        normalized_question = question.replace(" ", "")
        if personal_leave_follow_up and any(
            term in normalized_content
            for term in ("无薪事假", "事假原则上", "返岗后1个工作日", "请假审批、备案与销假流程")
        ):
            sentence_overlap += 8
        # Corpus adaptation: document and phrase preferences for the current dataset.
        if corpus_adaptations_enabled():
            if ("年假" in normalized_question or "年休假" in normalized_question) and any(
                term in normalized_question for term in ("新员工", "新入职", "入职第一年", "入职当年")
            ) and "年休假" in normalized_content and ("新入职" in normalized_content or "新员工" in normalized_content):
                sentence_overlap += 4
            title_text = result.chunk.title.replace(" ", "")
            source_text = result.chunk.source_path.lower()
            if "发票" in normalized_current_question and (
                "发票" in title_text or "invoice" in source_text
            ):
                sentence_overlap += 5
            if any(term in normalized_current_question for term in TRAVEL_QUERY_TERMS) and (
                "差旅" in title_text or "travel" in source_text
            ):
                sentence_overlap += 5
        if sentence_overlap < threshold:
            continue
        # Prefer a policy dedicated to the queried topic over a cross-reference
        # sentence in a neighboring policy (for example, invoice rules cited by
        # the travel policy). The title is a stable document-level signal, but
        # it must not make a one-token generic match look sufficient.
        title_overlap = len(query_tokens & set(tokenize_zh(result.chunk.title)))
        sentence_overlap += min(title_overlap, 2) * 2
        candidate = (sentence_overlap, -result.rank, result.chunk.document_id)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def _evidence_score(query_tokens: set[str], sentence: str, question: str = "") -> int:
    """Score lexical overlap and optional corpus-specific phrase mappings."""
    overlap = len(query_tokens & set(tokenize_zh(sentence)))
    normalized_sentence = sentence.replace(" ", "")
    normalized_question = question.replace(" ", "")
    phrase_bonus = 0
    # Corpus adaptation: phrase-level mappings tuned to the current sample corpus.
    if corpus_adaptations_enabled():
        phrase_bonus += _corpus_phrase_bonus(normalized_question, normalized_sentence, sentence)
    return overlap + phrase_bonus


def _corpus_phrase_bonus(
    normalized_question: str, normalized_sentence: str, raw_sentence: str
) -> int:
    """Corpus-specific phrase mappings controlled by the adaptation switch."""
    phrase_bonus = 0
    if "年假" in normalized_question or "年休假" in normalized_question:
        phrase_bonus += 2 if "年休假" in normalized_sentence or "年假" in normalized_sentence else 0
    if any(term in normalized_question for term in ("新员工", "新入职", "入职第一年", "入职当年")):
        if "新入职" in normalized_sentence or "新员工" in normalized_sentence:
            phrase_bonus += 4
    # Keep a small set of explicit policy-language mappings for common employee wording.
    if "上海" in normalized_question and "住宿" in normalized_question and "一线城市" in normalized_sentence:
        phrase_bonus += 4
    if "采购" in normalized_question and any(term in normalized_question for term in ("报价", "供应商", "万元")):
        if any(term in normalized_sentence for term in ("报价", "供应商", "采购门槛")):
            phrase_bonus += 3
    if "钓鱼" in normalized_question and any(term in normalized_sentence for term in ("IT-SERVICE", "保留截图", "事件报告")):
        phrase_bonus += 3
    if any(term in normalized_question for term in ("被盗", "遗失", "丢失")) and any(
        term in normalized_sentence for term in ("15分钟", "P1", "设备遗失", "被盗")
    ):
        phrase_bonus += 3
    if "发票" in normalized_question and "抬头" in normalized_question:
        if "发票抬头" in normalized_sentence:
            phrase_bonus += 4
        if "XQ-FIN-002" in raw_sentence:
            phrase_bonus -= 2
    if any(term in normalized_question for term in ("访客", "来访", "客户")) and "登记" in normalized_question:
        if "外部访客" in normalized_sentence and "登记" in normalized_sentence:
            phrase_bonus += 4
    # Employee wording often says “差旅费”, while the policy uses “差旅费用”,
    # “出差” or a specific travel step such as “返程”. Treat those terms as
    # equivalent only when the sentence also contains a travel-specific cue.
    if any(term in normalized_question for term in TRAVEL_QUERY_TERMS) and any(
        term in normalized_sentence for term in TRAVEL_POLICY_TERMS
    ):
        phrase_bonus += 3
    if "打车" in normalized_question and any(
        term in normalized_sentence for term in ("出租车", "网约车", "市内交通")
    ):
        phrase_bonus += 3
    return phrase_bonus


def _is_non_policy_sentence(sentence: str) -> bool:
    """Exclude PDF headers and catalog metadata from answer facts."""

    compact = sentence.replace(" ", "")
    if not compact:
        return True
    if compact.startswith(("文档名称：", "文档编号：", "版本：", "适用范围：", "责任部门：", "发布部门：", "负责人：")):
        return True
    if compact.startswith("第") and compact.endswith("页"):
        return True
    if "企业制度知识库样例文件" in compact or "样例制度" in compact:
        return True
    if compact.startswith("本文仅用于") or compact.startswith("本节为样例内容"):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.]", compact) and len(compact) < 20:
        return True
    # PDF page boundaries can leave a dangling clause such as
    # “年休假最小申请单位为”，which is not a complete employee-facing fact.
    if len(compact) < 20 and compact.endswith(("为", "的", "和", "或", "按", "并", "及", "在", "与", "于", "至", "：", ":")):
        return True
    return False


def _is_duplicate_policy_fact(sentence: str, selected_sentences: list[str]) -> bool:
    """Suppress PDF page-overlap fragments without dropping distinct policy rules."""
    candidate = re.sub(r"^\d+[.、]\s*", "", sentence).replace(" ", "")
    if len(candidate) < 12:
        return False
    for selected in selected_sentences:
        existing = re.sub(r"^\d+[.、]\s*", "", selected).replace(" ", "")
        if candidate in existing or existing in candidate:
            return True
        candidate_clause = re.split(r"[，,；;。！？!?]", candidate, maxsplit=1)[0]
        existing_clause = re.split(r"[，,；;。！？!?]", existing, maxsplit=1)[0]
        if len(candidate_clause) >= 12 and candidate_clause == existing_clause:
            return True
        if SequenceMatcher(None, candidate, existing).ratio() >= 0.86:
            return True
    return False


def _clean_answer_sentence(sentence: str) -> str:
    """Hide internal cross-reference IDs from employee-facing answer text."""
    cleaned = re.sub(r"^\s*\d+(?:[.．]\d+)*[.．、)]?\s*", "", sentence)
    return POLICY_CODE_PATTERN.sub("对应制度", cleaned)


def _new_employee_annual_leave_summary(question: str, results: list[SearchResult]) -> str:
    """Explain annual-leave tiers before applying the new-hire proration rule."""
    normalized_question = question.replace(" ", "")
    if not any(term in normalized_question for term in ("年假", "年休假")):
        return ""
    if any(term in _current_question_text(question).replace(" ", "") for term in APPLICATION_INTENT_TERMS):
        return ""
    if not any(term in normalized_question for term in ("新员工", "新入职", "入职当年", "入职第一年")):
        return ""

    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    compact_evidence = re.sub(
        r"\s+",
        "",
        "\n".join(
            result.chunk.content
            for result in results
            if result.chunk.document_id == primary_document_id
        ),
    )
    tier_patterns = (
        ("累计工作满 1 年不满 10 年", r"累计工作已满1年不满10年的，年度年休假为(\d+)天"),
        ("累计工作满 10 年不满 20 年", r"累计工作已满10年不满20年的，年度年休假为(\d+)天"),
        ("累计工作满 20 年", r"累计工作已满20年的，年度年休假为(\d+)天"),
    )
    tiers: list[tuple[str, str]] = []
    for label, pattern in tier_patterns:
        match = re.search(pattern, compact_evidence)
        if match is None:
            tiers = []
            break
        tiers.append((label, match.group(1)))
    if tiers and (
        re.search(r"新入职(?:或当年离职)?员工的可用额度.*?折算", compact_evidence)
        and re.search(r"平台(?:核定|展示)余额", compact_evidence)
    ):
        lines = "\n".join(f"- {label}：{days} 天" for label, days in tiers)
        return (
            "新员工的年假不是统一固定天数。制度先按经核验的累计社会工作年限确定全年标准，"
            "入职当年再由人力资源部折算，最终以平台核定余额为准：\n"
            f"{lines}"
        )

    # The fused 2025 handbook uses a different, more operational wording than
    # the older policy fixture. Normalize that wording into the same compact
    # employee-facing structure instead of falling back to unrelated chunks.
    fused_patterns = (
        ("实际工作 1~6 年（含）", r"实际工作\s*1~6\s*年（含）之间，可享(?:有|受)\s*(\d+)\s*天年休假"),
        ("实际工作 10~16 年（含）", r"实际工作\s*10~16\s*年（含）之间，可享(?:有|受)\s*(\d+)\s*天年休假"),
        ("实际工作满 20 年", r"实际工作满\s*20\s*年，可享(?:有|受)\s*(\d+)\s*天年休假"),
    )
    fused_tiers: list[tuple[str, str]] = []
    for label, pattern in fused_patterns:
        match = re.search(pattern, compact_evidence)
        if match is None:
            fused_tiers = []
            break
        fused_tiers.append((label, match.group(1)))
    has_fused_proration = bool(
        re.search(r"新入职员工的年休假可用额度.*?折算", compact_evidence)
        and re.search(r"员工服务平台核定余额", compact_evidence)
    )
    has_second_calendar_year_rule = "员工自入职后的第二个日历年起" in compact_evidence
    if not (fused_tiers and (has_fused_proration or has_second_calendar_year_rule)):
        return ""
    progression = re.search(
        r"工作满\s*6\s*年后，每服务一年增加一天，至可享有\s*(\d+)\s*天年休假止",
        compact_evidence,
    )
    lines = [f"- {label}：{days} 天" for label, days in fused_tiers]
    if progression:
        lines.insert(1, f"- 工作满 6 年后：每服务 1 年增加 1 天，最高 {progression.group(1)} 天")
    if has_fused_proration:
        qualifier = "入职当年由人力资源部折算，最终以员工服务平台核定余额为准"
    else:
        qualifier = "入职第一年的具体额度需由人力资源部按适用规则核定"
    return (
        "新员工的年假不是统一固定天数。员工自入职后的第二个日历年起享受，"
        f"全年标准按实际工作年限确定；{qualifier}：\n"
        + "\n".join(lines)
    )


def _is_travel_reimbursement_question(question: str) -> bool:
    # Follow-ups such as “那发票怎么弄？” carry the travel topic in the
    # contextualized prefix, while the current wording only contains the
    # narrower intent. Inspect the full contextualized question for topic
    # routing and keep the latest wording for focus selection below.
    normalized_question = question.replace(" ", "")
    return "报销" in normalized_question and any(
        term in normalized_question for term in TRAVEL_QUERY_TERMS
    )


def _travel_reimbursement_summary(question: str, results: list[SearchResult]) -> str:
    """Summarize the dedicated travel policy in its documented process order."""
    if not _is_travel_reimbursement_question(question):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""

    sentences: list[tuple[int, int, int, str]] = []
    for result in results:
        if result.chunk.document_id != primary_document_id:
            continue
        for position, sentence in enumerate(split_sentences(result.chunk.content)):
            if _is_non_policy_sentence(sentence):
                continue
            sentences.append((result.chunk.chunk_index, position, result.rank, sentence))
    sentences.sort(key=lambda item: (item[0], item[1], item[2]))

    current_question = _current_question_text(question).replace(" ", "")
    if "发票" in current_question:
        primary_results = [
            result for result in results if result.chunk.document_id == primary_document_id
        ]
        is_invoice_policy = any(
            "发票" in result.chunk.title or "invoice" in result.chunk.source_path.lower()
            for result in primary_results
        )
        if is_invoice_policy:
            invoice_rule_predicates = (
                lambda text: "发票抬头" in text,
                lambda text: "销售方" in text and ("真实业务" in text or "支付记录" in text),
                lambda text: "电子发票" in text and "原始电子文件" in text,
                lambda text: "同一发票" in text and "报销一次" in text,
                lambda text: "差旅费用" in text and "返程" in text and "提交" in text,
            )
            selected_rules: list[str] = []
            for predicate in invoice_rule_predicates:
                sentence = next((text for _, _, _, text in sentences if predicate(text)), "")
                if not sentence:
                    continue
                cleaned = _clean_answer_sentence(
                    re.sub(r"^\d+[.、)]\s*", "", sentence).strip()
                )
                if not _is_duplicate_policy_fact(cleaned, selected_rules):
                    selected_rules.append(cleaned)
            if selected_rules:
                return "差旅报销发票要求：\n" + "\n".join(
                    f"- {sentence}" for sentence in selected_rules
                )

        invoice_predicates = (
            lambda text: "返程" in text and "提交报销" in text,
            lambda text: "材料" in text and "发票" in text,
            lambda text: "发票抬头" in text or ("发票" in text and "税号" in text),
        )
        selected_invoice: list[str] = []
        for predicate in invoice_predicates:
            sentence = next((text for _, _, _, text in sentences if predicate(text)), "")
            if not sentence:
                continue
            cleaned = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
            if not _is_duplicate_policy_fact(cleaned, selected_invoice):
                selected_invoice.append(cleaned)
        if selected_invoice:
            return "差旅费报销材料：\n" + "\n".join(
                f"- {sentence}" for sentence in selected_invoice
            )

    stage_predicates = (
        lambda text: "出差申请" in text and ("提交" in text or "审批" in text),
        lambda text: "预订" in text and "行程" in text,
        lambda text: "返程" in text and "报销" in text,
        lambda text: "材料" in text and any(term in text for term in ("发票", "行程单", "支付记录", "凭证")),
        lambda text: any(term in text for term in ("直属经理", "预算负责人", "财务部"))
        and any(term in text for term in ("核验", "审核")),
        lambda text: "审批通过" in text and "付款" in text,
    )
    selected: list[str] = []
    for predicate in stage_predicates:
        sentence = next((text for _, _, _, text in sentences if predicate(text)), "")
        if not sentence:
            continue
        cleaned = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
        if not _is_duplicate_policy_fact(cleaned, selected):
            selected.append(cleaned)

    if not selected:
        return ""
    return "差旅费报销流程：\n" + "\n".join(f"- {sentence}" for sentence in selected)


def _transport_reimbursement_summary(question: str, results: list[SearchResult]) -> str:
    """Answer vehicle-class questions from the long-distance travel rule."""
    current_question = _current_question_text(question).replace(" ", "")
    if not any(term in current_question for term in TRANSPORT_QUERY_TERMS):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
        if not _is_non_policy_sentence(sentence)
    ]
    standard = next(
        (
            sentence
            for sentence in sentences
            if "员工级" in sentence
            and any(term in sentence for term in ("飞机经济舱", "动车组二等座", "轮船三等舱"))
        ),
        "",
    )
    booking = next(
        (
            sentence
            for sentence in sentences
            if all(term in sentence for term in ("机票", "火车票", "船票"))
            and "指定渠道" in sentence
        ),
        "",
    )
    if not standard:
        return ""

    if "800公里" in current_question or "连续乘车" in current_question or "超过6小时" in current_question:
        long_distance = next(
            (
                sentence for sentence in sentences
                if ("800公里" in sentence or "连续乘车" in sentence)
                and "飞机经济舱" in sentence
            ),
            "",
        )
        priority = next((sentence for sentence in sentences if "低于上述条件" in sentence and any(term in sentence for term in ("高铁", "动车"))), "")
        if long_distance:
            selected = [long_distance]
            if priority:
                selected.append(priority)
            return "长途交通选择：\n" + "\n".join(f"- {_clean_answer_sentence(sentence)}" for sentence in selected)
    if "高铁" in current_question:
        normalized_standard = (
            "高铁按制度对应动车组，可按动车组座位等级报销：员工级、专业及主管级原则上可报销动车组二等座；"
            "经理及以上可报销动车组一等座。"
        )
    elif "动车组" in current_question or "动车" in current_question:
        normalized_standard = (
            "员工级、专业及主管级原则上可报销动车组二等座；经理及以上可报销动车组一等座。"
        )
    elif "飞机" in current_question:
        normalized_standard = "员工级、专业及主管级和经理及以上均可按飞机经济舱标准报销；超出规定等级需按制度说明并审批。"
    elif "轮船" in current_question:
        normalized_standard = "员工级、专业及主管级原则上可报销轮船三等舱；经理及以上可报销轮船二等舱。"
    else:
        normalized_standard = _clean_answer_sentence(standard)

    selected = [normalized_standard]
    if booking:
        selected.append(_clean_answer_sentence(booking))
    return "交通费用报销：\n" + "\n".join(f"- {sentence}" for sentence in selected)


def _annual_leave_application_summary(question: str, results: list[SearchResult]) -> str:
    """Answer annual-leave application questions with the applicable branch."""
    current_question = _current_question_text(question).replace(" ", "")
    topic_question = question.replace(" ", "")
    if not any(term in topic_question for term in ("年假", "年休假")):
        return ""
    asks_application = any(term in current_question for term in APPLICATION_INTENT_TERMS)
    asks_duration_rule = bool(re.search(r"连续请.*\d+个工作日", current_question))
    if not (asks_application or asks_duration_rule):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
        if not _is_non_policy_sentence(sentence)
    ]
    try:
        requested_days = int(re.search(r"(\d+)个工作日", current_question).group(1))
    except (AttributeError, TypeError, ValueError):
        requested_days = None
    selected: list[str] = []
    if requested_days is not None and requested_days > 3:
        branch = next(
            (
                sentence
                for sentence in sentences
                if "连续请假超过 3 个工作日" in sentence
                or ("连续请假超过3个工作日" in sentence and "至少提前 5 个工作日" in sentence)
            ),
            "",
        )
    else:
        branch = next(
            (
                sentence
                for sentence in sentences
                if "连续请假不超过 3 个工作日" in sentence
                or ("连续请假不超过3个工作日" in sentence and "至少提前 2 个工作日" in sentence)
            ),
            "",
        )
    process = next((sentence for sentence in sentences if "查询余额并选择假别" in sentence), "")
    if process:
        selected.append(process)
    if branch:
        selected.append(branch)
    if not selected:
        selected = [
            sentence
            for sentence in sentences
            if any(term in sentence for term in ("选择假别", "提前", "直属经理"))
        ][:2]
    if not selected:
        return ""
    return "年休假申请要点：\n" + "\n".join(
        f"- {_clean_answer_sentence(re.sub(r'^\d+[.、)]\s*', '', sentence).strip())}"
        for sentence in selected
    )


def _new_hire_material_summary(question: str, results: list[SearchResult]) -> str:
    """Answer the onboarding non-critical-material deadline directly."""
    current_question = _current_question_text(question).replace(" ", "")
    if "非关键材料" not in current_question or "补交" not in current_question:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentence = next(
        (
            sentence
            for result in results
            if result.chunk.document_id == primary_document_id
            for sentence in split_sentences(result.chunk.content)
            if "非关键材料" in sentence and "工作日内补交" in sentence
        ),
        "",
    )
    if not sentence:
        return ""
    return "新员工材料补交：\n- " + _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())


def _onboarding_responsibility_summary(question: str, results: list[SearchResult]) -> str:
    """Keep combined onboarding-responsibility questions on their two owners."""
    current = _current_question_text(question).replace(" ", "")
    if "账号" not in current or "试用期目标" not in current:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    account = next((text for text in sentences if "账号开通申请" in text), "")
    trial = next(
        (
            text
            for text in sentences
            if "用人部门" in text and "试用期目标" in text
        ),
        "",
    )
    selected = [
        _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", text).strip())
        for text in (account, trial)
        if text
    ]
    if len(selected) != 2:
        return ""
    return "报到前职责分工：\n" + "\n".join(f"- {text}" for text in selected)


def _overseas_vpn_summary(question: str, results: list[SearchResult]) -> str:
    """Answer the explicit overseas VPN approval deadline."""
    current_question = _current_question_text(question).replace(" ", "")
    if "VPN" not in current_question.upper() or "境外" not in current_question:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentence = next(
        (
            sentence
            for result in results
            if result.chunk.document_id == primary_document_id
            for sentence in split_sentences(result.chunk.content)
            if "境外" in sentence and "提前 3 个工作日申请" in sentence
        ),
        "",
    )
    if not sentence:
        return ""
    return "境外使用 VPN：\n- " + _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())


def _invoice_expense_summary(question: str, results: list[SearchResult]) -> str:
    """Keep generic invoice questions on the invoice policy's own process."""
    current_question = _current_question_text(question).replace(" ", "")
    if (
        "发票" not in current_question
        or "抬头" in current_question
        or _is_travel_reimbursement_question(question)
    ):
        return ""
    if not any(term in current_question for term in ("怎么", "如何", "流程", "报销", "弄")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
        if not _is_non_policy_sentence(sentence)
    ]
    predicates = (
        lambda text: "取得真实、有效发票" in text,
        lambda text: "创建报销单" in text and "上传" in text,
        lambda text: "财务部执行验真" in text,
    )
    selected: list[str] = []
    for predicate in predicates:
        sentence = next((text for text in sentences if predicate(text)), "")
        if sentence and not _is_duplicate_policy_fact(sentence, selected):
            selected.append(sentence)
    if not selected:
        return ""
    return "发票报销流程：\n" + "\n".join(
        f"- {_clean_answer_sentence(re.sub(r'^\d+[.、)]\s*', '', sentence).strip())}"
        for sentence in selected
    )


def _marriage_leave_summary(question: str, results: list[SearchResult]) -> str:
    """Keep marriage-leave answers on duration and filing-window rules."""
    current = _current_question_text(question).replace(" ", "")
    if "婚假" not in current or not any(term in current for term in ("多久", "几天", "休完", "期限", "时间")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence for result in results if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
        if (
            "婚假" in sentence
            or ("基础标准" in sentence and "工作日" in sentence)
            or ("结婚证" in sentence and ("个月" in sentence or "登记证明" in sentence))
        )
    ]
    prioritized = sorted(
        sentences,
        key=lambda sentence: (
            0 if "基础标准" in sentence or ("工作日" in sentence and "婚假" in sentence) else 1,
            0 if "个月" in sentence and "结婚证" in sentence else 1,
        ),
    )
    selected: list[str] = []
    for sentence in prioritized:
        clean = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
        if not _is_duplicate_policy_fact(clean, selected):
            selected.append(clean)
    return "婚假规定：\n" + "\n".join(f"- {sentence}" for sentence in selected[:2]) if selected else ""


def _onboarding_process_summary(question: str, results: list[SearchResult]) -> str:
    """Extract onboarding steps and exclude neighboring retirement clauses."""
    current = _current_question_text(question).replace(" ", "")
    if "入职" not in current or not any(term in current for term in ("流程", "手续", "办理", "需要")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    keywords = ("录用确认", "报到前准备", "报到材料", "入职登记", "劳动合同", "账号、设备与门禁", "入职培训", "试用期", "转正")
    candidates: list[tuple[int, int, str]] = []
    for result in results:
        if result.chunk.document_id != primary_document_id:
            continue
        for position, sentence in enumerate(split_sentences(result.chunk.content)):
            if _is_non_policy_sentence(sentence) or "退休" in sentence or "养老金" in sentence:
                continue
            if not any(term in sentence for term in keywords):
                continue
            clean = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
            candidates.append((result.chunk.chunk_index, position, clean))
    selected: list[str] = []
    for _, _, clean in sorted(candidates, key=lambda item: (item[0], item[1])):
        if not _is_duplicate_policy_fact(clean, selected):
            selected.append(clean)
    return "新员工入职流程：\n" + "\n".join(f"- {sentence}" for sentence in selected[:8]) if selected else ""


def _travel_reimbursement_material_summary(question: str, results: list[SearchResult]) -> str:
    """Answer travel-material questions without appending exception rules."""
    current = _current_question_text(question).replace(" ", "")
    if not any(term in current for term in TRAVEL_QUERY_TERMS):
        return ""
    if not any(term in current for term in ("哪些材料", "什么材料", "材料", "凭证")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    selected: list[str] = []
    for result in results:
        if result.chunk.document_id != primary_document_id:
            continue
        for sentence in split_sentences(result.chunk.content):
            if _is_non_policy_sentence(sentence):
                continue
            if "报销单" not in sentence or not any(term in sentence for term in ("材料", "附件", "票据", "凭证")):
                continue
            if any(term in sentence for term in ("退改签", "公共交通停运", "不可抗力")):
                continue
            clean = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
            if not _is_duplicate_policy_fact(clean, selected):
                selected.append(clean)
    return "差旅费报销材料：\n" + "\n".join(f"- {sentence}" for sentence in selected[:3]) if selected else ""


def _personal_leave_summary(question: str, results: list[SearchResult]) -> str:
    """Answer personal-leave application questions without annual-leave rules."""
    current = _current_question_text(question).replace(" ", "")
    asks_pay = any(term in current for term in ("扣钱", "工资", "薪资", "薪水", "扣薪", "无薪"))
    if "事假" not in current or not (asks_pay or any(
        term in current for term in ("怎么", "如何", "申请", "流程", "办理", "提前", "累计", "最多", "多少")
    )):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    candidates: list[str] = []
    for result in results:
        if result.chunk.document_id != primary_document_id:
            continue
        for sentence in split_sentences(result.chunk.content):
            if _is_non_policy_sentence(sentence) or "年假" in sentence or "年休假" in sentence:
                continue
            if "事假" not in sentence and not any(term in sentence for term in ("提前1个工作日", "返岗后1个工作日")):
                continue
            clean = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
            if not _is_duplicate_policy_fact(clean, candidates):
                candidates.append(clean)
    selected: list[str] = []
    for predicate in (
        lambda text: "事假" in text and ("无薪" in text or "不计发请假期间工资" in text),
        lambda text: "提前1个工作日" in text or "返岗后1个工作日" in text,
        lambda text: "事假年度累计" in text and "10个工作日" in text,
        lambda text: "直属负责人" in text and "审批" in text,
        lambda text: "紧急情况" in text and "通知" in text,
    ):
        sentence = next((text for text in candidates if predicate(text)), "")
        if sentence and sentence not in selected:
            selected.append(sentence)
    if asks_pay and any("事假不计发请假期间" in result.chunk.content for result in results if result.chunk.document_id == primary_document_id):
        pay_sentence = "事假不计发请假期间工资，考勤和薪资处理按公司当月规则执行。"
        selected.insert(0, pay_sentence)
    if not selected:
        return ""
    if asks_pay:
        deadline = next((sentence for sentence in selected if "提前1个工作日" in sentence), "")
        reply = "按现行员工管理制度，事假属于无薪事假，请假期间不计发工资；考勤和薪资处理还要按公司当月规则执行。也就是说，具体应发金额要结合当月薪资规则核算，不能只按‘扣多少钱’一概而论。"
        if deadline:
            clean_deadline = re.sub(r"^（?\d+[）.、]\s*", "", deadline)
            clean_deadline = clean_deadline.replace("事假原则上应", "原则上应", 1)
            reply += f"如果你准备申请，{clean_deadline}"
        return reply
    return "事假申请可以这样办理：\n" + "\n".join(f"- {sentence}" for sentence in selected[:3])


def _travel_reimbursement_deadline_summary(question: str, results: list[SearchResult]) -> str:
    """Answer reimbursement submission deadlines without pre-trip application rules."""
    current = _current_question_text(question).replace(" ", "")
    if not any(term in current for term in TRAVEL_QUERY_TERMS) or not any(term in current for term in ("多久", "时限", "提交", "几天")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    selected: list[str] = []
    for result in results:
        if result.chunk.document_id != primary_document_id:
            continue
        for sentence in split_sentences(result.chunk.content):
            if _is_non_policy_sentence(sentence) or "出差前" in sentence or "出差申请" in sentence:
                continue
            if not any(term in sentence for term in ("提交报销", "次月", "逾期")):
                continue
            clean = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())
            if not _is_duplicate_policy_fact(clean, selected):
                selected.append(clean)
    return "差旅报销提交时限：\n" + "\n".join(f"- {sentence}" for sentence in selected[:3]) if selected else ""


def _invoice_title_summary(question: str, results: list[SearchResult]) -> str:
    """Answer invoice-header questions with the single governing rule."""
    current_question = _current_question_text(question).replace(" ", "")
    if "发票" not in current_question or "抬头" not in current_question:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentence = next(
        (
            sentence
            for result in results
            if result.chunk.document_id == primary_document_id
            for sentence in split_sentences(result.chunk.content)
            if "发票抬头" in sentence
            and any(term in sentence for term in ("样例主体全称", "公司全称", "统一社会信用代码"))
        ),
        "",
    )
    if not sentence:
        return ""
    lines = [_clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", sentence).strip())]
    repeat_rule = next(
        (
            text for result in results if result.chunk.document_id == primary_document_id
            for text in split_sentences(result.chunk.content)
            if "同一发票" in text and "报销一次" in text
        ),
        "",
    )
    if repeat_rule:
        lines.append(_clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", repeat_rule).strip()))
    return "发票抬头：\n" + "\n".join(f"- {line}" for line in lines)


def _travel_lodging_and_deadline_summary(question: str, results: list[SearchResult]) -> str:
    """Cover both requested facts when a travel question joins lodging and timing."""
    current = _current_question_text(question).replace(" ", "")
    if not (
        "住宿" in current
        and any(term in current for term in ("出差", "差旅"))
        and any(term in current for term in ("时限", "期限", "提交", "什么时候", "多久"))
    ):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    lodging = next(
        (text for text in sentences if "员工级住宿标准" in text and "一线城市" in text),
        "",
    )
    deadline = next(
        (text for text in sentences if "提交报销" in text and "10个工作日" in text),
        "",
    )
    selected = [
        _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", text).strip())
        for text in (lodging, deadline)
        if text
    ]
    if len(selected) != 2:
        return ""
    return "差旅住宿与报销时限：\n" + "\n".join(f"- {text}" for text in selected)


def _travel_attachment_and_invoice_title_summary(question: str, results: list[SearchResult]) -> str:
    """Answer the consequences of a missing travel attachment and invoice title together."""
    current = _current_question_text(question).replace(" ", "")
    if "行程单" not in current or "发票抬头" not in current:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    attachment = next((text for text in sentences if "报销单须附" in text), "")
    consequence = next((text for text in sentences if "财务部门可退回补充" in text), "")
    invoice_title = next(
        (text for text in sentences if "发票抬头" in text and "统一社会信用代码" in text),
        "",
    )
    selected = [
        _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", text).strip())
        for text in (attachment, consequence, invoice_title)
        if text
    ]
    if len(selected) != 3:
        return ""
    return "差旅票据与附件：\n" + "\n".join(f"- {text}" for text in selected)


def _contract_nonperformance_summary(question: str, results: list[SearchResult]) -> str:
    """Prefer the contract-performance exception path over archive clauses."""
    current = _current_question_text(question).replace(" ", "")
    if "合同" not in current or not any(term in current for term in ("不履行", "未履行", "不完全履行")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    explanation = next(
        (text for text in sentences if "承办人将情况向公司法律顾问作出详细说明" in text),
        "",
    )
    written = next((text for text in sentences if "书面方式" in text and "提出异议" in text), "")
    if explanation and written:
        action = (
            "承办人应向公司法律顾问详细说明情况；法律顾问提出合理化建议后，"
            "由承办人在法定、约定或合理期限内以书面方式向对方提出异议。"
        )
    else:
        action = ""
    if not action:
        return ""
    return f"合同履约异常处理：\n- {action}"


def _contract_counterparty_investigation_summary(question: str, results: list[SearchResult]) -> str:
    """Answer pre-draft counterparty checks without later performance clauses."""
    current = _current_question_text(question).replace(" ", "")
    if not _is_contract_drafting_question(current):
        return ""
    if not any(term in current for term in ("调查", "相对方", "相关方", "哪些方面")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    investigation = next(
        (
            text
            for text in sentences
            if "签署合同前" in text
            and "主体资格" in text
            and "资信状况" in text
            and "履行能力" in text
        ),
        "",
    )
    if not investigation:
        return ""
    cleaned = _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", investigation).strip())
    return f"合同相对方调查：\n- {cleaned}"


def _barge_supplier_admission_summary(question: str, results: list[SearchResult]) -> str:
    """Return the registration, licence, and insurance requirements as one admission answer."""
    current = _current_question_text(question).replace(" ", "")
    if "驳船" not in current or not any(term in current for term in ("注册资金", "许可证", "许可", "保险")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    registration = next((text for text in sentences if "注册资金不得少于200万元" in text.replace(" ", "")), "")
    licence = next((text for text in sentences if "水路运输许可证" in text), "")
    insurance = next((text for text in sentences if "船舶险" in text and "承运人责任险" in text), "")
    selected = [
        _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", text).strip())
        for text in (registration, licence, insurance)
        if text
    ]
    if len(selected) != 3:
        return ""
    return "驳船供应商准入要求：\n" + "\n".join(f"- {text}" for text in selected)


def _barge_supplier_application_summary(question: str, results: list[SearchResult]) -> str:
    """Keep new-barge-supplier questions on the documented application and materials."""
    current = _current_question_text(question).replace(" ", "")
    if "驳船供应商" not in current or "新增" not in current or not any(term in current for term in ("申请", "材料", "提交")):
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
    ]
    application = next((text for text in sentences if "采购需求、预算" in text and "新增" in text), "")
    materials = next((text for text in sentences if "同步提供相关资料" in text and "水路运输许可证" in text), "")
    vessel_materials = next(
        (text for text in sentences if "船舶配置与服务航线" in text and "船舶保单" in text),
        "",
    )
    selected = [
        _clean_answer_sentence(re.sub(r"^\d+[.、)]\s*", "", text).strip())
        for text in (application, materials, vessel_materials)
        if text
    ]
    if len(selected) != 3:
        return ""
    return "新增驳船供应商：\n" + "\n".join(f"- {text}" for text in selected)


def _source_results_for_question(question: str, results: list[SearchResult]) -> list[SearchResult]:
    """Hide retrieved chunks that are not evidence for the requested intent."""
    current = _current_question_text(question).replace(" ", "")
    contextual_topic = question.replace(" ", "")
    if not results:
        return results

    def keep(result: SearchResult) -> bool:
        content = re.sub(r"\s+", "", result.chunk.content)
        if "账号" in current and "试用期目标" in current:
            return "账号开通申请" in content or ("用人部门" in content and "试用期目标" in content)
        if "转正" in current and any(term in current for term in ("审核", "审批", "提前", "工作日")):
            return "转正" in content
        if "住宿" in current and any(term in current for term in ("时限", "期限", "多久", "报销单")):
            return "住宿标准" in content or "跨月费用" in content or "报销流程与时限" in content
        if "驳船" in current and any(term in current for term in ("准入", "注册资金", "许可", "保险")):
            return "驳船供应商" in content and any(term in content for term in ("注册资金", "水路运输许可证", "承运人责任险"))
        if _is_contract_drafting_question(current) and any(
            term in current for term in ("调查", "相对方", "相关方", "哪些方面")
        ):
            return "签署合同前" in content and "主体资格" in content
        if "合同" in current and any(term in current for term in ("不履行", "未履行", "不完全履行")):
            return "不履行或不完全履行" in content or (
                "提出异议" in content and "法律顾问" in content
            )
        if "驳船供应商" in current and "新增" in current and any(
            term in current for term in ("申请", "材料", "提交")
        ):
            return "申请新增供应商流程" in content or (
                "采购需求、预算" in content and "同步提供相关资料" in content
            )
        if "入职" in current and any(term in current for term in ("手续", "流程", "办理")):
            onboarding_terms = ("录用确认", "报到前准备", "报到材料", "入职登记", "新员工入职办理流程")
            return any(term in content for term in onboarding_terms)
        if "事假" in current or (
            "事假" in contextual_topic and any(term in current for term in ("流程", "具体", "怎么", "如何", "申请", "办理"))
        ):
            return (
                any(term in content for term in ("无薪事假", "事假原则上", "返岗后1个工作日", "事假年度累计"))
                and "退休" not in content
            )
        if "年假" in current or "年休假" in current:
            return "带薪年休假" in content or "年休假" in content
        if "行程单" in current and "发票抬头" in current:
            return "报销单须附" in content or "发票抬头" in content
        if "婚假" in current:
            return "婚假" in content or "结婚证" in content
        if any(term in current for term in TRAVEL_QUERY_TERMS) and any(term in current for term in ("材料", "凭证")):
            return "报销单须附" in content or "票据与附件" in content
        if any(term in current for term in TRAVEL_QUERY_TERMS) and any(term in current for term in ("多久", "时限", "提交")):
            return "提交报销" in content or "跨月费用" in content or "报销流程与时限" in content
        if "发票" in current and "抬头" in current:
            return "发票抬头" in content
        if "800公里" in current or "连续乘车" in current or "超过6小时" in current:
            return "800公里" in content or "连续乘车" in content
        if any(term in current for term in TRANSPORT_QUERY_TERMS):
            return any(term in content for term in ("飞机经济舱", "动车组二等座", "轮船三等舱", "指定渠道"))
        if "住宿" in current and "标准" in current:
            return "住宿标准" in content or ("一线城市" in content and "每人每天" in content)
        if any(term in current for term in ("晋升", "调薪")) and any(term in current for term in ("评审", "怎么", "如何")):
            return "晋升与调薪评审" in content
        return True

    filtered = [result for result in results if keep(result)]
    if filtered:
        return sorted(filtered, key=lambda result: result.rank)
    return results


def _lodging_standard_summary(question: str, results: list[SearchResult]) -> str:
    """Answer city lodging-cap questions without unrelated travel rules."""
    current_question = _current_question_text(question).replace(" ", "")
    if "住宿" not in current_question or "标准" not in current_question:
        return ""
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return ""
    sentences = [
        sentence
        for result in results
        if result.chunk.document_id == primary_document_id
        for sentence in split_sentences(result.chunk.content)
        if not _is_non_policy_sentence(sentence)
    ]
    cap = next((sentence for sentence in sentences if "住宿上限" in sentence), "")
    city_definition = next((sentence for sentence in sentences if "一线城市" in sentence and "指" in sentence), "")
    if not cap:
        return ""
    selected = [cap]
    if city_definition and "上海" not in current_question:
        selected.append(city_definition)
    return "住宿标准：\n" + "\n".join(
        f"- {_clean_answer_sentence(re.sub(r'^\d+[.、)]\s*', '', sentence).strip())}"
        for sentence in selected
    )


def compose_local_answer(question: str, results: list[SearchResult], max_sentences: int = 3) -> str:
    """Create a deterministic answer by extracting only retrieved sentences."""

    if not has_sufficient_evidence(question, results):
        return INSUFFICIENT_ANSWER

    query_tokens = _expanded_query_tokens(question)
    normalized_question = question.replace(" ", "")
    current_question = _current_question_text(question)
    normalized_current_question = current_question.replace(" ", "")
    asks_application = any(term in normalized_current_question for term in APPLICATION_INTENT_TERMS)
    primary_document_id = _best_evidence_document(question, results)
    if primary_document_id is None:
        return INSUFFICIENT_ANSWER
    # Corpus adaptation: bespoke summary composers for fixed-corpus cases.
    if corpus_adaptations_enabled():
        onboarding_responsibility = _onboarding_responsibility_summary(question, results)
        if onboarding_responsibility:
            return onboarding_responsibility
        annual_leave_summary = _new_employee_annual_leave_summary(question, results)
        if annual_leave_summary:
            return annual_leave_summary
        annual_leave_application = _annual_leave_application_summary(question, results)
        if annual_leave_application:
            return annual_leave_application
        material_summary = _new_hire_material_summary(question, results)
        if material_summary:
            return material_summary
        vpn_summary = _overseas_vpn_summary(question, results)
        if vpn_summary:
            return vpn_summary
        marriage_leave_summary = _marriage_leave_summary(question, results)
        if marriage_leave_summary:
            return marriage_leave_summary
        onboarding_summary = _onboarding_process_summary(question, results)
        if onboarding_summary:
            return onboarding_summary
        personal_leave_summary = _personal_leave_summary(question, results)
        if personal_leave_summary:
            return personal_leave_summary
        lodging_and_deadline_summary = _travel_lodging_and_deadline_summary(question, results)
        if lodging_and_deadline_summary:
            return lodging_and_deadline_summary
        attachment_and_invoice_title_summary = _travel_attachment_and_invoice_title_summary(question, results)
        if attachment_and_invoice_title_summary:
            return attachment_and_invoice_title_summary
        contract_counterparty_investigation = _contract_counterparty_investigation_summary(question, results)
        if contract_counterparty_investigation:
            return contract_counterparty_investigation
        contract_nonperformance_summary = _contract_nonperformance_summary(question, results)
        if contract_nonperformance_summary:
            return contract_nonperformance_summary
        barge_supplier_admission_summary = _barge_supplier_admission_summary(question, results)
        if barge_supplier_admission_summary:
            return barge_supplier_admission_summary
        barge_supplier_application_summary = _barge_supplier_application_summary(question, results)
        if barge_supplier_application_summary:
            return barge_supplier_application_summary
        travel_material_summary = _travel_reimbursement_material_summary(question, results)
        if travel_material_summary:
            return travel_material_summary
        travel_deadline_summary = _travel_reimbursement_deadline_summary(question, results)
        if travel_deadline_summary:
            return travel_deadline_summary
        invoice_title_summary = _invoice_title_summary(question, results)
        if invoice_title_summary:
            return invoice_title_summary
        lodging_summary = _lodging_standard_summary(question, results)
        if lodging_summary:
            return lodging_summary
        transport_summary = _transport_reimbursement_summary(question, results)
        if transport_summary:
            return transport_summary
        travel_summary = _travel_reimbursement_summary(question, results)
        if travel_summary:
            return travel_summary
        invoice_summary = _invoice_expense_summary(question, results)
        if invoice_summary:
            return invoice_summary
    candidates: list[tuple[float, int, str, str]] = []
    for result in (item for item in results if item.chunk.document_id == primary_document_id):
        for position, sentence in enumerate(split_sentences(result.chunk.content)):
            sentence_tokens = set(tokenize_zh(sentence))
            overlap = len(query_tokens & sentence_tokens)
            code_bonus = sum(1 for token in query_tokens if token.isascii() and token in sentence.lower())
            score = _evidence_score(query_tokens, sentence, question) + code_bonus * 2 + 1.0 / (result.rank + position + 1)
            # Corpus adaptation: intent-level sentence preferences.
            if corpus_adaptations_enabled() and asks_application:
                compact_sentence = sentence.replace(" ", "")
                intent_bonus = 0
                if "选择假别" in compact_sentence or "办理流程" in compact_sentence:
                    intent_bonus = 20
                elif "查询余额" in compact_sentence:
                    intent_bonus = 14
                elif any(
                    term in compact_sentence
                    for term in ("提交申请", "申请", "提交报销", "创建报销单", "上传原始")
                ):
                    intent_bonus = 8
                # A follow-up should answer its latest intent first, while the
                # previous turn remains available to identify the policy topic.
                score += intent_bonus
            if corpus_adaptations_enabled() and ("年假" in normalized_current_question or "年休假" in normalized_current_question) and any(
                term in normalized_current_question for term in ("新员工", "新入职", "入职第一年", "入职当年")
            ) and ("新入职" in sentence or "新员工" in sentence) and ("折算" in sentence or "平台" in sentence):
                score += 8
            candidates.append((score, result.rank, result.chunk.title, sentence))

    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    answer_limit = (
        2
        if corpus_adaptations_enabled() and "发票" in normalized_question and "抬头" in normalized_question
        else max_sentences
    )
    for score, _, title, sentence in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if _is_non_policy_sentence(sentence):
            continue
        dedup_key = re.sub(r"^\d+[.、]\s*", "", sentence).replace(" ", "")
        if dedup_key in seen:
            continue
        if _is_duplicate_policy_fact(sentence, [item[1] for item in selected]):
            continue
        seen.add(dedup_key)
        selected.append((title, sentence))
        if len(selected) >= answer_limit:
            break

    if not selected:
        return INSUFFICIENT_ANSWER
    facts = "\n".join(f"- {_clean_answer_sentence(sentence)}" for _, sentence in selected)
    cited_titles = "、".join(dict.fromkeys(title for title, _ in selected))
    return f"根据《{cited_titles}》中的相关规定：\n{facts}\n\n请以页面下方展示的原文来源为准。"


class KnowledgeAssistant:
    """Route catalog questions and policy questions through one small workflow."""

    def __init__(
        self,
        retriever: HybridRetriever,
        documents: list[KnowledgeDocument],
        *,
        llm_provider: str = "ollama",
        llm_client: ChatClient | None = None,
        answer_top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.documents = [document for document in documents if document.status == "ready"]
        self.llm_provider = llm_provider
        self.llm_client = llm_client
        self.answer_top_k = answer_top_k
        self.workflow_error = ""
        self.last_llm_error = ""
        self.workflow = self._build_workflow()

    def ask(
        self,
        question: str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        original_question = question.strip()
        initial: AgentState = {"question": contextualize_question(original_question, history)}
        if not original_question:
            return {
                "answer": "请输入要查询的问题。",
                "sources": [],
                "debug": {"route": "invalid", "llm_provider": self.llm_provider},
            }
        if self.workflow is not None:
            state = self.workflow.invoke(initial)
        else:
            state = self._run_sequential(initial)
        return {
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "debug": state.get("debug", {}),
        }

    def _route_question(self, state: AgentState) -> AgentState:
        if is_catalog_question(state["question"]):
            route = "catalog"
        elif is_project_meta_question(state["question"]):
            route = "project_meta"
        else:
            route = "retrieval"
        return {"route": route}

    def _retrieve_context(self, state: AgentState) -> AgentState:
        if state["route"] in {"catalog", "project_meta"}:
            return {"retrieval": {"final_results": []}}
        # Follow-ups need a little more context to recover the process section
        # that may sit in a later chunk, while standalone queries keep the
        # original compact top-k behavior.
        top_k = max(self.answer_top_k, 20) if "当前追问：" in state["question"] else max(self.answer_top_k, 10)
        retrieval_question = state["question"]
        if _is_contract_drafting_question(_current_question_text(retrieval_question)):
            # Retrieve by the requested workflow stage. Broad nouns such as
            # supplier must not swamp contract-drafting evidence with admission
            # policies. The original user question is preserved for generation.
            retrieval_question = "合同起草 签署合同前 对合同相关方调查"
        return {"retrieval": self.retriever.retrieve(retrieval_question, top_k=top_k)}

    def _compose_answer(self, state: AgentState) -> AgentState:
        self.last_llm_error = ""
        if state["route"] == "project_meta":
            return {
                "answer": (
                    "这个窗口目前用于查询已入库的员工制度和办事流程，不负责讲解软件项目，"
                    "如果要了解项目，请直接查看项目说明文档或向项目负责人咨询。"
                ),
                "sources": [],
                "debug": {
                    "route": "project_meta",
                    "llm_provider": "direct",
                    "workflow_error": self.workflow_error,
                },
            }
        if state["route"] == "catalog":
            return {
                "answer": answer_document_catalog(self.documents),
                "sources": [],
                "debug": {
                    "route": "catalog",
                    "llm_provider": "direct",
                    "workflow_error": self.workflow_error,
                },
            }

        retrieval = state.get("retrieval", {})
        results: list[SearchResult] = retrieval.get("final_results", [])
        evidence_document_id = _best_evidence_document(state["question"], results)
        evidence_sufficient = evidence_document_id is not None
        if not evidence_sufficient:
            results = []
        else:
            results = [result for result in results if result.chunk.document_id == evidence_document_id]
        results = _source_results_for_question(state["question"], results)
        llm_error = ""
        validation_error = ""
        answer_mode = "model"
        if not evidence_sufficient:
            answer = INSUFFICIENT_ANSWER
            answer_mode = "refusal"
        elif self.llm_client is None:
            llm_error = "未配置可用的大模型服务"
            self.last_llm_error = llm_error
            answer = MODEL_UNAVAILABLE_ANSWER
            answer_mode = "unavailable"
        else:
            try:
                answer = self._compose_llm_answer(state["question"], results)
            except AnswerConditionError as exc:
                validation_error = str(exc)
                answer_mode = "source_facts"
                answer = (
                    "本次生成答复未通过限定条件校验，以下为原文要点整理，请结合来源核对：\n\n"
                    + exc.facts
                )
            except Exception as exc:
                llm_error = str(exc)
                self.last_llm_error = llm_error
                answer = MODEL_UNAVAILABLE_ANSWER
                answer_mode = "unavailable"

        return {
            "answer": answer,
            "sources": self._format_sources(results),
            "debug": {
                "route": "retrieval",
                "llm_provider": self.llm_provider,
                "vector_backend": retrieval.get("vector_backend", "unknown"),
                "vector_error": retrieval.get("vector_error", ""),
                "reranker_error": retrieval.get("reranker_error", ""),
                "llm_error": llm_error,
                "workflow_error": self.workflow_error,
                "retrieved_count": len(results),
                "evidence_sufficient": evidence_sufficient,
                "prompt_version": PROMPT_VERSION,
                "answer_mode": answer_mode,
                "answer_validation_error": validation_error,
            },
        }

    def _compose_llm_answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return INSUFFICIENT_ANSWER
        current_question = _current_question_text(question)
        source_text = "\n".join(result.chunk.content for result in results)
        # Elapsed duration does not identify the joining calendar year: two
        # months may span December and January. Ask for dates instead of
        # inventing an eligibility decision from duration alone.
        employment_duration_match = re.fullmatch(
            r"(?:(?:我)?(?:刚)?入职\s*)?(\d+\s*(?:个)?(?:天|周|星期|月|个月|年|年半))",
            current_question.replace("，", "").replace("。", ""),
        )
        is_annual_leave_time_follow_up = (
            any(term in question for term in ("年假", "年休假"))
            and employment_duration_match is not None
            and "自入职后的第二个日历年起" in source_text
        )
        if is_annual_leave_time_follow_up:
            return (
                "现有制度写的是‘自入职后的第二个日历年起’，不是‘入职满一年’。"
                f"仅凭入职 {employment_duration_match.group(1)} 不能完整核定年假；"
                "请补充入职日期、拟休假日期和累计工作年限，并核对平台核定余额。"
            )
        context = "\n\n".join(
            f"[来源 {index}] 文档：{result.chunk.title}\n章节：{result.chunk.section}\n原文：{result.chunk.content}"
            for index, result in enumerate(results, start=1)
        )
        fact_checklist = compose_local_answer(question, results, max_sentences=1)
        if fact_checklist == INSUFFICIENT_ANSWER:
            fact_checklist = ""
        detail_requested = any(
            term in current_question for term in ("具体案例", "举例", "明细", "完整流程", "全部", "分别", "各档", "不同工龄")
        )
        system_prompt = POLICY_PROMPT_PATH.read_text(encoding="utf-8")
        system_prompt += (
            "本次员工没有明确要求详细展开：只用一段自然语言回答，不使用项目符号，不补充事实范围以外的提醒、分档或案例。"
            if not detail_requested
            else "本次员工明确要求详细展开：可分点说明，但仍不得照搬条款编号或圆圈序号。"
        )
        # Bespoke summaries are produced only for fixed-corpus questions whose
        # neighboring clauses previously contaminated otherwise correct answers.
        # Give the model that verified scope instead of the wider chunks; the
        # original chunks remain available in the response sources for review.
        has_strict_fact_scope = bool(fact_checklist) and not fact_checklist.startswith("根据《")
        if has_strict_fact_scope:
            user_prompt = (
                f"员工问题：{question}\n\n"
                f"经校验的唯一回答事实范围：\n{fact_checklist}\n\n"
                "请只改写上述事实范围，不得增加其他步骤、提醒或条件。"
            )
            if "原则上" in fact_checklist:
                user_prompt += "\n必须保留原文的‘原则上’限定语，不得把原则性要求改写为无例外的上限或硬性条件。"
        else:
            checklist_block = f"\n\n从原文抽取的事实清单：\n{fact_checklist}" if fact_checklist else ""
            user_prompt = f"员工问题：{question}{checklist_block}\n\n制度原文：\n{context}"
        answer = self.llm_client.generate(system_prompt, user_prompt) if self.llm_client else INSUFFICIENT_ANSWER
        if has_strict_fact_scope and answer.count("原则上") != fact_checklist.count("原则上"):
            # Conservative lexical check, not semantic certification. Even a
            # legitimate combined paraphrase can fall back to the scoped facts.
            # Neither remove a qualification nor invent an exception to a rule.
            raise AnswerConditionError(fact_checklist)
        if "自入职后的第二个日历年起" in source_text:
            answer = re.sub(r"(?:自)?入职满一年后", "自入职后的第二个日历年起", answer)
            answer = re.sub(r"如果您工龄刚满一年", "如果您刚进入入职后的第二个日历年", answer)
        if (
            not detail_requested
            and any(term in current_question for term in ("年假", "年休假"))
            and "自入职后的第二个日历年起" in source_text
        ):
            return "年假从入职后的第二个日历年起开始计算，具体天数要看累计工作年限，而不只看当前入职时间。你把累计工龄告诉我，我可以继续帮你核对对应天数。"
        return answer

    @staticmethod
    def _format_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "title": result.chunk.title,
                "section": result.chunk.section,
                "source_path": result.chunk.source_path,
                "source_type": Path(result.chunk.source_path).suffix.lstrip(".") or "text",
                "rank": result.rank,
                "score": round(result.score, 6),
                "content_preview": result.chunk.content,
            }
            for result in results
        ]

    def _build_workflow(self):
        try:
            from langgraph.graph import END, START, StateGraph

            builder = StateGraph(AgentState)
            builder.add_node("route_question", self._route_question)
            builder.add_node("retrieve_context", self._retrieve_context)
            builder.add_node("compose_answer", self._compose_answer)
            builder.add_edge(START, "route_question")
            builder.add_edge("route_question", "retrieve_context")
            builder.add_edge("retrieve_context", "compose_answer")
            builder.add_edge("compose_answer", END)
            return builder.compile()
        except Exception as exc:
            self.workflow_error = str(exc)
            return None

    def _run_sequential(self, state: AgentState) -> AgentState:
        state.update(self._route_question(state))
        state.update(self._retrieve_context(state))
        state.update(self._compose_answer(state))
        return state
