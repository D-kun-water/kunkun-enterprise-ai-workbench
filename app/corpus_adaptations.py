"""Explicit switch for corpus-specific answer adaptations.

The local answer pipeline is produced by two layers:

1. **Generic evidence layer** (always on): token overlap, BM25/vector retrieval,
   weak-token filtering, topic-token gating, title overlap, deduplication and
   metadata-sentence filtering. These are corpus-independent.

2. **Corpus adaptation layer** (switchable): phrase-level mappings and scoring
   bonuses tuned to the repository's sample policy corpus, plus bespoke
   summary composers (annual-leave proration and travel reimbursement). These
   rules exist to keep the fixed question set reproducible; they must be
   revalidated when the corpus changes.

Set ``KNOWLEDGEOPS_DISABLE_CORPUS_ADAPTATIONS=1`` to disable the adaptation
layer for ablation comparisons:

    python scripts/run_ablation.py
"""

from __future__ import annotations

import os

DISABLE_ENV_VAR = "KNOWLEDGEOPS_DISABLE_CORPUS_ADAPTATIONS"

# Human-readable inventory of every corpus-specific adaptation. Kept in one
# place so maintainers and reviewers can audit the corpus-specific layer.
ADAPTATIONS = [
    "年假/年休假 同义加分",
    "新员工/新入职 词形加分",
    "上海+住宿 → 一线城市 短语映射",
    "采购+报价/供应商/万元 → 采购门槛 短语映射",
    "钓鱼 → IT-SERVICE/保留截图 短语映射",
    "被盗/遗失 → 15分钟/P1 短语映射",
    "发票+抬头 → 发票抬头（XQ-FIN-002 减分）",
    "访客+登记 → 外部访客登记 短语映射",
    "差旅费 → 差旅/出差/返程 词形映射",
    "打车 → 出租车/网约车 词形映射",
    "发票问题优先发票制度文档",
    "差旅问题优先差旅制度文档",
    "新员工年假折算句加分",
    "申请类意图加分（选择假别/查询余额/申请流程）",
    "发票抬头答案上限 2 句",
    "新员工年假阶梯摘要器",
    "差旅报销流程摘要器",
    "高铁/动车/动车组 同义映射",
    "长途交通工具等级摘要器（飞机/动车组/轮船）",
]


def corpus_adaptations_enabled() -> bool:
    """Whether the corpus-specific adaptation layer is active."""
    return os.getenv(DISABLE_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}
