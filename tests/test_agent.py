from app.agent import (
    INSUFFICIENT_ANSWER,
    has_sufficient_evidence,
    KnowledgeAssistant,
    _best_evidence_document,
    _source_results_for_question,
    answer_document_catalog,
    contextualize_question,
    compose_local_answer,
    is_catalog_question,
    is_project_meta_question,
)
from app.ingestion import KnowledgeDocument, TextChunk
from app.rag import SearchResult


def _travel_result() -> SearchResult:
    chunk = TextChunk(
        "travel-0",
        "travel",
        "差旅报销制度",
        "财务",
        "市内交通",
        "travel.md",
        "出差期间因公务产生的出租车费用可以报销，但需要提供发票并注明出发地和目的地。",
        0,
    )
    return SearchResult(chunk, 1.0, "hybrid", 1)


def test_catalog_question_is_answered_without_llm() -> None:
    assert is_catalog_question("知识库里有哪些制度？")

    answer = answer_document_catalog(["考勤制度", "差旅报销制度"])

    assert "2 份" in answer
    assert "考勤制度" in answer


def test_project_meta_question_does_not_enter_policy_retrieval() -> None:
    assert is_project_meta_question("刚刚讲解项目")
    assistant = KnowledgeAssistant(FakeRetriever(), [], llm_provider="ollama")

    response = assistant.ask("请介绍项目是做什么的")

    assert response["debug"]["route"] == "project_meta"
    assert "VPN" not in response["answer"]
    assert response["sources"] == []


def test_local_answer_only_uses_retrieved_policy_facts() -> None:
    answer = compose_local_answer("出差打车可以报销吗？", [_travel_result()])

    assert "可以报销" in answer
    assert "提供发票" in answer
    assert "100元" not in answer


def test_local_answer_handles_travel_reimbursement_process_wording() -> None:
    result = SearchResult(
        TextChunk(
            "travel-process-0",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "办理流程",
            "04_travel_reimbursement.pdf",
            "员工应在返程之日起 10 个工作日内提交报销，材料包括已批准出差申请、行程单、发票和支付记录。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("差旅费怎么报销？", [result])

    assert "10 个工作日内提交报销" in answer


def test_high_speed_rail_question_maps_to_train_group_reimbursement_rule() -> None:
    result = SearchResult(
        TextChunk(
            "travel-transport-0",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "飞机、火车及长途交通",
            "04_travel_reimbursement.pdf",
            "员工级和专业及主管级原则上乘坐飞机经济舱、火车二等座、动车组二等座和轮船三等舱；"
            "经理及以上可乘坐飞机经济舱、火车一等座、动车组一等座和轮船二等舱。"
            "机票、火车票和船票应通过公司指定渠道预订，凭有效票据按差旅报销流程提交。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("坐高铁二等座的报销流程是什么", [result])

    assert answer.startswith("交通费用报销：")
    assert "高铁按制度对应动车组" in answer
    assert "员工级、专业及主管级原则上可报销动车组二等座" in answer
    assert "公司指定渠道预订" in answer
    assert "通讯补贴" not in answer


def test_air_and_ship_questions_use_the_same_long_distance_transport_rule() -> None:
    result = SearchResult(
        TextChunk(
            "travel-transport-1",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "飞机、火车及长途交通",
            "04_travel_reimbursement.pdf",
            "员工级和专业及主管级原则上乘坐飞机经济舱、火车二等座、动车组二等座和轮船三等舱；"
            "经理及以上可乘坐飞机经济舱、火车一等座、动车组一等座和轮船二等舱。"
            "机票、火车票和船票应通过公司指定渠道预订，凭有效票据按差旅报销流程提交。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    assert "飞机经济舱" in compose_local_answer("飞机经济舱怎么报销", [result])
    assert "轮船三等舱" in compose_local_answer("轮船三等舱怎么报销", [result])


def test_distance_question_returns_air_eligibility_rule() -> None:
    result = SearchResult(
        TextChunk(
            "travel-distance-0",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "飞机、火车及长途交通",
            "04_travel_reimbursement.pdf",
            "单程距离超过800公里或预计连续乘车超过6小时的，可选择飞机经济舱；"
            "低于上述条件的，优先选择高铁或动车。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("超过800公里可以坐飞机吗", [result])

    assert "超过800公里" in answer
    assert "可选择飞机经济舱" in answer
    assert "高铁或动车" in answer


def test_travel_material_question_excludes_transport_exception_rule() -> None:
    result = SearchResult(
        TextChunk(
            "travel-materials-0",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "票据与附件要求",
            "04_travel_reimbursement.pdf",
            "报销单须附出差申请单或外出审批记录、完整行程、交通票据、住宿发票、费用明细及支付凭证。"
            "因公司安排、公共交通停运或不可抗力产生的退改签费用，凭证明材料据实报销。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("差旅报销需要哪些材料", [result])

    assert "出差申请单" in answer
    assert "费用明细" in answer
    assert "公共交通停运" not in answer


def test_personal_leave_application_excludes_annual_leave_approval_rule() -> None:
    result = SearchResult(
        TextChunk(
            "personal-leave-0",
            "leave",
            "员工管理制度汇编",
            "人力资源",
            "员工假期",
            "handbook.pdf",
            "事假员工因处理个人事务需要占用工作时间的，可以申请无薪事假。"
            "事假原则上应提前1个工作日提交申请，说明事由、起止时间、工作交接人和联系方式。"
            "员工请年假须有书面申请，事先获得部门经理或直属总监批准。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("事假怎么申请", [result])

    assert "无薪事假" in answer
    assert "提前1个工作日" in answer
    assert "请年假" not in answer


def test_onboarding_summary_keeps_process_order() -> None:
    result = SearchResult(
        TextChunk(
            "onboarding-order-0",
            "onboarding",
            "入职与试用期管理制度",
            "人力资源",
            "新员工入职办理流程",
            "01_onboarding_probation.pdf",
            "（1）录用确认。招聘专员发送录用通知并同步人力资源部。"
            "（2）报到材料。员工提交身份证件、学历证明等材料。"
            "（3）入职登记。员工完成信息登记并签订劳动合同。"
            "（4）入职培训。员工完成制度与信息安全培训。",
            9,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工入职需要办理哪些手续", [result])

    assert answer.index("录用确认") < answer.index("报到材料")
    assert answer.index("报到材料") < answer.index("劳动合同")


def test_contextualize_question_only_inherits_history_for_follow_up_wording() -> None:
    history = [{"role": "user", "content": "新员工有几天年假？"}]

    assert contextualize_question("那怎么申请？", history).startswith("上一问：")
    assert contextualize_question("发票怎么弄", history) == "发票怎么弄"


def test_contextualize_question_inherits_employment_time_qualifier() -> None:
    history = [{"role": "user", "content": "新员工年假有几天？"}]

    contextualized = contextualize_question("刚入职2个月", history)

    assert "新员工年假有几天？" in contextualized
    assert contextualized.endswith("当前追问：刚入职2个月")


def test_contextualize_question_inherits_bare_duration_qualifier() -> None:
    history = [{"role": "user", "content": "新员工年假有几天？"}]

    contextualized = contextualize_question("2个月", history)

    assert "新员工年假有几天？" in contextualized
    assert contextualized.endswith("当前追问：2个月")


def test_annual_leave_follow_up_uses_latest_application_intent() -> None:
    result = SearchResult(
        TextChunk(
            "annual-follow-up-0",
            "annual-leave",
            "请假与年休假管理制度",
            "人力资源",
            "办理流程",
            "03_leave_annual_leave.pdf",
            "年休假按经人力资源部核验的累计社会工作年限确定。\n"
            "1. 员工查询余额并选择假别、日期和工作代理。\n"
            "2. 系统按天数发送直属经理、部门负责人或人力资源部。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    contextualized = contextualize_question(
        "那怎么申请？", [{"role": "user", "content": "新员工有几天年假？"}]
    )
    answer = compose_local_answer(contextualized, [result])

    assert "查询余额并选择假别" in answer
    assert "年假不是统一固定天数" not in answer


def test_annual_leave_duration_question_uses_matching_approval_rule() -> None:
    result = SearchResult(
        TextChunk(
            "annual-process-0",
            "annual-leave",
            "请假与年休假管理制度",
            "人力资源",
            "申请时限与审批",
            "03_leave_annual_leave.pdf",
            "1. 连续请假不超过 3 个工作日的，应至少提前 2 个工作日申请，由直属经理审批。\n"
            "2. 连续请假超过 3 个工作日的，应至少提前 5 个工作日申请，由直属经理和部门负责人审批。\n"
            "3. 单次连续年休假超过 5 个工作日的，人力资源部须复核余额与业务安排。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("连续请年休假 4 个工作日", [result])

    assert "至少提前 5 个工作日" in answer
    assert "部门负责人审批" in answer
    assert "超过 5 个工作日" not in answer


def test_new_hire_material_deadline_uses_onboarding_policy() -> None:
    result = SearchResult(
        TextChunk(
            "onboarding-material-0",
            "onboarding",
            "入职与试用期管理制度",
            "人力资源",
            "入职规则",
            "01_onboarding_probation.pdf",
            "因客观原因无法当日补齐的非关键材料，应在入职后 3 个工作日内补交；"
            "身份证明、合法用工资格等关键材料未核验前，不得安排接触生产数据。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工非关键材料最晚多久补交", [result])

    assert "3 个工作日内补交" in answer
    assert "年休假" not in answer


def test_overseas_vpn_question_returns_advance_approval_deadline() -> None:
    result = SearchResult(
        TextChunk(
            "vpn-overseas-0",
            "vpn",
            "远程办公与 VPN 使用制度",
            "IT 服务",
            "具体规则",
            "09_vpn_remote_work.pdf",
            "在境外或高风险地区远程办公，应至少提前 3 个工作日申请，"
            "并取得直属负责人和信息安全办公室批准。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("境外使用 VPN 需要提前多久申请", [result])

    assert "提前 3 个工作日申请" in answer
    assert "信息安全办公室批准" in answer


def test_lodging_question_does_not_append_reimbursement_materials() -> None:
    result = SearchResult(
        TextChunk(
            "travel-lodging-0", "travel", "国内差旅与报销制度", "财务", "交通与住宿标准",
            "04_travel_reimbursement.pdf",
            "一线城市住宿上限为每人每晚600元，其他城市为每人每晚450元。\n"
            "材料至少包括已批准的出差申请、行程单或车票、住宿发票。",
            0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("上海出差住宿标准是多少？", [result])

    assert "600元" in answer
    assert "材料至少包括" not in answer


def test_invoice_header_question_does_not_append_exception_process() -> None:
    result = SearchResult(
        TextChunk(
            "invoice-header-0", "invoice", "发票与日常费用报销制度", "财务", "有效凭证",
            "05_invoice_expense.pdf",
            "发票抬头应为系统展示的“鲲坤科技”样例主体全称。\n"
            "因销售方系统故障未能及时开票的，员工应在10个工作日时限内先提交费用说明。",
            0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("发票抬头怎么写？", [result])

    assert "鲲坤科技" in answer
    assert "销售方系统故障" not in answer


def test_marriage_leave_question_returns_duration_and_deadline() -> None:
    result = SearchResult(
        TextChunk(
            "marriage-leave-0",
            "leave",
            "请假与休假管理制度",
            "人力资源",
            "婚假",
            "03_leave_annual_leave.pdf",
            "婚假基础标准为 3 个工作日。原则上应在领取结婚证后 6 个月内一次性休完。"
            "申请时提交结婚证或登记证明，并按休假流程审批。",
            28,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("婚假应该在多久内休完", [result])

    assert "3 个工作日" in answer
    assert "6 个月内" in answer
    assert "依据不足" not in answer


def test_onboarding_question_does_not_mix_retirement_procedure() -> None:
    result = SearchResult(
        TextChunk(
            "onboarding-flow-0",
            "onboarding",
            "入职与试用期管理制度",
            "人力资源",
            "新员工入职办理流程",
            "01_onboarding_probation.pdf",
            "新员工入职办理流程包括录用确认、提交材料、报到与身份核验、签订劳动合同、"
            "制度与安全培训、岗位交接和试用期目标、试用期跟进与转正。"
            "员工到社保办理退休手续，员工退休手续办理后停止缴纳社会保险。",
            9,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工入职需要办理哪些手续", [result])

    assert "录用确认" in answer
    assert "签订劳动合同" in answer
    assert "退休" not in answer


def test_invoice_header_question_uses_full_company_identity_without_attendee_list() -> None:
    result = SearchResult(
        TextChunk(
            "invoice-header-fused-0",
            "invoice",
            "发票与日常费用报销制度",
            "财务",
            "有效凭证",
            "05_invoice_expense.pdf",
            "发票抬头填写公司全称和统一社会信用代码，发票内容应与实际业务一致。"
            "培训或会议费用另需提交发票和参与人员名单。",
            16,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("发票抬头怎么写", [result])

    assert "公司全称" in answer
    assert "统一社会信用代码" in answer
    assert "参与人员名单" not in answer


def test_travel_reimbursement_deadline_does_not_mix_advance_trip_application() -> None:
    result = SearchResult(
        TextChunk(
            "travel-deadline-0",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "报销时限与材料",
            "04_travel_reimbursement.pdf",
            "出差结束或费用发生后 10 个工作日内提交报销。"
            "跨月费用原则上在次月 5 日前完成提交，逾期需说明原因并经部门负责人批准。"
            "出差前至少 1 个工作日提交出差申请。",
            19,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("差旅费用多久提交", [result])

    assert "10 个工作日内提交报销" in answer
    assert "次月 5 日前" in answer
    assert "出差前" not in answer


def test_new_employee_annual_leave_answer_explains_tiers_and_proration() -> None:
    result = SearchResult(
        TextChunk(
            "annual-leave-0",
            "annual-leave",
            "请假与年休假管理制度",
            "人力资源",
            "年休假额度",
            "03_leave_annual_leave.pdf",
            "年休假按经人力资源部核验的累计社会工作年限确定。\n"
            "1. 累计工作已满 1 年不满 10 年的，年度年休假为 5 天。\n"
            "2. 累计工作已满 10 年不满 20 年的，年度年休假为 10 天。\n"
            "3. 累计工作已满 20 年的，年度年休假为 15 天。\n"
            "4. 新入职员工的可用额度由人力资源部按适用规则折算，员工应以平台核定余额为申请上限。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工有几天年假？", [result])

    assert answer.startswith("新员工的年假不是统一固定天数")
    assert "累计工作满 1 年不满 10 年：5 天" in answer
    assert "累计工作满 10 年不满 20 年：10 天" in answer
    assert "累计工作满 20 年：15 天" in answer
    assert "入职当年" in answer
    assert "平台核定余额" in answer


def test_new_employee_annual_leave_answer_handles_fused_handbook_wording() -> None:
    result = SearchResult(
        TextChunk(
            "annual-leave-fused-0",
            "annual-leave-fused",
            "鲲坤科技员工管理制度汇编",
            "人力资源",
            "带薪年休假",
            "鲲坤科技-员工管理制度汇编-2025融合版.pdf",
            "员工自入职后的第二个日历年起可享受的年假天数如下：实际工作1~6年（含）之间，可享有5天年休假；工作满6年后，每服务一年增加一天，至可享有9天年休假止。"
            "实际工作10~16年（含）之间，可享有10天年休假；实际工作满20年，可享有15天年休假。"
            "新入职员工的年休假可用额度由人力资源部按适用规则折算，并以员工服务平台核定余额为准。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工年假有几天？", [result])

    assert answer.startswith("新员工的年假不是统一固定天数")
    assert "入职后的第二个日历年起" in answer
    assert "实际工作 1~6 年（含）：5 天" in answer
    assert "实际工作 10~16 年（含）：10 天" in answer
    assert "实际工作满 20 年：15 天" in answer
    assert "服务平台核定余额" in answer
    assert "应当向人力资源部报案" not in answer


def test_fused_handbook_annual_leave_does_not_expose_internal_missing_rule_wording() -> None:
    result = SearchResult(
        TextChunk(
            "annual-leave-fused-incomplete",
            "annual-leave-fused-incomplete",
            "鲲坤科技员工管理制度汇编",
            "人力资源",
            "带薪年休假",
            "handbook.pdf",
            "员工自入职后的第二个日历年起可享受的年假天数如下：实际工作1~6年（含）之间，可享有5天年休假；"
            "实际工作10~16年（含）之间，可享有10天年休假；实际工作满20年，可享有15天年休假。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工年假有几天？", [result])

    assert "入职第一年的具体额度需由人力资源部按适用规则核定" in answer
    assert "文档未规定" not in answer


def test_travel_reimbursement_prefers_dedicated_travel_policy() -> None:
    travel = SearchResult(
        TextChunk(
            "travel-process",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "办理流程",
            "04_travel_reimbursement.pdf",
            "员工应在返程之日起 10 个工作日内提交报销，材料包括已批准出差申请、行程单、发票和支付记录。",
            0,
        ),
        0.8,
        "hybrid",
        2,
    )
    invoice = SearchResult(
        TextChunk(
            "invoice-cross-reference",
            "invoice",
            "发票与日常费用报销制度",
            "财务",
            "申请时限与审批",
            "05_invoice_expense.pdf",
            "差旅费用按返程之日起 10 个工作日内提交。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    assert _best_evidence_document("差旅费报销怎么弄", [invoice, travel]) == "travel"


def test_travel_reimbursement_answer_is_summarized_in_process_order() -> None:
    result = SearchResult(
        TextChunk(
            "travel-process-summary",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "办理流程",
            "04_travel_reimbursement.pdf",
            "3. 返程后 10 个工作日内提交报销及凭证。\n"
            "5. 审批通过后进入公司付款批次。\n"
            "1. 员工事前提交出差申请并完成审批。\n"
            "2. 员工按批准标准预订并完成行程。\n"
            "4. 材料至少包括已批准的出差申请、行程单、发票和支付记录。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer("差旅费报销怎么弄", [result])

    assert answer.startswith("差旅费报销流程：")
    assert answer.index("出差申请") < answer.index("返程后")
    assert answer.index("返程后") < answer.index("材料至少包括")
    assert answer.index("材料至少包括") < answer.index("付款批次")


def test_evidence_prefers_dedicated_invoice_policy_over_cross_reference() -> None:
    invoice = SearchResult(
        TextChunk(
            "invoice-0", "invoice", "发票与日常费用报销制度", "财务", "有效凭证", "invoice.pdf",
            "发票抬头应与系统展示的公司全称一致。", 0,
        ),
        0.9, "hybrid", 1,
    )
    travel = SearchResult(
        TextChunk(
            "travel-0", "travel", "国内差旅与报销制度", "财务", "报销材料", "travel.pdf",
            "发票抬头、日期、金额、重复报销校验和电子票据要求按 XQ-FIN-002 执行。", 0,
        ),
        0.8, "hybrid", 3,
    )

    assert _best_evidence_document("报销发票抬头写什么？", [invoice, travel]) == "invoice"

    answer = compose_local_answer("报销发票抬头写什么？", [invoice, travel])

    assert "公司全称一致" in answer
    assert "XQ-FIN-002" not in answer


def test_evidence_rejects_personal_account_match_without_the_queried_business_object() -> None:
    result = SearchResult(
        TextChunk(
            "account-scope",
            "account",
            "账号、密码与 MFA 管理制度",
            "IT",
            "适用范围",
            "08_account_password.pdf",
            "公司支持员工通过服务台处理个人名下的企业账号，不得共享账号。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    assert _best_evidence_document("公司是否支持个人游戏账号的密码找回？", [result]) is None


def test_evidence_rejects_repair_match_without_the_queried_business_object() -> None:
    result = SearchResult(
        TextChunk(
            "device-repair",
            "device",
            "IT 设备借用与归还制度",
            "IT",
            "维修流程",
            "12_equipment_checkout.docx",
            "确因维修替换需要临时使用无资产编号设备时，须提交例外申请，"
            "经系统负责人及信息安全办公室批准。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    assert _best_evidence_document("如何申请办公室咖啡机维修？", [result]) is None


def test_evidence_accepts_a_single_specific_account_topic() -> None:
    result = SearchResult(
        TextChunk(
            "account-application",
            "account",
            "账号、密码与 MFA 管理制度",
            "IT",
            "账号申请",
            "08_account_password.pdf",
            "企业账号申请应通过 IT-SERVICE 提交工单。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    assert _best_evidence_document("账号怎么申请？", [result]) == "account"


def test_local_answer_hides_current_policy_cross_reference_code() -> None:
    invoice = SearchResult(
        TextChunk(
            "invoice-kk", "invoice", "发票与日常费用报销制度", "财务", "有效凭证", "invoice.pdf",
            "发票抬头应与系统展示的公司全称一致。", 0,
        ),
        0.9, "hybrid", 1,
    )
    travel = SearchResult(
        TextChunk(
            "travel-kk", "travel", "国内差旅与报销制度", "财务", "报销材料", "travel.pdf",
            "发票抬头、日期、金额、重复报销校验和电子票据要求按 KK-FIN-002 执行。", 0,
        ),
        0.8, "hybrid", 3,
    )

    answer = compose_local_answer("报销发票抬头写什么？", [invoice, travel])

    assert "公司全称一致" in answer
    assert "KK-FIN-002" not in answer


def test_local_answer_hides_cross_reference_when_only_source_is_returned() -> None:
    travel = SearchResult(
        TextChunk(
            "travel-kk-only", "travel", "国内差旅与报销制度", "财务", "报销材料", "travel.pdf",
            "发票抬头、日期、金额、重复报销校验和电子票据要求按 KK-FIN-002 执行。", 0,
        ),
        0.8, "hybrid", 1,
    )

    answer = compose_local_answer("报销发票抬头写什么？", [travel])

    assert "KK-FIN-002" not in answer


def test_local_answer_skips_pdf_metadata_and_prioritizes_specific_rule() -> None:
    result = SearchResult(
        TextChunk(
            "invoice-0", "invoice", "发票与日常费用报销制度", "财务", "第 1 页", "invoice.pdf",
            "文档名称：发票与日常费用报销制度\n"
            "1. 同一发票只能报销一次。\n"
            "2. 发票抬头应与系统展示的公司全称一致。",
            0,
        ),
        1.0, "hybrid", 1,
    )

    answer = compose_local_answer("报销发票抬头写什么？", [result])

    assert "文档名称" not in answer
    assert answer.index("发票抬头") < answer.index("同一发票")


def test_local_answer_refuses_when_context_is_empty() -> None:
    answer = compose_local_answer("公司食堂几点开门？", [])

    assert "依据不足" in answer


def test_local_answer_refuses_when_retrieved_context_is_irrelevant() -> None:
    irrelevant = SearchResult(
        TextChunk(
            "expense-0",
            "expense",
            "费用报销制度",
            "财务",
            "不予报销情形",
            "expense.md",
            "与公司业务无关的个人消费不得报销。",
            0,
        ),
        0.5,
        "hybrid",
        1,
    )

    answer = compose_local_answer("公司食堂几点开门？", [irrelevant])

    assert "依据不足" in answer


def test_local_answer_refuses_when_only_a_generic_term_overlaps() -> None:
    vpn_policy = SearchResult(
        TextChunk(
            "vpn-0",
            "vpn",
            "VPN 使用制度",
            "IT 服务",
            "故障处理",
            "vpn.md",
            "系统返回 VPN-403 的准确含义是远程访问被策略拒绝，请联系 IT-SERVICE。",
            0,
        ),
        0.5,
        "hybrid",
        1,
    )

    answer = compose_local_answer("公司的股票代码是多少？", [vpn_policy])

    assert "依据不足" in answer


def test_local_answer_refuses_when_a_question_has_only_function_tokens() -> None:
    remote_work_policy = SearchResult(
        TextChunk(
            "remote-0",
            "remote",
            "远程办公制度",
            "IT 服务",
            "异地办公",
            "remote.md",
            "员工在异地办公时应使用公司批准的 VPN 客户端。",
            0,
        ),
        0.5,
        "hybrid",
        1,
    )

    answer = compose_local_answer("公司在哪？", [remote_work_policy])

    assert "依据不足" in answer


def test_local_answer_refuses_pet_medical_question_with_generic_reimbursement_hit() -> None:
    answer = compose_local_answer("宠物医疗是否报销？", [_travel_result()])

    assert "依据不足" in answer


def test_local_answer_refuses_pet_medical_question_with_fee_overlap() -> None:
    answer = compose_local_answer("宠物医疗费用可以报销吗？", [_travel_result()])

    assert "依据不足" in answer


def test_local_answer_does_not_mix_lower_ranked_policy() -> None:
    unrelated = SearchResult(
        TextChunk(
            "leave-0",
            "leave",
            "请假制度",
            "HR",
            "补交申请",
            "leave.md",
            "员工恢复工作后 1 个工作日内补交请假申请。",
            0,
        ),
        0.5,
        "hybrid",
        2,
    )

    answer = compose_local_answer("出差打车可以报销吗？", [_travel_result(), unrelated])

    assert "请假申请" not in answer


def test_local_answer_finds_evidence_in_lower_ranked_relevant_document() -> None:
    leave = SearchResult(
        TextChunk(
            "leave-0",
            "leave",
            "请假与年休假管理制度",
            "HR",
            "年休假额度",
            "leave.pdf",
            "新入职员工的年休假可用额度，由人力资源部按适用规则折算并在平台展示。",
            0,
        ),
        0.4,
        "hybrid",
        5,
    )
    unrelated = SearchResult(
        TextChunk(
            "attendance-0",
            "attendance",
            "考勤制度",
            "HR",
            "异常处理",
            "attendance.docx",
            "员工忘记打卡应在三个工作日内提交申请。",
            0,
        ),
        0.5,
        "hybrid",
        1,
    )

    answer = compose_local_answer("新员工有几天年假？", [unrelated, leave])

    assert "折算" in answer
    assert "平台展示" in answer


def test_new_hire_leave_answer_leads_with_prorated_balance_rule() -> None:
    leave = SearchResult(
        TextChunk(
            "leave-0",
            "leave",
            "请假与年休假管理制度",
            "HR",
            "年休假额度",
            "leave.pdf",
            "1. 累计工作已满1年不满10年的，年度年休假为5天。\n"
            "2. 新入职或当年离职员工的可用额度，由人力资源部按适用规则折算并在员工服务平台核定余额。",
            0,
        ),
        0.4,
        "hybrid",
        1,
    )

    answer = compose_local_answer("入职第一年年假几天？", [leave])

    assert "按适用规则折算" in answer
    assert "平台核定余额" in answer
    assert answer.index("按适用规则折算") < answer.index("年度年休假为5天")


def test_procurement_amount_question_returns_supplier_quote_rule() -> None:
    result = SearchResult(
        TextChunk(
            "procurement-0", "procurement", "采购申请与审批制度", "采购", "采购门槛", "06_procurement_approval.pdf",
            "预计总额达到5,000元但低于50,000元的，须事前提交采购申请，并取得至少2家可比供应商的有效报价。", 0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("采购4万元要几家报价？", [result])

    assert "至少2家" in answer
    assert "有效报价" in answer


def test_phishing_question_returns_report_and_evidence_rule() -> None:
    result = SearchResult(
        TextChunk(
            "security-0", "security", "信息安全与事件报告制度", "信息安全", "事件报告", "11_information_security.pdf",
            "疑似钓鱼点击应通过 IT-SERVICE 报告，发现人应保留截图、邮件原文、时间和资产编号等证据。", 0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("点击钓鱼链接后怎么处理？", [result])

    assert "IT-SERVICE" in answer
    assert "保留截图" in answer


def test_stolen_laptop_question_returns_p1_reporting_deadline() -> None:
    result = SearchResult(
        TextChunk(
            "equipment-0", "equipment", "IT设备借用与归还制度", "行政服务", "具体规则", "12_equipment_checkout.docx",
            "设备遗失或被盗按P1处理，发现后15分钟内通过电话和 IT-SERVICE 报告。", 0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("公司笔记本被盗多久报告？", [result])

    assert "15分钟内" in answer
    assert "P1" in answer


def test_shanghai_lodging_question_returns_city_cap() -> None:
    result = SearchResult(
        TextChunk(
            "travel-0", "travel", "国内差旅与报销制度", "财务", "交通与住宿标准", "04_travel_reimbursement.pdf",
            "一线城市住宿上限为每人每晚600元，其他城市为每人每晚450元。", 0,
        ), 1.0, "hybrid", 1,
    )

    answer = compose_local_answer("上海出差住宿标准是多少？", [result])

    assert "一线城市" in answer
    assert "600元" in answer


def test_local_answer_removes_near_duplicate_pdf_sentence_fragments() -> None:
    complete = SearchResult(
        TextChunk(
            "vpn-0", "vpn", "远程办公与 VPN 使用制度", "IT 服务", "具体规则", "vpn.pdf",
            "VPN-403 表示远程访问被策略拒绝，常见原因包括权限未获批、MFA 未完成、终端不合规或账号被锁定。", 0,
        ),
        1.0, "hybrid", 1,
    )
    truncated = SearchResult(
        TextChunk(
            "vpn-1", "vpn", "远程办公与 VPN 使用制度", "IT 服务", "第 1 页", "vpn.pdf",
            "VPN-403 表示远程访问被策略拒绝，常见原因包括权限未获批、MFA 未完成、终端不合规或账号被锁定或访问", 0,
        ),
        0.9, "hybrid", 2,
    )

    answer = compose_local_answer("VPN-403 是什么意思？", [complete, truncated])

    assert answer.count("VPN-403 表示远程访问被策略拒绝") == 1


def test_local_answer_removes_prefix_expansion_of_same_policy_fact() -> None:
    short = SearchResult(
        TextChunk(
            "travel-0", "travel", "国内差旅与报销制度", "财务", "住宿标准", "travel.pdf",
            "一线城市住宿上限为每人每晚 600 元。", 0,
        ),
        1.0, "hybrid", 1,
    )
    expanded = SearchResult(
        TextChunk(
            "travel-1", "travel", "国内差旅与报销制度", "财务", "第 2 页", "travel.pdf",
            "一线城市住宿上限为每人每晚 600 元，其他城市为每人每晚 450 元。", 0,
        ),
        0.9, "hybrid", 2,
    )

    answer = compose_local_answer("上海出差住宿标准是多少？", [short, expanded])

    assert answer.count("一线城市住宿上限为每人每晚") == 1


def test_follow_up_question_reuses_previous_user_context() -> None:
    leave = SearchResult(
        TextChunk(
            "leave-0", "leave", "请假与年休假管理制度", "HR", "办理流程", "leave.pdf",
            "年休假申请：员工应在员工服务平台查询余额，选择假别和日期后提交申请。", 0,
        ),
        1.0, "hybrid", 1,
    )

    class TrackingRetriever:
        def __init__(self) -> None:
            self.query = ""

        def retrieve(self, query: str, top_k: int = 5):
            self.query = query
            return {
                "final_results": [leave],
                "vector_results": [],
                "bm25_results": [],
                "reranker_error": "",
                "vector_backend": "numpy",
            }

    retriever = TrackingRetriever()
    assistant = KnowledgeAssistant(
        retriever,
        [KnowledgeDocument("leave", "请假与年休假管理制度", "HR", "leave.pdf", "制度正文", ["办理流程"])],
        llm_provider="ollama",
    )

    response = assistant.ask("那怎么申请？", history=[{"role": "user", "content": "新员工有几天年假？"}])

    assert "新员工有几天年假？" in retriever.query
    assert "暂时无法生成服务答复" in response["answer"]


def test_follow_up_application_question_prioritizes_application_steps() -> None:
    quota = SearchResult(
        TextChunk(
            "leave-quota",
            "leave",
            "请假与年休假管理制度",
            "HR",
            "年休假额度",
            "leave.pdf",
            "1. 新员工入职第一年不适用统一固定天数，年休假由人力资源部按累计工作年限和当年在岗时间折算。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )
    application = SearchResult(
        TextChunk(
            "leave-flow",
            "leave",
            "请假与年休假管理制度",
            "HR",
            "办理流程",
            "leave.pdf",
            "员工查询余额并选择假别、日期和工作代理后提交申请，系统发送直属经理审批。",
            0,
        ),
        0.8,
        "hybrid",
        3,
    )
    incomplete_fragment = SearchResult(
        TextChunk(
            "leave-fragment",
            "leave",
            "请假与年休假管理制度",
            "HR",
            "年休假额度",
            "leave.pdf",
            "年休假最小申请单位为",
            0,
        ),
        0.9,
        "hybrid",
        2,
    )

    class FollowUpRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return {
                "final_results": [quota, incomplete_fragment, application],
                "vector_results": [],
                "bm25_results": [],
                "reranker_error": "",
                "vector_backend": "numpy",
            }

    assistant = KnowledgeAssistant(
        FollowUpRetriever(),
        [KnowledgeDocument("leave", "请假与年休假管理制度", "HR", "leave.pdf", "制度正文", [])],
        llm_provider="ollama",
    )

    response = assistant.ask(
        "那怎么申请？", history=[{"role": "user", "content": "新员工有几天年假？"}]
    )

    assert "暂时无法生成服务答复" in response["answer"]
    assert "最小申请单位为" not in response["answer"]
    assert "五、办理流程" not in response["answer"]


def test_contextual_travel_invoice_follow_up_is_summarized_instead_of_mixed_fragments() -> None:
    result = SearchResult(
        TextChunk(
            "travel-process",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "报销时限与材料",
            "travel.pdf",
            "1. 员工应在返程之日起 10 个工作日内提交报销。\n"
            "2. 材料至少包括已批准的出差申请、行程单、发票和支付记录。\n"
            "3. 发票抬头、税号、金额、重复报销检查和电子票据要求按财务制度执行。\n"
            "4. 直属经理、预算负责人和财务部依次审核。\n"
            "5. 市内交通按实际、必要、合理原则凭有效凭证报销。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )

    answer = compose_local_answer(
        "上一问：差旅费报销怎么弄？\n当前追问：那发票怎么弄？",
        [result],
    )

    assert answer.startswith("差旅费报销材料：")
    assert "提交报销" in answer
    assert "发票抬头" in answer
    assert "直属经理" not in answer
    assert "市内交通" not in answer
    assert answer.count("提交报销") == 1


def test_contextual_invoice_follow_up_prefers_invoice_policy_when_available() -> None:
    invoice = SearchResult(
        TextChunk(
            "invoice-follow-up",
            "invoice",
            "发票与日常费用报销制度",
            "财务",
            "有效凭证要求",
            "05_invoice_expense.pdf",
            "电子发票须上传原始电子文件，发票抬头应为系统展示的公司全称。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )
    travel = SearchResult(
        TextChunk(
            "travel-follow-up",
            "travel",
            "国内差旅与报销制度",
            "财务",
            "报销材料",
            "04_travel_reimbursement.pdf",
            "差旅材料包括行程单、住宿发票和支付记录，发票要求按对应制度执行。",
            0,
        ),
        0.9,
        "hybrid",
        2,
    )
    question = "上一问：差旅费报销怎么弄？\n当前追问：那发票怎么弄？"

    assert _best_evidence_document(question, [invoice, travel]) == "invoice"


def test_contextual_invoice_follow_up_summarizes_actionable_invoice_rules() -> None:
    invoice = SearchResult(
        TextChunk(
            "invoice-rules",
            "invoice",
            "发票与日常费用报销制度",
            "财务",
            "有效凭证要求",
            "05_invoice_expense.pdf",
            "1. 发票抬头应为系统展示的公司全称。\n"
            "2. 发票的销售方、项目、日期、金额应与真实业务和支付记录一致。\n"
            "3. 电子发票须上传原始电子文件，截图不替代原始文件。\n"
            "4. 同一发票只能报销一次。\n"
            "5. 差旅费用按返程之日起 10 个工作日内提交。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )
    question = "上一问：差旅费报销怎么弄？\n当前追问：那发票怎么弄？"

    answer = compose_local_answer(question, [invoice])

    assert answer.startswith("差旅报销发票要求：")
    assert "公司全称" in answer
    assert "真实业务和支付记录一致" in answer
    assert "上传原始电子文件" in answer
    assert "只能报销一次" in answer
    assert "返程之日起 10 个工作日内" in answer


def test_insufficient_answer_uses_conversational_next_step() -> None:
    answer = compose_local_answer("公司是否提供宠物医疗报销？", [])

    assert "依据不足" in answer
    assert "补充具体场景" not in answer
    assert "联系对应责任部门" in answer


def test_meta_question_never_depends_on_retrieval():
    class UnavailableRetriever:
        def retrieve(self, *args, **kwargs):
            raise ConnectionError("unavailable")

    assistant = KnowledgeAssistant(UnavailableRetriever(), [], llm_provider="ollama")
    assert assistant.ask("刚刚讲解项目")["debug"]["route"] == "project_meta"


def test_no_evidence_refuses_even_without_model():
    assistant = KnowledgeAssistant(IrrelevantRetriever(), [], llm_provider="ollama")
    result = assistant.ask("公司是否提供宠物医疗报销？")
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["sources"] == []
    assert result["debug"]["llm_error"] == ""


def test_matching_repair_and_reimbursement_actions_do_not_authorize_unseen_objects():
    results = [SearchResult(TextChunk(
        "vehicle-1", "vehicle", "费用制度", "财务", "车辆维修", "vehicle.md",
        "办公室员工因私车公用发生的车辆保险、维修、保养费不得报销，需联系行政部门。", 0,
    ), 1.0, "hybrid", 1)]
    for subject in ("咖啡机", "投影仪", "净水器"):
        assert not has_sufficient_evidence(f"办公室{subject}坏了，能找行政维修并报销吗？", results)


def test_specific_object_can_be_answered_when_explicitly_supported():
    results = [SearchResult(TextChunk(
        "device-1", "device", "设备制度", "行政", "咖啡机", "device.md",
        "办公室咖啡机维修须向行政部门提交申请，不得个人报销维修费。", 0,
    ), 1.0, "hybrid", 1)]
    assert has_sufficient_evidence("办公室咖啡机坏了，能找行政维修并报销吗？", results)


def test_contract_drafting_does_not_use_supplier_admission_as_its_evidence():
    results = [SearchResult(TextChunk(
        "supplier-1", "supplier", "供应商制度", "采购", "准入", "supplier.md",
        "采购供应商前需要调查供应商的资信和履约能力，驳船供应商应持水路运输许可证。", 0,
    ), 1.0, "hybrid", 1)]
    assert not has_sufficient_evidence("起草采购合同时要先调查供应商哪些方面？", results)


def test_catalog_headings_are_never_treated_as_policy_evidence():
    results = [SearchResult(TextChunk(
        "toc", "contract", "合同制度", "法务", "目录", "contract.md",
        "目录\n第一章 总则\n3\n第二章 合同谈判与起草\n4\n第三章 合同审核\n5", 0,
    ), 1.0, "hybrid", 1)]
    assert not has_sufficient_evidence("起草合同前要调查哪些方面？", results)


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        return {
            "final_results": [_travel_result()],
            "vector_results": [],
            "bm25_results": [],
            "reranker_error": "",
            "vector_backend": "numpy",
        }


class DeepCandidateRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        relevant = _travel_result()
        relevant.rank = 7
        distractors = [
            SearchResult(
                TextChunk(f"noise-{index}", "noise", "无关制度", "其他", "正文", "noise.md", "完全无关的内容。", 0),
                0.1,
                "hybrid",
                index,
            )
            for index in range(1, 7)
        ]
        results = distractors + [relevant]
        return {
            "final_results": results[:top_k],
            "vector_results": [],
            "bm25_results": [],
            "reranker_error": "",
            "vector_backend": "numpy",
        }


def test_assistant_requests_deeper_candidate_pool_for_evidence() -> None:
    documents = [
        KnowledgeDocument("travel", "差旅报销制度", "财务", "travel.md", "制度正文", ["市内交通"])
    ]
    assistant = KnowledgeAssistant(DeepCandidateRetriever(), documents, llm_provider="ollama")

    response = assistant.ask("出差打车可以报销吗？")

    assert "暂时无法生成服务答复" in response["answer"]
    assert response["sources"][0]["title"] == "差旅报销制度"


class IrrelevantRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        result = SearchResult(
            TextChunk(
                "expense-0",
                "expense",
                "费用报销制度",
                "财务",
                "不予报销情形",
                "expense.md",
                "与公司业务无关的个人消费不得报销。",
                0,
            ),
            0.5,
            "hybrid",
            1,
        )
        return {
            "final_results": [result],
            "vector_results": [result],
            "bm25_results": [result],
            "reranker_error": "",
            "vector_backend": "numpy",
        }


class TrackingChatClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return "不应生成这个回答"


class PromptCaptureChatClient:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "人力资源部发起账号开通申请，用人部门准备试用期目标。"


class FailingChatClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("Ollama connection failed")


def test_assistant_returns_structured_sources() -> None:
    documents = [
        KnowledgeDocument("travel", "差旅报销制度", "财务", "travel.md", "制度正文", ["市内交通"])
    ]
    assistant = KnowledgeAssistant(FakeRetriever(), documents, llm_provider="ollama")

    response = assistant.ask("出差打车可以报销吗？")

    assert response["sources"][0]["title"] == "差旅报销制度"
    assert response["sources"][0]["section"] == "市内交通"
    assert response["debug"]["route"] == "retrieval"


def test_llm_prompt_includes_extracted_fact_checklist_for_multi_part_responsibility_question() -> None:
    result = SearchResult(
        TextChunk(
            "onboarding-0",
            "onboarding",
            "员工管理制度汇编",
            "人力资源",
            "新员工入职",
            "handbook.pdf",
            "人力资源部应在员工报到前完成账号开通申请；用人部门负责试用期目标；行政与IT部门负责系统权限的开通。",
            0,
        ),
        1.0,
        "hybrid",
        1,
    )
    client = PromptCaptureChatClient()
    assistant = KnowledgeAssistant(
        FakeRetriever(),
        [],
        llm_provider="ollama",
        llm_client=client,
    )

    answer = assistant._compose_llm_answer(
        "新员工报到前，谁负责发起账号开通申请，试用期目标由谁准备？",
        [result],
    )

    assert "人力资源部发起账号开通申请" in answer
    assert "经校验的唯一回答事实范围" in client.user_prompt
    assert "人力资源部" in client.user_prompt
    assert "用人部门" in client.user_prompt
    assert "申请发起人、审核人和实际执行人" in client.system_prompt


def test_assistant_filters_onboarding_sources_to_onboarding_chunks() -> None:
    onboarding = SearchResult(
        TextChunk(
            "onboarding-source",
            "onboarding",
            "员工管理制度汇编",
            "人力资源",
            "第 10 页",
            "handbook.pdf",
            "（2）报到材料。员工提交身份证件。",
            22,
        ),
        0.9,
        "hybrid",
        2,
    )
    retirement = SearchResult(
        TextChunk(
            "retirement-source",
            "retirement",
            "员工管理制度汇编",
            "人力资源",
            "第 31 页",
            "handbook.pdf",
            "员工到社保办理退休手续后，由社保发放养老金。",
            78,
        ),
        1.0,
        "hybrid",
        1,
    )

    class Retriever:
        def retrieve(self, query: str, top_k: int = 5):
            return {"final_results": [retirement, onboarding], "vector_backend": "numpy"}

    assistant = KnowledgeAssistant(
        Retriever(),
        [KnowledgeDocument("handbook", "员工管理制度汇编", "人力资源", "handbook.pdf", "正文", [])],
        llm_provider="ollama",
    )

    response = assistant.ask("新员工入职需要办理哪些手续")

    assert response["sources"]
    assert all("退休" not in source["content_preview"] for source in response["sources"])
    assert response["sources"][0]["section"] == "第 10 页"


def test_assistant_filters_lodging_sources_to_lodging_standard_chunks() -> None:
    lodging = SearchResult(
        TextChunk("lodging", "travel", "员工管理制度汇编", "财务", "第 18 页", "handbook.pdf", "员工级住宿标准：一线城市每人每天不超过450元。", 43),
        0.8, "hybrid", 2,
    )
    unrelated = SearchResult(
        TextChunk("unrelated", "travel", "员工管理制度汇编", "财务", "第 5 页", "handbook.pdf", "公司提供暑托班服务。", 10),
        1.0, "hybrid", 1,
    )

    class Retriever:
        def retrieve(self, query: str, top_k: int = 5):
            return {"final_results": [unrelated, lodging], "vector_backend": "numpy"}

    assistant = KnowledgeAssistant(
        Retriever(),
        [KnowledgeDocument("handbook", "员工管理制度汇编", "财务", "handbook.pdf", "正文", [])],
        llm_provider="ollama",
    )

    response = assistant.ask("上海出差住宿标准是多少")

    assert response["sources"]
    assert all("暑托班" not in source["content_preview"] for source in response["sources"])


def test_assistant_filters_promotion_sources_to_promotion_review_chunks() -> None:
    promotion = SearchResult(
        TextChunk("promotion", "hr", "员工管理制度汇编", "人力资源", "第 26 页", "handbook.pdf", "晋升与调薪评审：公司结合岗位职责、绩效结果开展晋升评审。", 61),
        0.8, "hybrid", 2,
    )
    unrelated = SearchResult(
        TextChunk("unrelated", "hr", "员工管理制度汇编", "人力资源", "第 37 页", "handbook.pdf", "员工沟通与申诉程序。", 95),
        1.0, "hybrid", 1,
    )

    class Retriever:
        def retrieve(self, query: str, top_k: int = 5):
            return {"final_results": [unrelated, promotion], "vector_backend": "numpy"}

    assistant = KnowledgeAssistant(
        Retriever(),
        [KnowledgeDocument("handbook", "员工管理制度汇编", "人力资源", "handbook.pdf", "正文", [])],
        llm_provider="ollama",
    )

    response = assistant.ask("晋升和调薪怎么评审")

    assert response["sources"]
    assert all("申诉程序" not in source["content_preview"] for source in response["sources"])


def test_assistant_records_llm_error_without_local_answer_fallback() -> None:
    documents = [
        KnowledgeDocument("travel", "差旅报销制度", "财务", "travel.md", "制度正文", ["市内交通"])
    ]
    assistant = KnowledgeAssistant(
        FakeRetriever(),
        documents,
        llm_provider="ollama",
        llm_client=FailingChatClient(),
    )

    response = assistant.ask("出差打车可以报销吗？")

    assert "暂时无法生成服务答复" in response["answer"]
    assert "Ollama connection failed" in response["debug"]["llm_error"]
    assert "Ollama connection failed" in assistant.last_llm_error


def test_assistant_does_not_call_llm_for_irrelevant_context() -> None:
    client = TrackingChatClient()
    documents = [
        KnowledgeDocument("expense", "费用报销制度", "财务", "expense.md", "制度正文", ["不予报销情形"])
    ]
    assistant = KnowledgeAssistant(
        IrrelevantRetriever(),
        documents,
        llm_provider="ollama",
        llm_client=client,
    )

    response = assistant.ask("公司食堂几点开门？")

    assert "依据不足" in response["answer"]
    assert response["sources"] == []
    assert client.calls == 0


def test_source_preview_contains_the_full_retrieved_chunk() -> None:
    content = "前置说明。" * 50 + "关键审批条件在这里。"
    result = SearchResult(
        TextChunk("policy-0", "policy", "制度", "分类", "规则", "policy.md", content, 0),
        1.0,
        "hybrid",
        1,
    )

    sources = KnowledgeAssistant._format_sources([result])

    assert "关键审批条件在这里" in sources[0]["content_preview"]


def test_drafting_evidence_accepts_supplier_counterparty_wording() -> None:
    result = SearchResult(
        TextChunk("draft", "contract", "合同管理制度", "商务", "合同起草", "contract.md",
                  "合同起草前，应对合同相关方的主体资格、资信状况及履行能力进行调查。", 0),
        1.0, "hybrid", 1,
    )
    assert has_sufficient_evidence("准备起草一份采购合同时，我们要先调查供应商哪几方面？", [result])


def test_generated_absolute_limit_is_replaced_with_labelled_source_facts() -> None:
    result = SearchResult(
        TextChunk("leave", "hr", "员工管理制度", "人力资源", "事假", "hr.md",
                  "事假原则上应提前1个工作日提交申请；事假年度累计原则上不超过10个工作日。", 0),
        1.0, "hybrid", 1,
    )

    class Retriever:
        def retrieve(self, query, top_k=5):
            return {"final_results": [result]}

    class Client:
        def generate(self, system_prompt, user_prompt):
            return "原则上提前1个工作日申请，一年最多只能请10个工作日。"

    assistant = KnowledgeAssistant(Retriever(), [], llm_client=Client())
    response = assistant.ask("事假需要提前多久？一年最多请几天？")
    assert "未通过限定条件校验" in response["answer"]
    assert "年度累计原则上不超过10个工作日" in response["answer"]
    assert "最多只能" not in response["answer"]
    assert response["debug"]["answer_mode"] == "source_facts"
    assert response["debug"]["answer_validation_error"]


def test_employment_duration_does_not_prove_calendar_year_eligibility() -> None:
    result = SearchResult(
        TextChunk("leave", "hr", "员工管理制度", "人力资源", "年假", "hr.md",
                  "员工自入职后的第二个日历年起可享受年假，具体额度按累计工作年限核定。", 0),
        1.0, "hybrid", 1,
    )
    assistant = KnowledgeAssistant(FakeRetriever(), [], llm_client=TrackingChatClient())
    for duration in ("刚入职2个月", "入职10年"):
        answer = assistant._compose_llm_answer("上一问：年假有几天？\n当前追问：" + duration, [result])
        assert "入职日期" in answer
        assert "暂不适用" not in answer
        assert "尚未进入" not in answer


def test_source_selection_supports_workflow_paraphrases_without_unrelated_pages() -> None:
    cases = [
        ("账号申请是谁发起？试用期目标谁定？", "人力资源部发起账号开通申请。"),
        ("转正走审批要提前几个工作日？", "试用期届满前10个工作日发起转正申请。"),
        ("上海住宿限额和报销单提交期限？", "住宿标准：员工级450元。"),
        ("驳船运输商准入要什么许可？", "驳船供应商须持水路运输许可证。"),
    ]
    for question, relevant in cases:
        results = [SearchResult(TextChunk(str(i), "d", "制度", "商务", str(i), "d.md", text, i), 1, "hybrid", i + 1)
                   for i, text in enumerate(("公司公章申请需核实印章使用范围。", relevant))]
        filtered = _source_results_for_question(question, results)
        assert len(filtered) == 1
        assert filtered[0].chunk.content == relevant


def test_generated_exception_cannot_weaken_mandatory_supplier_limit() -> None:
    result = SearchResult(TextChunk("supplier", "s", "供应商制度", "商务", "准入", "s.md",
                                   "驳船供应商注册资金不得少于200万元；必须持有有效的水路运输许可证。"
                                   "必须为所经营船舶投保船舶险、承运人责任险，保险金额不低于500万元。", 0),
                          1, "hybrid", 1)

    class Retriever:
        def retrieve(self, query, top_k=5):
            return {"final_results": [result]}

    class Client:
        def generate(self, system_prompt, user_prompt):
            return "驳船供应商注册资金原则上不得少于200万元。"

    response = KnowledgeAssistant(Retriever(), [], llm_client=Client()).ask("驳船运输商注册资金和保险有什么要求？")
    assert response["debug"]["answer_mode"] == "source_facts"
    assert "原则上" not in response["answer"]
    assert "水路运输许可证" in response["answer"]
