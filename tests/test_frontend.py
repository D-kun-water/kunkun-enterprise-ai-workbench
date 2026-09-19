from frontend import streamlit_app


def _sample_ai_news() -> tuple[list[dict[str, str]], None]:
    return (
        [
            {
                "title": "示例模型升级",
                "summary": "用于验证行业资讯卡片的测试数据。",
                "published": "2026-09-16",
                "link": "https://example.com/ai-update",
                "source": "官方来源",
                "category": "模型升级",
                "image": "https://example.com/logo.png",
                "image_kind": "logo",
                "region": "国内",
            }
        ],
        None,
    )


class _FeedbackColumn:
    def __init__(self, clicked_label: str | None = None) -> None:
        self.clicked_label = clicked_label
        self.success_messages: list[str] = []

    def button(self, label: str, **kwargs) -> bool:
        return label == self.clicked_label

    def success(self, message: str) -> None:
        self.success_messages.append(message)

    def error(self, message: str) -> None:
        raise AssertionError(message)


class _FeedbackStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.columns_result = [_FeedbackColumn("👍"), _FeedbackColumn(), _FeedbackColumn()]
        self.rerun_count = 0

    def columns(self, spec):
        return self.columns_result

    def container(self, **kwargs):
        return _FeedbackContainer()

    def caption(self, message: str) -> None:
        return None

    def markdown(self, *args, **kwargs) -> None:
        return None

    def selectbox(self, *args, **kwargs):
        return "未解决"

    def button(self, *args, **kwargs) -> bool:
        return False

    def success(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        raise AssertionError(message)

    def rerun(self) -> None:
        self.rerun_count += 1


class _FeedbackContainer:
    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def test_llm_fallback_warning_is_only_shown_for_runtime_fallback() -> None:
    assert hasattr(streamlit_app, "llm_fallback_warning")
    assert "未生成答复" in streamlit_app.llm_fallback_warning(
        {"debug": {"llm_error": "Ollama connection failed"}}
    )
    assert streamlit_app.llm_fallback_warning({"debug": {"llm_error": ""}}) == ""


def test_reranker_status_distinguishes_pending_and_failed_loads() -> None:
    assert (
        streamlit_app.reranker_status_label(
            {"reranker_enabled": True, "reranker_effective": False, "errors": {}}
        )
        == "已配置，首次问答时加载"
    )
    assert (
        streamlit_app.reranker_status_label(
            {
                "reranker_enabled": True,
                "reranker_effective": False,
                "errors": {"reranker_runtime": "model unavailable"},
            }
        )
        == "加载失败，已回退"
    )


def test_render_data_table_falls_back_when_arrow_unavailable(monkeypatch) -> None:
    rendered: list[str] = []

    def raise_arrow_error(*args, **kwargs):
        raise ImportError("pyarrow DLL unavailable")

    monkeypatch.setattr(streamlit_app.st, "dataframe", raise_arrow_error)
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda body, **kwargs: rendered.append(body),
    )

    streamlit_app.render_data_table([{"文档": "测试制度", "状态": "可用"}])

    assert rendered
    assert "<table" in rendered[0]
    assert "测试制度" in rendered[0]


def test_format_evaluation_row_labels_refusal_case_as_passed() -> None:
    row = streamlit_app.format_evaluation_row(
        {
            "case_id": "HF-011",
            "question": "公司是否提供宠物医疗报销？",
            "expected_source": "",
            "rank": 0,
            "hit_at_3": 0,
            "failure_type": "passed",
            "should_refuse": True,
        }
    )

    assert row["排名"] == "不应召回"
    assert row["Hit@3"] == "拒答通过"
    assert row["失败类型"] == "passed"


def test_employee_branding_uses_kunkun_name_and_mascot_asset() -> None:
    assert streamlit_app.COMPANY_NAME == "鲲坤科技"
    assert streamlit_app.MASCOT_PATH.is_file()
    assert streamlit_app.SIDEBAR_LOGO_PATH.is_file()


def test_sidebar_role_profile_drives_the_active_identity() -> None:
    profile = streamlit_app.role_profile("legal")
    markup = streamlit_app.sidebar_user_profile_html("legal")

    assert profile["label"] == "法务"
    assert profile["name"] in markup
    assert profile["department"] in markup
    assert "当前身份" in markup


def test_role_profiles_include_distinct_avatar_and_personal_work_items() -> None:
    profiles = [streamlit_app.role_profile(role) for role in streamlit_app.ROLE_PROFILES]
    assert len({profile["avatar_color"] for profile in profiles}) == 4
    assert len({profile["todos"][0]["title"] for profile in profiles}) == 4
    assert len({profile["applications"][0]["title"] for profile in profiles}) == 4


def test_sidebar_role_trigger_contains_identity_details() -> None:
    label = streamlit_app.sidebar_role_trigger_label("business")
    assert "周予安" in label
    assert "业务发展中心" in label
    assert "当前身份 · 业务" in label


def test_sidebar_user_card_uses_clickable_role_trigger_and_bottom_anchor() -> None:
    markup = streamlit_app.sidebar_user_profile_html("business")
    assert 'data-role-trigger="true"' in markup
    assert "sidebar-user-card-content" in markup
    assert streamlit_app.SIDEBAR_USER_CARD_KEY == "sidebar-user-card"


def test_sidebar_user_card_uses_identity_trigger_and_name_only_role_list(monkeypatch) -> None:
    popover_labels: list[str] = []
    selected: list[tuple[str, tuple[str, ...], dict]] = []

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _SidebarStreamlit:
        def container(self, **kwargs):
            return _Container()

        def popover(self, label, **kwargs):
            popover_labels.append(label)
            return _Container()

        def radio(self, label, options, **kwargs):
            selected.append((label, options, kwargs))
            return "legal"

    monkeypatch.setattr(streamlit_app, "st", _SidebarStreamlit())

    role = streamlit_app.render_sidebar_user_card("employee")

    assert role == "legal"
    assert len(popover_labels) == 1
    assert "林知夏" in popover_labels[0]
    assert "产品与增长中心" in popover_labels[0]
    assert len(selected) == 1
    assert selected[0][0] == "切换人员"
    assert selected[0][2]["label_visibility"] == "collapsed"
    assert selected[0][2]["format_func"]("legal") == "苏闻"


def test_role_change_closes_assistant_and_clears_previous_identity_conversation() -> None:
    streamlit_app.st.session_state.clear()
    streamlit_app.st.session_state.update(
        {
            "active_role": "employee",
            "messages": [{"role": "user", "content": "旧身份的问题"}],
            "assistant_dialog_open": True,
            "assistant_dialog_page": "概览",
            "pending_question": "旧身份待发送的问题",
        }
    )

    streamlit_app._handle_role_change()

    assert streamlit_app.st.session_state["messages"] == []
    assert streamlit_app.st.session_state["assistant_dialog_open"] is False
    assert "assistant_dialog_page" not in streamlit_app.st.session_state
    assert "pending_question" not in streamlit_app.st.session_state


def test_role_change_keeps_conversations_isolated_by_identity() -> None:
    streamlit_app.st.session_state.clear()
    streamlit_app.st.session_state.update(
        {
            "active_role": "legal",
            "_assistant_role": "employee",
            "messages": [{"role": "user", "content": "法务不应看到这条"}],
        }
    )

    streamlit_app._handle_role_change()

    assert streamlit_app.st.session_state["messages"] == []
    assert streamlit_app.st.session_state["assistant_messages_by_role"]["employee"][0]["content"] == "法务不应看到这条"

    streamlit_app.st.session_state["messages"] = [{"role": "user", "content": "法务自己的问题"}]
    streamlit_app.st.session_state["active_role"] = "employee"
    streamlit_app._handle_role_change()

    assert streamlit_app.st.session_state["messages"][0]["content"] == "法务不应看到这条"


def test_contract_review_entry_supports_all_assignable_roles() -> None:
    assert "合同评审" in streamlit_app.employee_pages_for_role("employee")
    assert "合同评审" in streamlit_app.employee_pages_for_role("business")
    assert "合同评审" in streamlit_app.employee_pages_for_role("legal")
    assert "合同评审" in streamlit_app.employee_pages_for_role("admin")
    assert "合同评审" not in streamlit_app.OPERATIONS_PAGES


def test_sidebar_user_card_markup_exposes_role_switcher_container() -> None:
    assert "sidebar-user-card" in streamlit_app.SIDEBAR_USER_CARD_KEY


def test_unified_navigation_exposes_employee_and_company_pages() -> None:
    assert "管理视图" not in streamlit_app.PAGE_OPTIONS
    assert streamlit_app.PAGE_OPTIONS == (
        "概览",
        "人事服务",
        "财务中心",
        "ERP 系统",
        "合同评审",
        "个人中心",
        "知识库状态",
        "基础评测",
    )
    assert streamlit_app.EMPLOYEE_PAGES == (
        "概览",
        "人事服务",
        "财务中心",
        "ERP 系统",
        "合同评审",
        "个人中心",
    )
    assert streamlit_app.OPERATIONS_PAGES == ("知识库状态", "基础评测")


def test_contract_review_page_is_a_workspace_surface_with_boundary_copy(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))
    monkeypatch.setattr(streamlit_app, "fetch_contracts", lambda *args: [])

    streamlit_app.render_contract_review_tab("http://api.example", "legal")

    joined = "\n".join(rendered)
    assert "<h2>合同评审</h2>" in joined
    assert "员工工作台" in joined
    assert "当前身份" in joined
    assert "不构成法律结论" in joined


def test_contract_review_page_denies_unauthorized_role(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_contract_review_tab("http://api.example", "unknown")

    joined = "\n".join(rendered)
    assert "无权查看合同评审" in joined
    assert "<h2>合同评审</h2>" not in joined


def test_erp_page_renders_realistic_enterprise_operations_surface(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_erp_page()

    joined = "\n".join(rendered)
    for label in (
        "ERP 系统",
        "采购与供应链",
        "销售订单",
        "库存概览",
        "生产协同",
        "待处理流程",
        "采购订单",
        "SO-260828",
        "ERP 页面使用样例数据",
    ):
        assert label in joined


def test_erp_style_defines_dense_responsive_business_surface(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert ".erp-shell {" in css
    assert ".erp-kpi-grid {" in css
    assert ".erp-table {" in css
    assert ".erp-status.pending {" in css


def test_company_overview_is_an_enterprise_information_page_without_operations(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))
    monkeypatch.setattr(streamlit_app, "fetch_ai_frontier_news", _sample_ai_news)

    streamlit_app.render_company_overview()

    joined = "\\n".join(rendered)
    assert "公司公告" in joined
    assert "公司任命公示" in joined
    assert "AI 前沿资讯" in joined
    assert "鲲坤科技" in joined
    assert "portal-nav" in joined
    assert "portal-banner" in joined
    assert "portal-backdrop" in joined
    assert "portal-banner-poster" not in joined
    assert "portal-banner-visual" not in joined
    assert "公司运营概览" not in joined
    assert "我的待办" not in joined


def test_company_dashboard_contains_employee_and_leadership_surfaces(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))
    monkeypatch.setattr(streamlit_app, "fetch_ai_frontier_news", _sample_ai_news)

    streamlit_app.render_company_overview()

    joined = "\n".join(rendered)
    for label in (
        "AI 前沿资讯",
        "公司任命公示",
        "公司公告",
        "产品与解决方案",
        "智能装配与检测产线",
        "晶圆级封装与测试方案",
        "高可靠功率器件与连接器",
        "鲲坤企业智能平台",
        "页面内容仅用于内部员工信息展示",
        "portal-nav",
        "portal-banner",
    ):
        assert label in joined
    assert "我的待办" not in joined
    assert "公司实时核心指标" not in joined
    assert "快速发起审批" not in joined
    assert "全局搜索" not in joined
    assert "今日员工活跃数" not in joined
    assert "KUN KUN TECHNOLOGY" not in joined
    assert "上海市浦东新区示例路 88 号" not in joined
    assert "portal-footer" not in joined


def test_personal_center_contains_private_workspace_and_joining_day(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_personal_center()

    joined = "\n".join(rendered)
    for label in ("个人中心", "我的待办", "我的申请", "今天是你入职的第", "入职日期", "个人资料", "通知设置"):
        assert label in joined
    assert "data:image/png;base64," in joined


def test_personal_center_uses_selected_role_profile(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_personal_center("legal")

    joined = "\n".join(rendered)
    assert "苏闻" in joined
    assert "法务合规部" in joined
    assert "当前身份 · 法务" in joined


def test_personal_center_uses_role_specific_todos_and_applications(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_personal_center("business")

    joined = "\n".join(rendered)
    assert "确认客户合同付款节点" in joined
    assert "华东项目报价评审" in joined
    assert "补充差旅报销材料" not in joined


def test_main_does_not_render_connection_status(monkeypatch) -> None:
    rendered: list[str] = []

    class _Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _MainStreamlit:
        sidebar = _Sidebar()
        session_state: dict[str, object] = {}

        def set_page_config(self, **kwargs):
            return None

        def markdown(self, body, **kwargs):
            rendered.append(body)

        def radio(self, label, options, **kwargs):
            return options[0] if label == "员工工作台" else None

        def selectbox(self, label, options, **kwargs):
            return "employee"

        def divider(self):
            return None

    monkeypatch.setattr(streamlit_app, "st", _MainStreamlit())
    monkeypatch.setattr(streamlit_app, "inject_workspace_style", lambda: None)
    monkeypatch.setattr(streamlit_app, "render_company_overview", lambda: None)
    monkeypatch.setattr(streamlit_app, "render_global_assistant", lambda *args, **kwargs: None)

    streamlit_app.main()

    assert not any("服务已连接" in body for body in rendered)


def test_hr_and_finance_pages_include_approval_and_reimbursement_flows(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.render_hr_services_page()
    streamlit_app.render_finance_center_page()

    joined = "\n".join(rendered)
    for label in ("人事服务", "请假与考勤", "审批进度", "财务中心", "发票报销", "报销流程"):
        assert label in joined


def test_static_employee_pages_render_their_expected_content(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda body, **kwargs: rendered.append(body),
    )
    monkeypatch.setattr(streamlit_app, "fetch_ai_frontier_news", _sample_ai_news)

    streamlit_app.render_ai_news_page()

    joined = "\\n".join(rendered)
    assert "AI 行业资讯" in joined
    assert "LIVE MULTI-SOURCE" in joined
    assert "模型升级" in joined


def test_ai_news_feed_parser_keeps_source_date_image_and_original_link() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss xmlns:media="http://search.yahoo.com/mrss/" version="2.0"><channel><item>
      <title>Example AI model gets a major upgrade</title>
      <link>https://example.com/model-upgrade</link>
      <description><![CDATA[<p>A company released a faster enterprise model.</p>]]></description>
      <pubDate>Wed, 16 Sep 2026 08:00:00 GMT</pubDate>
      <source>Example News</source>
      <media:thumbnail url="https://example.com/cover.jpg" />
    </item></channel></rss>"""
    config = {
        "source": "AI 行业动态",
        "category": "模型升级",
        "image": "https://example.com/fallback.png",
        "use_entry_source": "1",
    }

    items = streamlit_app._parse_ai_news_feed(feed, config)

    assert items == [
        {
            "title": "Example AI model gets a major upgrade",
            "summary": "A company released a faster enterprise model.",
            "published": "2026-09-16",
            "link": "https://example.com/model-upgrade",
            "source": "Example News",
            "category": "模型升级",
            "image": "https://example.com/cover.jpg",
            "image_kind": "photo",
            "region": "国际",
        }
    ]


def test_employee_header_embeds_mascot_image_for_streamlit(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda body, **kwargs: rendered.append(body),
    )

    streamlit_app.render_employee_header()

    assert rendered
    assert "data:image/png;base64," in rendered[0]


def test_news_cards_remove_empty_call_to_action_but_keep_clickable_title() -> None:
    item = _sample_ai_news()[0][0]
    item["summary"] = '<div><a href="https://example.com">点击查看原文&gt;</a></div>'
    for portal in (True, False):
        markup = streamlit_app._news_card_markup(item, portal=portal)
        assert "点击查看原文" not in markup
        assert "阅读原文" not in markup
        assert "news-title-link" in markup
        assert 'href="https://example.com/ai-update"' in markup


def test_article_cover_uses_publisher_image_instead_of_qbitai_site_logo(monkeypatch) -> None:
    class Response:
        url = "https://www.qbitai.com/2026/09/test.html"
        content = b'''<meta content="https://www.qbitai.com/logo.png" property="og:image">
        <img src="/qrcode.jpg"><img src="https://i.qbitai.com/wp-content/uploads/2026/09/story.jpg">'''

        def raise_for_status(self):
            pass

    monkeypatch.setattr(streamlit_app.requests, "get", lambda *args, **kwargs: Response())
    result = streamlit_app._article_metadata.__wrapped__(Response.url)
    assert result["image"] == "https://i.qbitai.com/wp-content/uploads/2026/09/story.jpg"


def test_anthropic_news_parser_keeps_company_source_and_publication_date() -> None:
    html = '''<a href="/news/test-release"><time>Sep 10, 2026</time>
    <h4>New Claude model</h4><p>Model release details.</p></a>
    <a href="https://untrusted.example/news/test"><time>Sep 11, 2026</time><h4>Ignore</h4></a>'''
    items = streamlit_app._parse_anthropic_news(html)
    assert len(items) == 1
    assert items[0]["published"] == "2026-09-10"
    assert items[0]["source"] == "Anthropic"
    assert items[0]["link"] == "https://www.anthropic.com/news/test-release"


def test_workspace_style_defines_the_polished_responsive_surface(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda body, **kwargs: rendered.append(body),
    )

    streamlit_app.inject_workspace_style()

    assert "--canvas: #f5f5f7" in rendered[0]
    assert "backdrop-filter: blur" in rendered[0]
    assert "max-height: 300px" in rendered[0]
    assert "object-fit: cover" in rendered[0]
    assert "stBaseButton-header" in rendered[0]
    assert "padding: 3.25rem 2rem 5rem" in rendered[0]
    assert "@media (max-width: 700px)" in rendered[0]


def test_overview_style_prioritizes_public_portal_content(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert ".portal-banner {" in css
    assert ".portal-backdrop {" in css
    assert ".sidebar-logo {" in css
    assert "width: 53%" in css
    assert "height: 600px" in css
    assert "opacity: .42" in css
    assert "filter: blur(2px)" in css
    assert "mask-composite: intersect" in css
    assert "transparent 70%" in css
    assert "rgba(26,66,130,.3) 54%" in css
    assert "mask-image: linear-gradient" in css
    assert ".portal-public-card.notice {" in css
    assert ".portal-ai-card {" in css
    assert "grid-template-columns: repeat(3, 1fr)" in css


def test_format_answer_html_preserves_list_items_and_escapes_source_text() -> None:
    html = streamlit_app.format_answer_html(
        "根据制度原文：\n- 需要提交 <审批> 记录\n- 3 个工作日内完成"
    )

    assert "<ul>" in html
    assert "<li>需要提交 &lt;审批&gt; 记录</li>" in html
    assert "3 个工作日内完成" in html


def test_employee_answer_text_hides_internal_citation_copy() -> None:
    answer = streamlit_app.employee_answer_text(
        "根据《请假与年休假管理制度》中的相关规定：\n- 年休假最小申请单位为 0.5 天。\n\n请以页面下方展示的原文来源为准。"
    )

    assert answer == "- 年休假最小申请单位为 0.5 天。"
    assert "原文来源" not in answer
    assert "制度管理" not in answer


def test_employee_answer_text_normalizes_source_numbering_for_readability() -> None:
    answer = streamlit_app.employee_answer_text(
        "- 3. 返程后 10 个工作日内提交报销及凭证。\n"
        "- 2、材料包括发票、行程单和支付记录。"
    )

    assert answer == "- 返程后 10 个工作日内提交报销及凭证。\n- 材料包括发票、行程单和支付记录。"


def test_employee_answer_text_removes_dangling_section_headings() -> None:
    answer = streamlit_app.employee_answer_text("- 返程后 10 个工作日内提交报销。\n- 出差申请：")

    assert answer == "- 返程后 10 个工作日内提交报销。"


def test_employee_answer_text_removes_raw_markdown_emphasis() -> None:
    answer = streamlit_app.employee_answer_text("若关注的是**试用期延长**，最长不得超过一个月。")

    assert answer == "若关注的是试用期延长，最长不得超过一个月。"


def test_assistant_dialog_typography_is_readable_for_chinese_policy_answers(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert 'font-family: "PingFang SC", "Microsoft YaHei"' in css
    assert ".assistant-dialog-body .answer-panel" in css
    assert "line-height: 1.7" in css


def test_assistant_message_avatars_use_neutral_enterprise_style(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert "stChatMessageAvatarUser\"] { background: #e8d5e8" in css
    assert "stChatMessageAvatarAssistant\"] { background: #eaf3ff" in css


def test_conversation_turns_render_oldest_turn_first() -> None:
    messages = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
        {"role": "user", "content": "第二个问题"},
        {"role": "assistant", "content": "第二个回答"},
    ]

    turns = streamlit_app.conversation_turns_oldest_first(messages)

    assert [turn[0]["content"] for turn in turns] == ["第一个问题", "第二个问题"]


def test_feedback_key_is_unique_for_repeated_questions() -> None:
    first = {"role": "assistant", "content": "同一个回答", "feedback_id": "answer-1"}
    second = {"role": "assistant", "content": "同一个回答", "feedback_id": "answer-2"}

    assert streamlit_app.feedback_key_for_message(first, 0) != streamlit_app.feedback_key_for_message(second, 1)


def test_api_history_contains_previous_turns_but_not_current_question() -> None:
    messages = [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]

    history = streamlit_app.conversation_history_for_api(messages)

    assert history == messages
    assert all("feedback_id" not in message for message in history)


def test_successful_feedback_reruns_to_show_the_recorded_state(monkeypatch) -> None:
    fake_st = _FeedbackStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "api_request", lambda *args, **kwargs: {"feedback_id": "f-1"})

    streamlit_app.render_feedback_controls(
        "http://api.example", "年休假怎么申请？", "请在平台提交申请。", "message-1"
    )

    assert fake_st.session_state["feedback-state-message-1"] == "up"
    assert fake_st.rerun_count == 1


def test_assistant_launcher_uses_compact_text_surface(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: False)

    assert streamlit_app.render_assistant_launcher() is False

    joined = "\n".join(rendered)
    assert "data:image/png;base64," in joined


def test_assistant_dialog_style_is_small_and_bottom_right(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert ".assistant-launcher" in css
    assert "[role=\"dialog\"]" in css
    assert "bottom: 1.25rem" in css
    assert "width: min(460px, calc(100vw - 2rem))" in css


def test_assistant_dialog_has_scrollable_body_and_sticky_composer(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert ".assistant-dialog-body" in css
    assert "overflow-y: auto" in css
    assert "position: sticky" in css
    assert "overscroll-behavior: contain" in css
    assert "order: 10" in css


def test_assistant_dialog_uses_neutral_palette(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda body, **kwargs: rendered.append(body))

    streamlit_app.inject_workspace_style()

    css = rendered[0]
    assert "--assistant-sky: #8ed3df" in css
    assert "--assistant-pink: #d49ac2" in css
    assert ".assistant-dialog-head img" in css
    assert ".assistant-dialog-badge" not in css


def test_assistant_dialog_uses_streamlit_141_compatible_arguments(monkeypatch) -> None:
    dialog_calls: list[tuple[str, str]] = []

    def fake_dialog(title: str, *, width: str):
        dialog_calls.append((title, width))

        def decorator(function):
            return function

        return decorator

    monkeypatch.setattr(streamlit_app.st, "dialog", fake_dialog)
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app, "render_chat_tab", lambda *args, **kwargs: None)

    streamlit_app.render_assistant_dialog("http://api.example")

    assert dialog_calls == [("员工制度服务台", "small")]


def test_global_assistant_reopens_dialog_after_fragment_rerun(monkeypatch) -> None:
    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {"assistant_dialog_open": True}

        def dialog(self, *args, **kwargs):
            def decorator(function):
                return function

            return decorator

        def markdown(self, *args, **kwargs):
            return None

    fake_st = FakeStreamlit()
    dialog_calls: list[str] = []
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "render_chat_tab", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app, "render_assistant_launcher", lambda: False)
    monkeypatch.setattr(streamlit_app, "render_assistant_dialog", lambda api_url: dialog_calls.append(api_url))

    streamlit_app.render_global_assistant("http://api.example")

    assert fake_st.session_state["assistant_dialog_open"] is True
    assert dialog_calls == ["http://api.example"]


def test_global_assistant_clears_open_state_when_page_changes(monkeypatch) -> None:
    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "assistant_dialog_open": True,
                "assistant_dialog_page": "概览",
            }

    fake_st = FakeStreamlit()
    dialog_calls: list[str] = []
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "render_assistant_launcher", lambda *args, **kwargs: False)
    monkeypatch.setattr(streamlit_app, "render_assistant_dialog", lambda api_url: dialog_calls.append(api_url))

    streamlit_app.render_global_assistant("http://api.example", current_page="ERP 系统")

    assert fake_st.session_state["assistant_dialog_open"] is False
    assert dialog_calls == []


def test_main_renders_selected_page_before_global_assistant(monkeypatch) -> None:
    """Page controls must run before the assistant consumes pending questions."""

    class _Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _MainStreamlit:
        def __init__(self):
            self.sidebar = _Sidebar()
            self.session_state: dict[str, object] = {}

        def set_page_config(self, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def radio(self, label, options, **kwargs):
            return options[0] if label == "员工工作台" else None

        def selectbox(self, label, options, **kwargs):
            return "employee"

        def divider(self):
            return None

    events: list[str] = []
    monkeypatch.setattr(streamlit_app, "st", _MainStreamlit())
    monkeypatch.setattr(streamlit_app, "inject_workspace_style", lambda: None)
    monkeypatch.setattr(streamlit_app, "fetch_health", lambda _api_url: {"status": "ok"})
    def render_page_with_preset_question():
        events.append("page")
        streamlit_app.st.session_state["pending_question"] = "忘记打卡后几天内可以补卡？"
        streamlit_app.st.session_state["assistant_dialog_open"] = True

    def render_assistant_after_page(*args, **kwargs):
        events.append("assistant")
        assert streamlit_app.st.session_state.get("pending_question") == "忘记打卡后几天内可以补卡？"

    monkeypatch.setattr(streamlit_app, "render_company_overview", render_page_with_preset_question)
    monkeypatch.setattr(
        streamlit_app,
        "render_global_assistant",
        render_assistant_after_page,
    )

    streamlit_app.main()

    assert events == ["page", "assistant"]


def test_sidebar_status_reports_unknown_when_health_unavailable(monkeypatch) -> None:
    def unavailable(_api_url):
        raise streamlit_app.requests.ConnectionError("connection refused")

    monkeypatch.setattr(streamlit_app, "fetch_health", unavailable)

    assert streamlit_app.sidebar_status_label("http://api.example") == "服务未连接/状态未知"


def test_sidebar_marks_knowledge_base_as_operations_view(monkeypatch) -> None:
    rendered: list[str] = []

    class _Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _MainStreamlit:
        sidebar = _Sidebar()

        def set_page_config(self, **kwargs):
            return None

        def markdown(self, body, **kwargs):
            rendered.append(body)

        def radio(self, label, options, **kwargs):
            return options[0] if label == "员工工作台" else None

        def selectbox(self, label, options, **kwargs):
            return "employee"

        def divider(self):
            return None

    monkeypatch.setattr(streamlit_app, "st", _MainStreamlit())
    monkeypatch.setattr(streamlit_app, "inject_workspace_style", lambda: None)
    monkeypatch.setattr(streamlit_app, "fetch_health", lambda _api_url: {"status": "ok"})
    monkeypatch.setattr(streamlit_app, "render_company_overview", lambda: None)
    monkeypatch.setattr(streamlit_app, "render_global_assistant", lambda *args, **kwargs: None)

    streamlit_app.main()

    assert any("知识库运营" in body for body in rendered)
