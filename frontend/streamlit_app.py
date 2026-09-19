"""鲲坤科技企业 AI 协作工作台的员工端工作台。"""

from __future__ import annotations

import os
import re
import uuid
import base64
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit
from xml.etree import ElementTree

import requests
import streamlit as st
import streamlit.components.v1 as components

from app.contracts import export_contract_review


DEFAULT_API_URL = os.getenv("KNOWLEDGE_API_URL", "http://127.0.0.1:8000")
ROOT_DIR = Path(__file__).resolve().parent.parent
MASCOT_PATH = ROOT_DIR / "frontend" / "assets" / "kunqun-mascot.png"
SIDEBAR_LOGO_PATH = ROOT_DIR / "frontend" / "assets" / "kunkun-logo.png"
PORTRAIT_PATH = ROOT_DIR / "frontend" / "assets" / "portrait.png"
COMPANY_NAME = "鲲坤科技"
EMPLOYEE_PAGES = (
    "概览",
    "人事服务",
    "财务中心",
    "ERP 系统",
    "合同评审",
    "个人中心",
)
OPERATIONS_PAGES = (
    "知识库状态",
    "基础评测",
)
PAGE_OPTIONS = EMPLOYEE_PAGES + OPERATIONS_PAGES
CONTRACT_REVIEW_ROLES = frozenset({"employee", "business", "legal", "admin"})
SIDEBAR_USER_CARD_KEY = "sidebar-user-card"
ROLE_PROFILES = {
    "employee": {
        "label": "员工",
        "name": "林知夏",
        "department": "产品与增长中心 · 员工",
        "avatar": "林",
        "avatar_color": "#2563eb",
        "title": "AI 产品运营",
        "joining_date": "2024-03-18",
        "employee_id": "KKT-0248",
        "manager": "周予安",
        "office": "上海 · A 座 12F",
        "status": "正式员工",
        "todos": [
            {"title": "补充差旅报销材料", "meta": "财务中心 · 今天 18:00 前", "chip": "即将到期", "tone": "amber"},
            {"title": "确认新员工入职信息", "meta": "人事服务 · 明天 12:00 前", "chip": "待确认", "tone": ""},
            {"title": "阅读信息安全制度更新", "meta": "公司公告 · 2 天前", "chip": "未读", "tone": ""},
        ],
        "applications": [
            {"title": "8 月差旅报销 · BX-20260821", "meta": "财务中心 · 审批中", "chip": "第 2/4 步", "tone": ""},
            {"title": "年休假 · 2026-09-04", "meta": "人事服务 · 已通过", "chip": "已完成", "tone": ""},
            {"title": "显示器更换申请 · IT-260817", "meta": "行政与 IT · 待处理", "chip": "待处理", "tone": "amber"},
        ],
    },
    "business": {
        "label": "业务",
        "name": "周予安",
        "department": "业务发展中心 · 业务负责人",
        "avatar": "周",
        "avatar_color": "#0f766e",
        "title": "业务负责人",
        "joining_date": "2023-08-12",
        "employee_id": "KKT-0136",
        "manager": "陈知远",
        "office": "上海 · A 座 15F",
        "status": "正式员工",
        "todos": [
            {"title": "确认客户合同付款节点", "meta": "合同协同 · 今天 17:00 前", "chip": "待确认", "tone": ""},
            {"title": "完成华东项目报价评审", "meta": "业务发展中心 · 明天 12:00 前", "chip": "进行中", "tone": ""},
            {"title": "更新重点客户跟进记录", "meta": "客户成功部 · 本周五前", "chip": "待处理", "tone": "amber"},
        ],
        "applications": [
            {"title": "华东项目报价评审 · BD-20260903", "meta": "业务中心 · 评审中", "chip": "第 3/5 步", "tone": ""},
            {"title": "重点客户拜访预算 · BD-20260829", "meta": "财务中心 · 已通过", "chip": "已完成", "tone": ""},
            {"title": "客户合同付款计划 · BD-20260826", "meta": "合同协同 · 待确认", "chip": "待确认", "tone": "amber"},
        ],
    },
    "legal": {
        "label": "法务",
        "name": "苏闻",
        "department": "法务合规部 · 法务专员",
        "avatar": "苏",
        "avatar_color": "#7c3aed",
        "title": "法务专员",
        "joining_date": "2022-11-07",
        "employee_id": "KKT-0098",
        "manager": "陈知远",
        "office": "上海 · A 座 10F",
        "status": "正式员工",
        "todos": [
            {"title": "复核供应商合同续约条款", "meta": "合同审阅 · 今天 16:00 前", "chip": "待复核", "tone": ""},
            {"title": "整理合同模板版本差异", "meta": "法务合规部 · 明天 18:00 前", "chip": "进行中", "tone": ""},
            {"title": "回复业务合同咨询", "meta": "法务合规部 · 2 小时前", "chip": "未读", "tone": "amber"},
        ],
        "applications": [
            {"title": "采购合同审阅 · LEG-20260902", "meta": "合同审阅 · 审阅中", "chip": "第 2/3 步", "tone": ""},
            {"title": "服务协议模板更新 · LEG-20260830", "meta": "法务合规部 · 已完成", "chip": "已归档", "tone": ""},
            {"title": "供应商合规材料核验 · LEG-20260827", "meta": "供应商管理 · 待补充", "chip": "待处理", "tone": "amber"},
        ],
    },
    "admin": {
        "label": "领导",
        "name": "陈知远",
        "department": "公司管理层 · 管理负责人",
        "avatar": "陈",
        "avatar_color": "#b45309",
        "title": "管理负责人",
        "joining_date": "2021-06-21",
        "employee_id": "KKT-0001",
        "manager": "董事会",
        "office": "上海 · A 座 18F",
        "status": "管理成员",
        "todos": [
            {"title": "审批季度经营复盘材料", "meta": "经营管理 · 今天 18:00 前", "chip": "待审批", "tone": ""},
            {"title": "确认年度预算调整方案", "meta": "财务管理部 · 明天 12:00 前", "chip": "待确认", "tone": ""},
            {"title": "阅读合同风险汇总", "meta": "法务合规部 · 1 天前", "chip": "未读", "tone": "amber"},
        ],
        "applications": [
            {"title": "年度预算调整 · ADM-20260828", "meta": "经营管理 · 审批中", "chip": "第 4/6 步", "tone": ""},
            {"title": "重大合同授权审批 · ADM-20260825", "meta": "法务合规部 · 已通过", "chip": "已完成", "tone": ""},
            {"title": "组织任命备案 · ADM-20260820", "meta": "人事服务 · 已归档", "chip": "已完成", "tone": ""},
        ],
    },
}


def _select_employee_nav() -> None:
    st.session_state["operations_nav"] = None


def _select_operations_nav() -> None:
    st.session_state["employee_nav"] = None


def _handle_role_change() -> None:
    """Reset private assistant state when the active local identity changes."""
    new_role = str(st.session_state.get("active_role", "employee"))
    previous_role = str(st.session_state.get("_assistant_role", "employee"))
    role_histories = st.session_state.setdefault("assistant_messages_by_role", {})
    current_messages = list(st.session_state.get("messages", []))
    if current_messages and previous_role != new_role:
        role_histories[previous_role] = current_messages
    st.session_state["messages"] = list(role_histories.get(new_role, []))
    st.session_state["_assistant_role"] = new_role
    # Give the popover wrapper a new key on the callback rerun.  Streamlit
    # otherwise preserves the open client-side popover even after selection.
    st.session_state["_role_selector_revision"] = int(
        st.session_state.get("_role_selector_revision", 0)
    ) + 1
    st.session_state["assistant_dialog_open"] = False
    st.session_state.pop("assistant_dialog_page", None)
    st.session_state.pop("pending_question", None)
    # Feedback controls are keyed to individual messages. Remove their
    # transient state so a new identity cannot inherit an old answer status.
    for key in list(st.session_state):
        if str(key).startswith(("feedback-state-", "show-reason-", "reason-", "up-", "down-", "submit-", "contract-")):
            st.session_state.pop(key, None)


def role_profile(role: str) -> dict[str, Any]:
    """Return the local identity shown for a supported access-boundary role."""
    return ROLE_PROFILES.get(role, ROLE_PROFILES["employee"])


def assistant_user_avatar() -> Any:
    """Use the active personal-center avatar in the employee service dialog."""
    role = str(st.session_state.get("active_role", "employee"))
    if role == "employee" and PORTRAIT_PATH.is_file():
        return str(PORTRAIT_PATH)
    profile = role_profile(role)
    try:
        from PIL import Image, ImageDraw, ImageFont

        size = 96
        image = Image.new("RGB", (size, size), "#ffffff")
        canvas = ImageDraw.Draw(image)
        canvas.ellipse((3, 3, size - 3, size - 3), fill=str(profile["avatar_color"]))
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 44)
        initial = str(profile["avatar"])
        box = canvas.textbbox((0, 0), initial, font=font)
        canvas.text(
            ((size - (box[2] - box[0])) / 2, (size - (box[3] - box[1])) / 2 - box[1]),
            initial,
            fill="#ffffff",
            font=font,
        )
        return image
    except Exception:
        return "👤"


def scroll_assistant_to_latest_turn() -> None:
    """Keep the newest user turn and its answer in the dialog viewport."""
    components.html(
        """
        <script>
          const parentDocument = window.parent.document;

          const getScrollContainer = (anchor) => {
            let node = anchor.parentElement;
            while (node) {
              const style = window.parent.getComputedStyle(node);
              if (
                node.scrollHeight > node.clientHeight &&
                /(auto|scroll)/.test(style.overflowY)
              ) return node;
              node = node.parentElement;
            }
            return parentDocument.querySelector(
              '[data-baseweb="modal"] [role="dialog"] > div:nth-child(2)'
            );
          };

          const placeLatestTurnAtTop = () => {
            const anchor = parentDocument.getElementById('assistant-latest-turn');
            const userMessages = Array.from(
              parentDocument.querySelectorAll('[data-testid="stChatMessage"]')
            ).filter((message) => message.querySelector('[data-testid="stChatMessageAvatarUser"]'));
            // Streamlit can move a zero-height markdown anchor into a wrapper
            // during reruns.  The actual final user card is the reliable
            // visual target, with the anchor kept only as a fallback.
            const target = userMessages.at(-1) || anchor;
            if (!target) return;
            const scrollContainer = getScrollContainer(target);
            // Streamlit moves focus back to the text field after a form
            // submit, which can override an earlier scroll.  Native
            // scrollIntoView updates every scroll ancestor; the explicit
            // container adjustment then removes the small header offset.
            target.scrollIntoView({ block: 'start', inline: 'nearest', behavior: 'auto' });
            if (scrollContainer) {
              const anchorBox = target.getBoundingClientRect();
              const containerBox = scrollContainer.getBoundingClientRect();
              const nextTop = scrollContainer.scrollTop + anchorBox.top - containerBox.top - 8;
              scrollContainer.scrollTop = Math.max(0, nextTop);
            }
          };

          // The answer card changes height while Streamlit finishes rendering.
          // Set the position twice so the newest question stays at the top instead
          // of leaving the employee halfway down the previous conversation.
          window.parent.requestAnimationFrame(placeLatestTurnAtTop);
          window.setTimeout(placeLatestTurnAtTop, 320);
          window.setTimeout(placeLatestTurnAtTop, 900);
          window.setTimeout(placeLatestTurnAtTop, 1600);
        </script>
        """,
        height=0,
    )


def employee_pages_for_role(role: str) -> tuple[str, ...]:
    """Filter sensitive workspace entries before rendering navigation."""
    if role in CONTRACT_REVIEW_ROLES:
        return EMPLOYEE_PAGES
    return tuple(page for page in EMPLOYEE_PAGES if page != "合同评审")


def sidebar_user_profile_html(role: str) -> str:
    profile = role_profile(role)
    return (
        f'<div class="sidebar-user-card-content" data-role-trigger="true" aria-label="当前身份 · {escape(profile["label"])}"><div class="sidebar-user">'
        f'<span class="sidebar-avatar" style="--avatar-color:{escape(profile["avatar_color"])}">{escape(profile["avatar"])}</span>'
        "<div>"
        f'<strong>{escape(profile["name"])}</strong>'
        f'<small>{escape(profile["department"])}</small>'
        "</div></div></div>"
    )


def sidebar_role_trigger_label(role: str) -> str:
    profile = role_profile(role)
    return f"{profile['name']} · {profile['department']} · 当前身份 · {profile['label']}"


def sidebar_role_option_label(role: str) -> str:
    profile = role_profile(role)
    return f"{profile['avatar']}  {profile['name']}\n{profile['department']} · 当前身份 · {profile['label']}"


def sidebar_role_card_label(role: str) -> str:
    profile = role_profile(role)
    # Keep the identity mark in the button label itself.  The previous CSS
    # avatar relied on a popover key that older Streamlit versions reject.
    return f"{profile['avatar']}  **{profile['name']}**\n{profile['department']}\n当前身份 · {profile['label']}"


def render_sidebar_user_card(role: str) -> str:
    """Render the identity card with the original popover role switcher."""
    container_factory = getattr(st, "container", None)
    selector_kwargs = {
        "label_visibility": "collapsed",
        "key": "active_role",
        "format_func": lambda value: role_profile(value)["name"],
        "on_change": _handle_role_change,
    }
    if container_factory is None:
        return st.selectbox("切换身份", tuple(ROLE_PROFILES), **selector_kwargs)

    with container_factory(key=SIDEBAR_USER_CARD_KEY):
        markdown_factory = getattr(st, "markdown", None)
        if markdown_factory is not None:
            markdown_factory(sidebar_user_profile_html(role), unsafe_allow_html=True)
        popover_factory = getattr(st, "popover", None)
        if popover_factory is not None:
            profile = role_profile(role)
            session_state = getattr(st, "session_state", {})
            revision = int(session_state.get("_role_selector_revision", 0))
            # Keep the visible card in the stable outer container while
            # remounting only the popover wrapper after a role change.
            with container_factory(key=f"{SIDEBAR_USER_CARD_KEY}-selector-{revision}"):
                # CSS overlays this trigger on the rendered identity card.  Keep
                # the label descriptive for accessibility while hiding it visually.
                with popover_factory(f"切换角色 · {profile['name']} · {profile['department']}"):
                    selected_role = st.radio("切换人员", tuple(ROLE_PROFILES), **selector_kwargs)
            return selected_role
        return st.selectbox("切换身份", tuple(ROLE_PROFILES), **selector_kwargs)


def api_request(method: str, api_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, f"{api_url.rstrip('/')}{path}", timeout=(5, 180), **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("detail", "")).strip()
        except (ValueError, TypeError, AttributeError):
            detail = response.text.strip()
        if detail:
            raise requests.HTTPError(f"{exc}；{detail}", response=response) from exc
        raise
    return response.json()


def llm_fallback_warning(result: dict[str, Any]) -> str:
    if result.get("debug", {}).get("llm_error"):
        return "大模型服务本次不可用，未生成答复，请稍后重试。"
    if result.get("debug", {}).get("vector_error"):
        return "向量检索本次不可用，已降级为关键词检索；结果仍须核对原文。"
    return ""


def reranker_status_label(health: dict[str, Any]) -> str:
    if health.get("reranker_effective"):
        return "已启用"
    if not health.get("reranker_enabled"):
        return "未启用"
    if health.get("errors", {}).get("reranker_runtime"):
        return "加载失败，已回退"
    return "已配置，首次问答时加载"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health(api_url: str) -> dict[str, Any]:
    return api_request("GET", api_url, "/health")


def sidebar_status_label(api_url: str) -> str:
    """Return a non-blocking, truthful service status for the sidebar."""
    try:
        health = fetch_health(api_url)
    except (requests.RequestException, ValueError, TypeError):
        return "服务未连接/状态未知"
    if health.get("status") == "ok":
        return "服务已连接"
    if health.get("status"):
        return f"服务状态：{health['status']}"
    return "服务状态未知"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_documents(api_url: str) -> list[dict[str, Any]]:
    return api_request("GET", api_url, "/knowledge-base/documents")["documents"]


def fetch_contracts(api_url: str, role: str) -> list[dict[str, Any]]:
    return api_request("GET", api_url, "/contracts", params={"role": role})["contracts"]


def role_display_name(role: str) -> str:
    profile = role_profile(role)
    return f"{profile['label']}（{profile['name']}）"


def render_data_table(rows: list[dict[str, Any]]) -> None:
    """Render a small table even when Streamlit's optional Arrow runtime is unavailable."""
    if not rows:
        st.info("暂无数据")
        return
    try:
        st.dataframe(rows, width="stretch", hide_index=True)
        return
    except (ImportError, OSError):
        # Some Windows Python environments have Streamlit installed without a
        # loadable pyarrow DLL. The status and evaluation tables are small, so a
        # safe static HTML fallback keeps the rest of the MVP usable.
        pass

    columns = list(dict.fromkeys(column for row in rows for column in row))
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, "")))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="table-fallback"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def format_answer_html(answer: str, *, show_overline: bool = True) -> str:
    """Render a concise employee-service response inside the answer surface."""
    body: list[str] = []
    list_open = False
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if not list_open:
                body.append("<ul>")
                list_open = True
            body.append(f"<li>{escape(line[2:])}</li>")
            continue
        if list_open:
            body.append("</ul>")
            list_open = False
        body.append(f"<p>{escape(line)}</p>")
    if list_open:
        body.append("</ul>")
    overline = '<p class="answer-overline">制度依据</p>' if show_overline else ""
    return '<section class="answer-panel">' + overline + "".join(body) + "</section>"


def employee_answer_text(answer: str) -> str:
    """Keep the employee-facing response concise and free of internal citation copy."""
    lines = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^根据《.+》中的相关规定：?$", line):
            continue
        if line == "请以页面下方展示的原文来源为准。":
            continue
        # A retrieved chunk or an imperfect model response may end with a
        # section heading such as “出差申请：”. It is not actionable without
        # following content, so never show a dangling heading to employees.
        if re.fullmatch(r"(?:[-•]\s*)?[\u4e00-\u9fffA-Za-z0-9（）()、·\s]{1,24}[：:]", line):
            continue
        # PDF/DOCX extraction may retain source numbering after a markdown
        # bullet marker (for example, “- 3.”). Keep the bullet but remove the
        # document's section number so employees read a clean ordered summary.
        line = re.sub(r"^(-\s*)\d+[.、)]\s*", r"\1", line)
        # The compact employee card renders escaped HTML rather than Markdown;
        # remove model emphasis markers so employees never see raw **text**.
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        lines.append(line)
    return "\n".join(lines).strip()


def conversation_turns_oldest_first(
    messages: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Pair user/assistant messages in natural conversation order."""
    turns: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "user":
            assistant = messages[index + 1] if index + 1 < len(messages) and messages[index + 1].get("role") == "assistant" else None
            turns.append((message, assistant))
            index += 2 if assistant else 1
        else:
            turns.append((message, None))
            index += 1
    return turns


def feedback_key_for_message(message: dict[str, Any], fallback_index: int) -> str:
    return f"message-{message.get('feedback_id') or fallback_index}"


def conversation_history_for_api(messages: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    """Send only the recent visible conversation, excluding UI-only metadata."""
    if limit <= 0:
        return []
    return [
        {"role": str(message.get("role")), "content": str(message.get("content", ""))}
        for message in messages[-limit:]
        if message.get("role") in {"user", "assistant"} and str(message.get("content", "")).strip()
    ]


def render_feedback_controls(api_url: str, question: str, answer: str, feedback_key: str) -> None:
    """Keep feedback available after Streamlit reruns without duplicate submissions."""
    if not question:
        return
    state_key = f"feedback-state-{feedback_key}"
    submitted = st.session_state.get(state_key)
    if submitted == "up":
        st.caption("已记录：回答有帮助")
        return
    if submitted == "down":
        st.caption("已记录：回答需要改进")
        return

    with st.container(key=f"assistant-feedback-{feedback_key}"):
        st.markdown('<p class="assistant-feedback-prompt">这条回复有帮助吗？</p>', unsafe_allow_html=True)
        feedback_cols = st.columns([1, 1, 6.2])
        if feedback_cols[0].button("👍", key=f"up-{feedback_key}", help="有帮助"):
            try:
                api_request("POST", api_url, "/feedback", json={"question": question, "answer": answer, "rating": "up"})
                st.session_state[state_key] = "up"
                feedback_cols[2].success("已记录")
                st.rerun()
            except requests.RequestException as exc:
                feedback_cols[2].error(f"反馈提交失败：{exc}")
        if feedback_cols[1].button("👎", key=f"down-{feedback_key}", help="仍需帮助"):
            st.session_state[f"show-reason-{feedback_key}"] = True

        if st.session_state.get(f"show-reason-{feedback_key}"):
            reason_col, submit_col = st.columns([4.7, 1.3], gap="small")
            reason = reason_col.selectbox(
                "问题原因", ["答非所问", "没有依据", "制度过期", "未解决"],
                key=f"reason-{feedback_key}", label_visibility="collapsed",
            )
            if submit_col.button("提交", key=f"submit-{feedback_key}", help="提交反馈"):
                try:
                    api_request("POST", api_url, "/feedback", json={"question": question, "answer": answer, "rating": "down", "reason": reason})
                    st.session_state[state_key] = "down"
                    st.session_state.pop(f"show-reason-{feedback_key}", None)
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"反馈提交失败：{exc}")


def render_assistant_launcher(current_page: str | None = None) -> bool:
    """Render the globally available employee assistant trigger."""
    icon_data = ""
    if MASCOT_PATH.is_file():
        icon_data = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""<style>div.st-key-assistant-launcher button {{ background-image: url('data:image/png;base64,{icon_data}'); }}</style>""",
        unsafe_allow_html=True,
    )
    opened = st.button(
        "问制度",
        key="assistant-launcher",
        help="打开员工制度服务台",
        type="secondary",
    )
    if opened:
        st.session_state["assistant_dialog_open"] = True
        if current_page is not None:
            st.session_state["assistant_dialog_page"] = current_page
    return opened


def render_assistant_dialog(api_url: str) -> None:
    """Open the compact employee Q&A dialog while preserving shared history."""

    @st.dialog("员工制度服务台", width="small")
    def _dialog() -> None:
        mascot_markup = ""
        if MASCOT_PATH.is_file():
            encoded_mascot = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
            mascot_markup = f'<img src="data:image/png;base64,{encoded_mascot}" alt="员工制度服务台" />'
        st.markdown(
            f"""
            <div class="assistant-dialog-head">
              {mascot_markup}
              <div>
                <strong>{COMPANY_NAME}员工服务台</strong>
                <small>制度查询 · 流程指引 · 员工服务</small>
              </div>
              <span class="assistant-dialog-status">在线服务</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="assistant-dialog-body"):
            render_chat_tab(api_url, show_sources=False, compact=True)

    _dialog()


def render_global_assistant(api_url: str, *, current_page: str | None = None) -> None:
    """Expose the assistant launcher on every employee-facing page."""
    opened = render_assistant_launcher(current_page) if current_page is not None else render_assistant_launcher()
    if current_page is not None and st.session_state.get("assistant_dialog_open", False):
        if st.session_state.get("assistant_dialog_page", current_page) != current_page:
            st.session_state["assistant_dialog_open"] = False
    if opened or st.session_state.get("assistant_dialog_open", False):
        render_assistant_dialog(api_url)


def render_scene_questions(scene: str, questions: list[str]) -> None:
    """Render scene-specific preset questions that open the global assistant."""
    st.markdown(
        '<p class="suggestion-caption">关于本模块，可以直接问员工助手</p>',
        unsafe_allow_html=True,
    )
    columns = st.columns(len(questions), gap="small")
    for index, question in enumerate(questions):
        if columns[index].button(question, key=f"scene-{scene}-{index}"):
            st.session_state.pending_question = question
            st.session_state["assistant_dialog_open"] = True
            st.session_state["assistant_dialog_page"] = scene


def render_source(source: dict[str, Any], index: int) -> None:
    title = source.get("title", "未命名文档")
    section = source.get("section", "正文")
    with st.expander(f"来源 {index}｜{title} · {section}"):
        st.caption(
            f"文件：{source.get('source_path', '-')}　"
            f"类型：{source.get('source_type', '-')}　"
            f"召回排名：{source.get('rank', '-') }"
        )
        st.write(source.get("content_preview", "暂无原文预览"))


def inject_workspace_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --canvas: #f5f5f7; --surface: rgba(255,255,255,.84); --ink: #1d1d1f;
          --muted: #6e6e73; --line: rgba(29,29,31,.14); --accent: #0071e3;
          --accent-soft: #eaf3ff; --success: #198754; --warning: #9a6700;
          --assistant-sky: #8ed3df; --assistant-pink: #d49ac2; --assistant-yellow: #f6d77a;
          --assistant-orange: #f4a340; --assistant-red: #ed3d32; --assistant-ink: #161616;
          --shadow: 0 16px 38px rgba(29,29,31,.08); --soft-shadow: 0 5px 18px rgba(29,29,31,.055);
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid=stHeader] { background: rgba(245,245,247,.82); border-bottom: 1px solid rgba(29,29,31,.08); backdrop-filter: blur(18px); }
        [data-testid=stSidebarCollapseButton],
        [data-testid=stSidebarCollapseButton] button { visibility: visible !important; opacity: 1 !important; }
        [data-testid=stSidebar][aria-expanded=false] [data-testid=stSidebarCollapseButton] {
          position: fixed !important; left: 8px; top: 44px; z-index: 9999;
          transform: translateX(300px) !important;
        }
        [data-testid=stDecoration] { display: none; }
        [data-testid=stToolbar], button[data-testid="stBaseButton-header"] { display: none; }
        .block-container { max-width: 1220px; padding: 3.25rem 2rem 5rem; }
        [data-testid=stSidebar] { background: rgba(255,255,255,.9); border-right: 1px solid var(--line); }
        [data-testid=stSidebar] > div:first-child { padding-top: 1.5rem; }
        [data-testid=stSidebar] .stMarkdown p, [data-testid=stSidebar] label { color: var(--muted); }
        [data-testid=stSidebar] .stTextInput input { background: #fff; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); }
        [data-testid=stSidebar] .stTextInput input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,113,227,.12); }
        .sidebar-brand-row { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
        .sidebar-logo { width: 46px; height: 46px; flex: 0 0 auto; border-radius: 11px; object-fit: cover; object-position: center; border: 1px solid rgba(29,29,31,.12); box-shadow: 0 6px 14px rgba(29,29,31,.1); background: #fff; }
        .sidebar-brand { color: var(--ink); font-size: 1.05rem; font-weight: 720; margin: 0; }
        .sidebar-label { color: var(--muted); font-size: .72rem; font-weight: 700; margin: 1rem 0 .45rem; }
        [data-testid=stSidebar] [data-testid=stRadio] > div { gap: .18rem; }
        [data-testid=stSidebar] [data-testid=stRadio] label > div:first-child { display: none; }
        [data-testid=stSidebar] [data-testid=stRadio] label { color: var(--ink); border-radius: 9px; padding: .58rem .7rem; transition: background .18s ease, color .18s ease, transform .18s ease; }
        [data-testid=stSidebar] [data-testid=stRadio] label:hover { background: #f3f6fb; transform: translateX(1px); }
        [data-testid=stSidebar] [data-testid=stRadio] label:has(input:checked) { color: #0b63ce; background: #eaf3ff; font-weight: 700; }
        [data-testid=stSidebar] [data-testid=stRadio] label:has(input:checked) p { color: #0b63ce; }
        div.st-key-sidebar-user-card { position: fixed; left: .65rem; bottom: .85rem; z-index: 20; width: min(16.5rem, calc(100vw - 1.3rem)); }
        div.st-key-sidebar-user-card .sidebar-user-card-content { margin: 0; padding: .72rem .75rem; border: 1px solid rgba(37,99,235,.16); border-radius: 10px; background: #fff; transition: background .18s ease, border-color .18s ease; pointer-events: none; }
        div.st-key-sidebar-user-card:hover .sidebar-user-card-content { background: #f8fbff; border-color: rgba(37,99,235,.28); }
        div.st-key-sidebar-user-card [data-testid="stPopoverButton"] { position: absolute; inset: 0; z-index: 2; width: 100%; height: 100%; min-height: 0; padding: 0; border: 0; border-radius: 10px; background: transparent; color: transparent; opacity: 0; cursor: pointer; }
        div.st-key-sidebar-user-card [data-testid="stPopoverButton"]:focus-visible { opacity: 1; outline: 2px solid rgba(37,99,235,.42); outline-offset: -2px; }
        .st-key-active_role { width: 100%; }
        .st-key-active_role [role="radiogroup"] { gap: 4px; }
        .st-key-active_role label[data-baseweb="radio"] { width: 100%; margin: 0; padding: 8px 12px; border-radius: 6px; transition: background .18s ease; }
        .st-key-active_role label[data-baseweb="radio"] > div:first-child { display: none; }
        .st-key-active_role label[data-baseweb="radio"] p { color: #334155; font-size: 14px; font-weight: 500; }
        .st-key-active_role label[data-baseweb="radio"]:hover { background: #f3f6fb; }
        .st-key-active_role label[data-baseweb="radio"]:has(input:checked) { background: #eaf3ff; }
        .st-key-active_role label[data-baseweb="radio"]:has(input:focus-visible) { outline: 2px solid #2563eb; }
        .sidebar-user { display: flex; align-items: center; gap: .55rem; }
        .sidebar-avatar { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 50%; color: #fff; background: var(--avatar-color, #2563eb); font-size: .92rem; font-weight: 700; box-shadow: 0 5px 12px color-mix(in srgb, var(--avatar-color, #2563eb) 28%, transparent); }
        .sidebar-user strong { display: block; color: #334155; font-size: .74rem; }
        .sidebar-user small { display: block; color: #94a3b8; font-size: .64rem; margin-top: .1rem; }
        .overview-hero { position: relative; overflow: hidden; min-height: 272px; margin: .15rem 0 1.15rem; padding: 2.1rem 2.25rem; border: 1px solid rgba(37,99,235,.16); border-radius: 18px; background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 58%, #eaf3ff 100%); box-shadow: 0 18px 42px rgba(29,78,216,.10); }
        .overview-hero::after { content: ""; position: absolute; width: 340px; height: 340px; right: -110px; top: -135px; border-radius: 50%; background: radial-gradient(circle, rgba(37,99,235,.16), rgba(37,99,235,0) 68%); pointer-events: none; }
        .overview-hero-content { position: relative; z-index: 1; max-width: 720px; }
        .overview-hero .overview-brand { color: #2563eb; font-size: .78rem; font-weight: 760; letter-spacing: .08em; margin: 0 0 .75rem; }
        .overview-company-name { display: block; color: #111827; font-size: clamp(2.45rem, 5vw, 4.9rem); font-weight: 780; line-height: 1; letter-spacing: -.055em; margin-bottom: .55rem; }
        .overview-hero h1 { color: #334155; font-size: clamp(1.35rem, 2.2vw, 2rem); line-height: 1.2; letter-spacing: -.035em; font-weight: 690; margin: 0; }
        .overview-hero .overview-lead { color: #475569; font-size: 1.05rem; line-height: 1.7; max-width: 43rem; margin: 1rem 0 1.2rem; }
        .overview-hero .overview-meta { display: flex; flex-wrap: wrap; gap: .5rem; color: #52637a; font-size: .78rem; }
        .overview-meta span { border: 1px solid rgba(37,99,235,.18); border-radius: 999px; background: rgba(255,255,255,.72); padding: .38rem .65rem; }
        .overview-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; margin-bottom: 1.35rem; }
        .overview-kpi { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.82); padding: 1rem 1.05rem; box-shadow: var(--soft-shadow); }
        .overview-kpi span { display: block; color: var(--muted); font-size: .76rem; }
        .overview-kpi strong { display: block; color: #111827; font-size: 1.45rem; letter-spacing: -.02em; margin-top: .35rem; }
        .overview-kpi em { display: block; color: #64748b; font-size: .7rem; font-style: normal; margin-top: .2rem; }
        .overview-section { margin-top: 1.3rem; }
        .overview-section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; border-top: 1px solid var(--line); padding: 1.25rem 0 .8rem; }
        .overview-section-heading h2 { color: #111827; font-size: 1.45rem; letter-spacing: -.025em; margin: 0; }
        .overview-section-heading p { color: var(--muted); font-size: .84rem; margin: .3rem 0 0; }
        .overview-grid { display: grid; grid-template-columns: 1.18fr .82fr; gap: .8rem; }
        .overview-panel { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.78); padding: 1.15rem 1.2rem; box-shadow: var(--soft-shadow); }
        .overview-panel h3 { color: #1f2937; font-size: 1rem; margin: 0 0 .55rem; }
        .overview-panel p { color: #52606f; font-size: .88rem; line-height: 1.7; margin: .4rem 0; }
        .overview-timeline { display: grid; gap: .7rem; }
        .overview-timeline-item { display: grid; grid-template-columns: 62px 1fr; gap: .7rem; align-items: start; }
        .overview-timeline-year { color: #2563eb; font-size: .78rem; font-weight: 760; padding-top: .14rem; }
        .overview-timeline-copy { color: #52606f; font-size: .84rem; line-height: 1.5; border-left: 2px solid #dbeafe; padding-left: .75rem; }
        .overview-products { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; }
        .overview-product { min-height: 118px; border: 1px solid rgba(37,99,235,.15); border-radius: 12px; background: linear-gradient(160deg, #fff, #f7faff); padding: .95rem; }
        .overview-product .product-index { color: #2563eb; font-size: .7rem; font-weight: 760; letter-spacing: .08em; }
        .overview-product h3 { color: #1f2937; font-size: .98rem; margin: .45rem 0 .35rem; }
        .overview-product p { color: #64748b; font-size: .79rem; line-height: 1.5; margin: 0; }
        .overview-achievements { display: grid; grid-template-columns: repeat(2, 1fr); gap: .55rem .8rem; }
        .overview-achievement { display: flex; gap: .55rem; align-items: flex-start; color: #52606f; font-size: .84rem; line-height: 1.5; }
        .overview-achievement::before { content: "✓"; color: #1677ff; font-weight: 800; }
        .dashboard-topbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: .15rem 0 1rem; }
        .dashboard-breadcrumb { color: #64748b; font-size: .78rem; }
        .dashboard-breadcrumb strong { color: #0f172a; font-size: 1.25rem; margin-right: .55rem; }
        .dashboard-toolbar-actions { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; justify-content: flex-end; }
        .dashboard-toolbar-chip, .dashboard-toolbar-button { border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.78); color: #475569; font-size: .76rem; padding: .48rem .65rem; }
        .dashboard-toolbar-button { color: #2563eb; cursor: pointer; transition: background .18s ease, transform .18s ease; }
        .dashboard-toolbar-button:hover { background: #eaf3ff; transform: translateY(-1px); }
        .dashboard-hero { display: grid; grid-template-columns: 1.15fr .85fr; gap: 1rem; position: relative; overflow: hidden; min-height: 150px; margin-bottom: .8rem; padding: 1.1rem 1.4rem; border: 1px solid rgba(37,99,235,.16); border-radius: 16px; background: linear-gradient(120deg, #fff 0%, #f3f7ff 66%, #eaf3ff 100%); box-shadow: 0 15px 34px rgba(37,99,235,.08); }
        .dashboard-hero::after { content: ""; position: absolute; width: 320px; height: 320px; right: -100px; top: -120px; border-radius: 50%; background: radial-gradient(circle, rgba(14,165,233,.14), rgba(14,165,233,0) 70%); pointer-events: none; }
        .dashboard-hero-copy { position: relative; z-index: 1; }
        .dashboard-kicker { color: #2563eb; font-size: .72rem; font-weight: 760; letter-spacing: .08em; margin: 0 0 .5rem; }
        .dashboard-hero h1 { color: #0f172a; font-size: clamp(1.6rem, 2.35vw, 2.05rem); letter-spacing: -.04em; line-height: 1.15; margin: 0; white-space: nowrap; }
        .dashboard-hero p { color: #52606f; font-size: .86rem; line-height: 1.5; margin: .45rem 0 .65rem; max-width: 38rem; }
        .dashboard-hero-actions { display: flex; gap: .45rem; flex-wrap: wrap; }
        .dashboard-primary { border: 0; border-radius: 8px; background: #2563eb; color: #fff; font-size: .78rem; font-weight: 650; padding: .55rem .8rem; }
        .dashboard-secondary { border: 1px solid rgba(37,99,235,.2); border-radius: 8px; background: rgba(255,255,255,.7); color: #2563eb; font-size: .78rem; padding: .52rem .78rem; }
        .dashboard-hero-visual { position: relative; z-index: 1; min-height: 120px; display: grid; place-items: center; }
        .dashboard-mascot-glow { position: absolute; width: 210px; height: 210px; border-radius: 50%; background: radial-gradient(circle, rgba(37,99,235,.18), rgba(37,99,235,0) 68%); }
        .dashboard-mascot { position: relative; width: 116px; height: 116px; object-fit: cover; object-position: center 28%; border-radius: 28px; border: 6px solid rgba(255,255,255,.78); box-shadow: 0 14px 28px rgba(37,99,235,.2); transform: rotate(3deg); }
        .dashboard-orbit { width: 156px; height: 156px; border: 1px solid rgba(37,99,235,.2); border-radius: 50%; position: relative; box-shadow: 0 0 0 18px rgba(37,99,235,.035), 0 0 0 38px rgba(37,99,235,.025); }
        .dashboard-orbit::before, .dashboard-orbit::after { content: ""; position: absolute; inset: 22px; border: 1px dashed rgba(14,165,233,.28); border-radius: 50%; }
        .dashboard-orbit::after { inset: 50px; border: 0; background: linear-gradient(145deg, #2563eb, #06b6d4); box-shadow: 0 10px 25px rgba(37,99,235,.3); }
        .dashboard-orbit-dot { position: absolute; width: 9px; height: 9px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 0 5px rgba(245,158,11,.12); }
        .dashboard-orbit-dot.one { left: 8px; top: 58px; }
        .dashboard-orbit-dot.two { right: 15px; top: 25px; background: #06b6d4; }
        .dashboard-orbit-dot.three { right: 22px; bottom: 22px; background: #8b5cf6; }
        .dashboard-section { margin-top: 1rem; }
        .dashboard-section-title { display: flex; align-items: center; justify-content: space-between; gap: .8rem; margin-bottom: .62rem; }
        .dashboard-section-title h2 { color: #0f172a; font-size: 1.08rem; letter-spacing: -.015em; margin: 0; }
        .dashboard-section-title p { color: #94a3b8; font-size: .74rem; margin: .2rem 0 0; }
        .dashboard-link { color: #2563eb; font-size: .76rem; text-decoration: none; }
        .dashboard-link:hover { text-decoration: underline; }
        .dashboard-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; }
        .dashboard-metric { position: relative; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.78); padding: .9rem .95rem; box-shadow: var(--soft-shadow); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; cursor: pointer; }
        .dashboard-metric:hover, .service-card:hover, .dashboard-panel:hover { border-color: rgba(37,99,235,.34); transform: translateY(-2px); box-shadow: 0 10px 24px rgba(37,99,235,.11); }
        .dashboard-metric-head { display: flex; align-items: center; justify-content: space-between; gap: .5rem; }
        .dashboard-metric-label { color: #64748b; font-size: .75rem; }
        .metric-icon { display: grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; color: #2563eb; background: #eaf3ff; font-size: .84rem; }
        .metric-icon.cyan { color: #0891b2; background: #ecfeff; }
        .metric-icon.green { color: #15803d; background: #ecfdf5; }
        .metric-icon.amber { color: #b45309; background: #fffbeb; }
        .dashboard-metric strong { display: block; color: #0f172a; font-size: 1.45rem; letter-spacing: -.025em; margin-top: .55rem; }
        .dashboard-metric-foot { display: flex; align-items: center; justify-content: space-between; gap: .4rem; margin-top: .32rem; }
        .dashboard-trend { color: #15803d; font-size: .7rem; }
        .dashboard-trend.down { color: #b45309; }
        .dashboard-status { color: #15803d; font-size: .7rem; }
        .dashboard-columns { display: grid; grid-template-columns: 1.05fr .95fr; gap: .75rem; }
        .dashboard-panel { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.76); padding: 1rem; box-shadow: var(--soft-shadow); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
        .service-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
        .service-card { display: flex; align-items: center; gap: .5rem; min-height: 60px; border: 1px solid rgba(148,163,184,.2); border-radius: 9px; background: rgba(248,250,252,.75); padding: .55rem; cursor: pointer; transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
        .service-icon { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 8px; color: #2563eb; background: #eaf3ff; font-size: .82rem; flex: 0 0 auto; }
        .service-card strong { color: #334155; font-size: .76rem; font-weight: 650; }
        .service-card small { display: block; color: #94a3b8; font-size: .65rem; margin-top: .15rem; }
        .todo-list { display: grid; gap: .35rem; }
        .todo-item { display: grid; grid-template-columns: 7px 1fr auto; align-items: center; gap: .55rem; border-bottom: 1px solid rgba(148,163,184,.14); padding: .42rem 0; }
        .todo-item:last-child { border-bottom: 0; }
        .todo-dot { width: 7px; height: 7px; border-radius: 50%; background: #2563eb; }
        .todo-dot.amber { background: #f59e0b; }
        .todo-dot.green { background: #22c55e; }
        .todo-copy strong { display: block; color: #334155; font-size: .76rem; font-weight: 650; }
        .todo-copy small { display: block; color: #94a3b8; font-size: .65rem; margin-top: .12rem; }
        .todo-tag { color: #2563eb; background: #eaf3ff; border-radius: 999px; font-size: .62rem; padding: .2rem .42rem; white-space: nowrap; }
        .todo-tag.amber { color: #b45309; background: #fffbeb; }
        .cockpit-grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: .75rem; }
        .chart-panel h3 { color: #334155; font-size: .82rem; margin: 0; }
        .chart-subtitle { color: #94a3b8; font-size: .68rem; margin: .22rem 0 .75rem; }
        .chart-bars { height: 112px; display: flex; align-items: end; gap: .7rem; border-bottom: 1px solid #e2e8f0; padding: 0 .2rem; }
        .chart-bar { flex: 1; min-width: 14px; border-radius: 5px 5px 0 0; background: linear-gradient(180deg, #60a5fa, #2563eb); position: relative; }
        .chart-bar.alt { background: linear-gradient(180deg, #67e8f9, #0891b2); }
        .chart-bar span { position: absolute; left: 50%; transform: translateX(-50%); bottom: -1.2rem; color: #94a3b8; font-size: .62rem; }
        .cockpit-kpis { display: grid; grid-template-columns: repeat(2, 1fr); gap: .45rem; margin-top: .75rem; }
        .cockpit-kpi { border-radius: 8px; background: #f8fafc; padding: .55rem .6rem; }
        .cockpit-kpi span { display: block; color: #94a3b8; font-size: .64rem; }
        .cockpit-kpi strong { display: block; color: #334155; font-size: .9rem; margin-top: .18rem; }
        .donut-wrap { display: flex; align-items: center; gap: 1rem; }
        .donut { width: 108px; height: 108px; border-radius: 50%; background: conic-gradient(#2563eb 0 58%, #06b6d4 58% 82%, #cbd5e1 82% 100%); display: grid; place-items: center; flex: 0 0 auto; }
        .donut::after { content: ""; width: 66px; height: 66px; border-radius: 50%; background: #fff; }
        .legend { display: grid; gap: .45rem; }
        .legend-item { display: flex; align-items: center; gap: .38rem; color: #64748b; font-size: .7rem; }
        .legend-dot { width: 7px; height: 7px; border-radius: 50%; background: #2563eb; }
        .legend-dot.cyan { background: #06b6d4; }
        .legend-dot.gray { background: #cbd5e1; }
        .risk-list { display: grid; gap: .45rem; margin-top: .8rem; }
        .risk-item { display: flex; align-items: start; gap: .45rem; color: #64748b; font-size: .7rem; line-height: 1.45; }
        .risk-badge { color: #b45309; background: #fffbeb; border-radius: 999px; padding: .16rem .35rem; font-size: .6rem; white-space: nowrap; }
        .activity-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
        .timeline, .announcement-list { display: grid; gap: .45rem; }
        .timeline-item { display: grid; grid-template-columns: 42px 1fr; gap: .55rem; align-items: start; padding: .35rem 0; border-bottom: 1px solid rgba(148,163,184,.14); }
        .timeline-item:last-child, .announcement-item:last-child { border-bottom: 0; }
        .timeline-time { color: #2563eb; font-size: .65rem; font-weight: 700; }
        .timeline-copy { color: #52606f; font-size: .72rem; line-height: 1.45; }
        .timeline-copy small { display: block; color: #94a3b8; font-size: .62rem; margin-top: .12rem; }
        .announcement-item { display: grid; grid-template-columns: 7px 1fr auto; gap: .5rem; align-items: center; padding: .42rem 0; border-bottom: 1px solid rgba(148,163,184,.14); }
        .announcement-pin { width: 7px; height: 7px; border-radius: 50%; background: #2563eb; }
        .announcement-copy { color: #52606f; font-size: .72rem; line-height: 1.45; }
        .announcement-copy small { display: block; color: #94a3b8; font-size: .62rem; margin-top: .12rem; }
        .announcement-unread { color: #2563eb; font-size: .6rem; white-space: nowrap; }
        .dashboard-note { color: #94a3b8; font-size: .68rem; text-align: right; margin-top: .8rem; }
        .assistant-launcher { position: fixed; width: 1px; height: 1px; overflow: hidden; }
        div.st-key-assistant-launcher { position: fixed !important; right: 1.25rem !important; bottom: 1.25rem !important; z-index: 1000 !important; width: 68px !important; height: 68px !important; }
        div.st-key-assistant-launcher > div { width: 68px !important; height: 68px !important; }
        div.st-key-assistant-launcher button { width: 68px !important; height: 68px !important; min-height: 68px !important; padding: 0 !important; border: 3px solid rgba(255,255,255,.96) !important; border-radius: 50% !important; color: transparent !important; font-size: 0 !important; background-color: var(--assistant-sky) !important; background-position: center 28% !important; background-repeat: no-repeat !important; background-size: cover !important; box-shadow: 0 10px 24px rgba(22,22,22,.18), 0 0 0 5px rgba(142,211,223,.32) !important; cursor: pointer !important; transition: transform .18s ease, box-shadow .18s ease !important; }
        div.st-key-assistant-launcher button:hover, div.st-key-assistant-launcher button:focus-visible { transform: translateY(-3px) scale(1.03) !important; box-shadow: 0 14px 30px rgba(22,22,22,.22), 0 0 0 6px rgba(212,154,194,.28) !important; }
        div.st-key-assistant-launcher button > div { display: none !important; }
        [data-baseweb="modal"] { align-items: flex-end !important; justify-content: flex-end !important; padding: 1.25rem !important; background: rgba(15,23,42,.08) !important; }
        [data-testid="stDialog"] > div { align-items: flex-end !important; justify-content: flex-end !important; height: 100% !important; width: 100% !important; padding: 0 !important; }
        [data-baseweb="modal"] [role="dialog"] { width: min(460px, calc(100vw - 2rem)) !important; max-width: min(460px, calc(100vw - 2rem)) !important; max-height: min(720px, calc(100vh - 2rem)) !important; margin: 0 !important; border: 1px solid #dbe3ee !important; border-radius: 16px !important; background: #f7f9fc !important; box-shadow: 0 24px 60px rgba(22,34,51,.18) !important; overflow: hidden !important; }
        [data-baseweb="modal"] [role="dialog"] > div:nth-child(2) { display: flex !important; flex: 1 1 auto !important; flex-direction: column !important; min-height: 0 !important; max-height: calc(min(720px, 100vh - 2rem) - 78px) !important; overflow-y: auto !important; overflow-x: hidden !important; overscroll-behavior: contain !important; scrollbar-color: rgba(142,211,223,.9) transparent !important; scrollbar-width: thin !important; }
        [data-baseweb="modal"] [role="dialog"] > div:nth-child(2) > div { min-height: 0 !important; }
        [data-baseweb="modal"] [role="dialog"] .assistant-dialog-body,
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body { min-height: 0 !important; width: 100% !important; padding-bottom: 0 !important; flex: 1 1 auto !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body:has(#assistant-latest-turn)::after { content: ""; display: block; flex: 0 0 min(13rem, 32vh); order: 9; pointer-events: none; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-composer { order: 10 !important; position: sticky !important; bottom: 0 !important; z-index: 4 !important; padding-top: .35rem !important; background: #f8fcfd !important; }
        .st-key-assistant-dialog-body > [data-testid="stLayoutWrapper"]:has(> .st-key-assistant-composer) { order: 10; position: sticky; bottom: 0; z-index: 4; margin-top: auto; background: #f8fcfd; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-composer { margin: .4rem -.35rem -.35rem; padding: .65rem .55rem .72rem !important; border-top: 1px solid #dbe3ee; background: rgba(255,255,255,.96) !important; box-shadow: 0 -8px 20px rgba(31,55,86,.04); }
        .st-key-assistant-composer [data-testid="InputInstructions"], .st-key-chat-composer [data-testid="InputInstructions"] { display: none; }
        .st-key-assistant-composer [data-testid="stHorizontalBlock"], .st-key-chat-composer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; align-items: center; gap: 8px; }
        .st-key-assistant-composer [data-testid="stColumn"], .st-key-chat-composer [data-testid="stColumn"] { min-width: 0; }
        .st-key-assistant-composer [data-testid="stColumn"]:last-child, .st-key-chat-composer [data-testid="stColumn"]:last-child { flex: 0 0 40px; width: 40px; }
        .st-key-assistant-composer [data-testid="stFormSubmitButton"] button, .st-key-chat-composer [data-testid="stFormSubmitButton"] button { width: 40px; height: 40px; min-width: 40px; padding: 0; border-radius: 8px; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-composer input { border: 1px solid #b9c7d8 !important; border-radius: 8px !important; background: #fff !important; color: #172b4d !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-composer input:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-composer [data-testid="stFormSubmitButton"] button { background: #2563eb !important; border-color: #2563eb !important; color: #fff !important; }
        [data-baseweb="modal"] [role="dialog"] .assistant-dialog-body,
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body,
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body *:not([data-testid="stIconMaterial"]) { font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", Inter, sans-serif !important; }
        [data-baseweb="modal"] [role="dialog"] .assistant-dialog-body .answer-panel,
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body .answer-panel { font-size: .9rem !important; line-height: 1.7 !important; border-radius: 12px !important; border-color: #dbe3ee !important; background: #fff !important; box-shadow: 0 5px 14px rgba(30,58,95,.07) !important; padding: .9rem 1rem !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stChatMessageContent"] { color: var(--assistant-ink) !important; font-size: .9rem !important; line-height: 1.68 !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stChatMessage"] { border-top-color: rgba(22,22,22,.08) !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stChatMessageAvatarUser"] { background: #e8d5e8 !important; color: #49324f !important; border: 1px solid #c9a8c9 !important; border-radius: 8px !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stChatMessageAvatarAssistant"] { background: #eaf3ff !important; border: 1px solid #b9c7d8 !important; border-radius: 8px !important; overflow: hidden !important; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stChatMessage"] { padding: .7rem 0 !important; }
        .assistant-dialog-head { display: flex; align-items: center; gap: .65rem; padding: .1rem 0 .8rem; border-bottom: 1px solid #d9e0e8; }
        .assistant-dialog-head img { width: 40px; height: 40px; object-fit: cover; object-position: center 28%; border-radius: 8px; border: 1px solid #b9c7d8; background: #eaf3ff; }
        .assistant-dialog-head strong { display: block; color: #172b4d; font-size: 1rem; letter-spacing: .01em; }
        .assistant-dialog-head small { display: block; color: #6b778c; font-size: .72rem; margin-top: .22rem; }
        .assistant-dialog-status { margin-left: auto; color: #1d4ed8; background: #eaf3ff; border: 1px solid #bfdbfe; border-radius: 999px; padding: .2rem .45rem; font-size: .62rem; white-space: nowrap; }
        .assistant-dialog-intro { color: #7b8794; font-size: .66rem; font-weight: 600; line-height: 1.4; margin: 0 0 .32rem; }
        #assistant-latest-turn { scroll-margin-top: 10px; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body .answer-overline { color: #2563eb; font-size: .66rem; font-weight: 700; letter-spacing: .04em; margin: 0 0 .38rem; }
        [data-baseweb="modal"] [role="dialog"] .st-key-assistant-dialog-body [data-testid="stButton"] button { min-height: 30px; padding: .2rem .48rem; border-radius: 7px; border-color: #dbe3ee; background: #fff; color: #52606f; font-size: .68rem; }
        [data-baseweb="modal"] [role="dialog"] [class*="st-key-assistant-feedback-"] { margin: -.1rem 0 .5rem 2.85rem; padding: .38rem .48rem; border: 1px solid #e4eaf1; border-radius: 8px; background: rgba(255,255,255,.76); }
        [data-baseweb="modal"] [role="dialog"] .assistant-feedback-prompt { margin: 0 0 .22rem; color: #8a94a6; font-size: .62rem; line-height: 1.3; }
        [data-baseweb="modal"] [role="dialog"] [class*="st-key-assistant-feedback-"] [data-testid="stButton"] button { width: 29px; min-width: 29px; min-height: 27px; padding: 0; border-radius: 6px; color: #52606f; font-size: .76rem; line-height: 1; }
        [data-baseweb="modal"] [role="dialog"] [class*="st-key-assistant-feedback-"] [data-testid="stButton"] button:hover { color: #1d4ed8; border-color: #93c5fd; background: #eff6ff; }
        [data-baseweb="modal"] [role="dialog"] [class*="st-key-assistant-feedback-"] [data-baseweb="select"] > div { min-height: 30px; border-color: #dbe3ee; background: #fff; font-size: .72rem; }
        .assistant-page-guide { display: grid; gap: .55rem; max-width: 620px; margin-top: 1rem; border: 1px solid rgba(37,99,235,.14); border-radius: 14px; background: rgba(255,255,255,.78); padding: 1.1rem 1.2rem; box-shadow: var(--soft-shadow); }
        .assistant-page-guide strong { color: #1e293b; font-size: .9rem; }
        .assistant-page-guide p { color: #64748b; font-size: .78rem; line-height: 1.6; margin: 0; }
        .assistant-page-guide span { color: #2563eb; font-size: .72rem; font-weight: 700; }
        .erp-shell { display: grid; gap: .85rem; }
        .erp-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .9rem 1rem; border: 1px solid #dbe3ee; border-radius: 10px; background: #fff; box-shadow: var(--soft-shadow); }
        .erp-header-title { color: #0f172a; font-size: 1.02rem; font-weight: 720; }
        .erp-header-subtitle { color: #94a3b8; font-size: .7rem; margin-top: .2rem; }
        .erp-header-tools { display: flex; align-items: center; gap: .45rem; color: #64748b; font-size: .72rem; }
        .erp-period, .erp-org { border: 1px solid #dbe3ee; border-radius: 7px; background: #f8fafc; padding: .45rem .6rem; white-space: nowrap; }
        .erp-period strong, .erp-org strong { color: #334155; font-weight: 650; }
        .erp-kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; }
        .erp-kpi { position: relative; overflow: hidden; border: 1px solid #dbe3ee; border-radius: 10px; background: #fff; padding: .9rem; box-shadow: var(--soft-shadow); }
        .erp-kpi::after { content: ""; position: absolute; width: 86px; height: 86px; right: -34px; top: -38px; border-radius: 50%; background: rgba(37,99,235,.07); }
        .erp-kpi-label { display: block; color: #64748b; font-size: .7rem; }
        .erp-kpi strong { display: block; color: #0f172a; font-size: 1.35rem; letter-spacing: -.025em; margin-top: .38rem; }
        .erp-kpi-foot { display: flex; justify-content: space-between; gap: .45rem; margin-top: .35rem; color: #94a3b8; font-size: .64rem; }
        .erp-positive { color: #15803d; }
        .erp-warning { color: #b45309; }
        .erp-shortcuts { display: grid; grid-template-columns: repeat(5, 1fr); gap: .55rem; }
        .erp-shortcut { display: flex; align-items: center; gap: .5rem; min-height: 58px; border: 1px solid #dbe3ee; border-radius: 9px; background: #fff; padding: .55rem .65rem; box-shadow: var(--soft-shadow); }
        .erp-shortcut-mark { display: grid; place-items: center; width: 27px; height: 27px; border-radius: 7px; color: #2563eb; background: #eaf3ff; font-size: .7rem; font-weight: 760; flex: 0 0 auto; }
        .erp-shortcut strong { display: block; color: #334155; font-size: .72rem; font-weight: 650; }
        .erp-shortcut small { display: block; color: #94a3b8; font-size: .61rem; margin-top: .12rem; }
        .erp-main-grid, .erp-bottom-grid { display: grid; grid-template-columns: 1.2fr .8fr; gap: .75rem; }
        .erp-bottom-grid { grid-template-columns: 1fr 1fr; }
        .erp-panel { border: 1px solid #dbe3ee; border-radius: 10px; background: #fff; padding: .95rem 1rem; box-shadow: var(--soft-shadow); }
        .erp-panel-head { display: flex; align-items: center; justify-content: space-between; gap: .65rem; border-bottom: 1px solid #edf0f4; padding-bottom: .6rem; margin-bottom: .45rem; }
        .erp-panel-head h2 { color: #1e293b; font-size: .92rem; margin: 0; }
        .erp-panel-head span { color: #94a3b8; font-size: .64rem; }
        .erp-table { width: 100%; border-collapse: collapse; font-size: .7rem; }
        .erp-table th, .erp-table td { border-bottom: 1px solid #edf0f4; padding: .55rem .35rem; text-align: left; white-space: nowrap; }
        .erp-table th { color: #64748b; background: #f8fafc; font-weight: 650; }
        .erp-table td { color: #475569; }
        .erp-table tr:last-child td { border-bottom: 0; }
        .erp-table .amount { color: #0f172a; font-variant-numeric: tabular-nums; font-weight: 650; }
        .erp-status { display: inline-block; border-radius: 999px; padding: .17rem .4rem; font-size: .6rem; }
        .erp-status.pending { color: #b45309; background: #fffbeb; }
        .erp-status.processing { color: #1d4ed8; background: #eaf3ff; }
        .erp-status.done { color: #166534; background: #dcfce7; }
        .erp-status.paused { color: #64748b; background: #f1f5f9; }
        .erp-workflow { display: grid; gap: .62rem; }
        .erp-workflow-row { display: grid; grid-template-columns: 1fr auto; gap: .55rem; align-items: start; padding-bottom: .58rem; border-bottom: 1px solid #edf0f4; }
        .erp-workflow-row:last-child { border-bottom: 0; padding-bottom: 0; }
        .erp-workflow-row strong { color: #334155; display: block; font-size: .72rem; font-weight: 650; }
        .erp-workflow-row small { color: #94a3b8; display: block; font-size: .62rem; margin-top: .16rem; }
        .erp-workflow-step { display: flex; align-items: center; gap: .25rem; margin-top: .4rem; }
        .erp-workflow-step i { display: block; height: 4px; flex: 1; border-radius: 999px; background: #e2e8f0; }
        .erp-workflow-step i.done { background: #60a5fa; }
        .erp-workflow-step i.current { background: #2563eb; }
        .erp-inventory-list { display: grid; gap: .5rem; }
        .erp-inventory-row { display: grid; grid-template-columns: 1fr 76px 58px; gap: .45rem; align-items: center; padding: .45rem 0; border-bottom: 1px solid #edf0f4; }
        .erp-inventory-row:last-child { border-bottom: 0; }
        .erp-inventory-row strong { color: #475569; font-size: .7rem; font-weight: 650; }
        .erp-inventory-row small { color: #94a3b8; display: block; font-size: .6rem; margin-top: .12rem; }
        .erp-stock-bar { height: 6px; overflow: hidden; border-radius: 999px; background: #e2e8f0; }
        .erp-stock-bar i { display: block; height: 100%; border-radius: inherit; background: #60a5fa; }
        .erp-stock-bar i.low { background: #f59e0b; }
        .erp-stock-count { color: #475569; font-size: .65rem; text-align: right; }
        .erp-note { color: #94a3b8; font-size: .66rem; text-align: right; margin: 0; }
        .overview-triad { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
        .overview-triad-card { min-height: 240px; border: 1px solid var(--line); border-radius: 13px; background: rgba(255,255,255,.78); padding: 1.35rem; box-shadow: var(--soft-shadow); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; cursor: pointer; }
        .overview-triad-card:hover { border-color: rgba(37,99,235,.34); transform: translateY(-2px); box-shadow: 0 10px 24px rgba(37,99,235,.11); }
        .overview-triad-card.notice { background: linear-gradient(145deg, #ffffff, #f4f8ff); }
        .overview-triad-card.news { color: #fff; border-color: rgba(129,140,248,.3); background: radial-gradient(circle at 90% 8%, rgba(79,70,229,.65), transparent 42%), linear-gradient(145deg, #111827, #1e1b4b); }
        .overview-triad-card .triad-kicker { color: #64748b; font-size: .66rem; font-weight: 760; letter-spacing: .08em; }
        .overview-triad-card.news .triad-kicker { color: #c4b5fd; }
        .overview-triad-card h3 { color: #1e293b; font-size: 1.18rem; line-height: 1.4; margin: .72rem 0 .55rem; }
        .overview-triad-card.news h3 { color: #fff; }
        .overview-triad-card p { color: #64748b; font-size: .84rem; line-height: 1.65; margin: 0; }
        .overview-triad-card.news p { color: #c7d2fe; }
        .triad-meta { display: flex; justify-content: space-between; gap: .5rem; color: #94a3b8; font-size: .65rem; margin-top: .8rem; }
        .overview-triad-card.news .triad-meta { color: #a5b4fc; }
        .notice-list { display: grid; gap: .44rem; margin-top: .35rem; }
        .notice-row { display: grid; grid-template-columns: 7px 1fr auto; gap: .45rem; align-items: center; border-bottom: 1px solid rgba(148,163,184,.14); padding: .34rem 0; }
        .notice-row:last-child { border-bottom: 0; }
        .notice-dot { width: 7px; height: 7px; border-radius: 50%; background: #2563eb; }
        .notice-row strong { color: #475569; font-size: .74rem; line-height: 1.4; font-weight: 650; }
        .notice-row small { color: #94a3b8; font-size: .62rem; white-space: nowrap; }
        .personal-layout { display: grid; grid-template-columns: .9fr 1.1fr; gap: .75rem; }
        .personal-profile { border: 1px solid rgba(37,99,235,.17); border-radius: 15px; background: linear-gradient(145deg, #eff6ff, #fff); padding: 1.25rem; box-shadow: var(--soft-shadow); }
        .personal-profile-head { display: flex; align-items: center; gap: .8rem; }
        .personal-avatar { width: 74px; height: 74px; border-radius: 22px; object-fit: cover; object-position: center 28%; border: 1px solid rgba(37,99,235,.17); box-shadow: 0 8px 18px rgba(37,99,235,.16); }
        .personal-avatar-initial { display: grid; place-items: center; color: #fff; background: var(--avatar-color, #2563eb); font-size: 1.55rem; font-weight: 750; }
        .personal-profile h2 { color: #0f172a; font-size: 1.28rem; margin: 0; }
        .personal-profile p { color: #64748b; font-size: .76rem; margin: .25rem 0 0; }
        .joining-badge { display: inline-flex; align-items: baseline; gap: .28rem; margin: 1rem 0 .7rem; color: #1d4ed8; background: rgba(255,255,255,.72); border: 1px solid rgba(37,99,235,.15); border-radius: 10px; padding: .55rem .7rem; }
        .joining-badge strong { font-size: 1.45rem; }
        .joining-badge span { color: #64748b; font-size: .9rem; }
        .profile-facts { display: grid; grid-template-columns: repeat(2, 1fr); gap: .45rem; }
        .profile-fact { border-top: 1px solid rgba(148,163,184,.17); padding-top: .45rem; }
        .profile-fact span { display: block; color: #94a3b8; font-size: .64rem; }
        .profile-fact strong { display: block; color: #475569; font-size: .76rem; margin-top: .16rem; }
        .detail-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .7rem; }
        .detail-card { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.78); padding: 1rem; box-shadow: var(--soft-shadow); }
        .detail-card h3 { color: #334155; font-size: .9rem; margin: 0 0 .65rem; }
        .detail-card p { color: #64748b; font-size: .74rem; line-height: 1.55; margin: .35rem 0; }
        .task-list, .workflow-list { display: grid; gap: .42rem; }
        .task-row { display: flex; align-items: center; justify-content: space-between; gap: .6rem; border-bottom: 1px solid rgba(148,163,184,.14); padding: .4rem 0; }
        .task-row:last-child { border-bottom: 0; }
        .task-row strong { color: #475569; font-size: .74rem; font-weight: 650; }
        .task-row small { color: #94a3b8; font-size: .64rem; }
        .task-chip { color: #2563eb; background: #eaf3ff; border-radius: 999px; font-size: .62rem; padding: .18rem .4rem; white-space: nowrap; }
        .task-chip.amber { color: #b45309; background: #fffbeb; }
        .workflow-item { border-bottom: 1px solid rgba(148,163,184,.14); padding: .5rem 0 .6rem; }
        .workflow-item:last-child { border-bottom: 0; }
        .workflow-head { display: flex; align-items: center; justify-content: space-between; gap: .6rem; }
        .workflow-head strong { color: #475569; font-size: .76rem; }
        .workflow-head small { color: #94a3b8; font-size: .64rem; }
        .workflow-track { display: flex; align-items: center; gap: .28rem; margin-top: .55rem; }
        .workflow-step { height: 5px; flex: 1; border-radius: 999px; background: #e2e8f0; }
        .workflow-step.done { background: #60a5fa; }
        .workflow-step.current { background: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,.1); }
        .finance-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: .55rem; margin-bottom: .8rem; }
        .finance-summary-card { border: 1px solid #e2e8f0; border-radius: 9px; background: #fff; padding: .7rem; }
        .finance-summary-card span { display: block; color: #64748b; font-size: .68rem; }
        .finance-summary-card strong { display: block; color: #0f172a; font-size: 1.05rem; margin-top: .25rem; }
        .portal-shell { position: relative; max-width: 1440px; margin: 0 auto; }
        .portal-shell > *:not(.portal-backdrop) { position: relative; z-index: 1; }
        .portal-backdrop { position: absolute; z-index: 0; top: -2.2rem; right: -1.8rem; width: 53%; height: 600px; background-size: cover; background-position: center 18%; opacity: .42; filter: blur(2px); transform: scale(1.02); pointer-events: none; -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.18) 8%, #000 19%, #000 84%, transparent 100%), linear-gradient(180deg, #000 0%, rgba(0,0,0,.94) 54%, transparent 70%); -webkit-mask-composite: source-in; mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.18) 8%, #000 19%, #000 84%, transparent 100%), linear-gradient(180deg, #000 0%, rgba(0,0,0,.94) 54%, transparent 70%); mask-composite: intersect; }
        .portal-brandbar { display: flex; align-items: center; justify-content: space-between; gap: 1.2rem; padding: .35rem 0 1.15rem; }
        .portal-brand { display: flex; align-items: center; gap: .72rem; }
        .portal-brand-mark { display: grid; place-items: center; width: 48px; height: 48px; border-radius: 14px; color: #fff; background: linear-gradient(145deg, #0f172a, #2563eb 68%, #06b6d4); font-size: 1.25rem; font-weight: 800; box-shadow: 0 10px 22px rgba(37,99,235,.22); }
        .portal-brand-title { color: #0f172a; font-size: 1.2rem; font-weight: 780; letter-spacing: .02em; margin: 0; }
        .portal-brand-subtitle { color: #64748b; font-size: .68rem; letter-spacing: .1em; margin: .16rem 0 0; }
        .portal-utilities { display: flex; align-items: center; gap: .8rem; color: #64748b; font-size: .74rem; }
        .portal-utilities span { cursor: default; }
        .portal-tagline { color: #2563eb; font-size: .72rem; font-weight: 700; letter-spacing: .05em; margin-left: auto; }
        .portal-nav { display: flex; align-items: center; gap: 1.4rem; border-top: 1px solid rgba(148,163,184,.2); border-bottom: 1px solid rgba(148,163,184,.2); padding: .68rem .1rem; margin-bottom: 1.05rem; }
        .portal-nav a { color: #64748b; font-size: .82rem; text-decoration: none; padding: .34rem .1rem; border-bottom: 2px solid transparent; transition: color .18s ease, border-color .18s ease; }
        .portal-nav a:hover, .portal-nav a.active { color: #2563eb; border-bottom-color: #2563eb; }
        .portal-banner { position: relative; overflow: hidden; min-height: 205px; border-radius: 18px; padding: 1.65rem 2.1rem; color: #fff; background: radial-gradient(circle at 82% 24%, rgba(37,99,235,.34), transparent 31%), linear-gradient(120deg, rgba(11,18,32,.98) 0%, rgba(19,43,85,.82) 38%, rgba(26,66,130,.3) 54%, rgba(29,78,216,.04) 68%); box-shadow: 0 20px 34px rgba(15,23,42,.1); -webkit-mask-image: linear-gradient(90deg, #000 0%, #000 48%, rgba(0,0,0,.94) 58%, rgba(0,0,0,.72) 72%, rgba(0,0,0,.3) 88%, transparent 100%); mask-image: linear-gradient(90deg, #000 0%, #000 48%, rgba(0,0,0,.94) 58%, rgba(0,0,0,.72) 72%, rgba(0,0,0,.3) 88%, transparent 100%); }
        .portal-banner::before { content: ""; position: absolute; inset: 0; opacity: .2; background-image: linear-gradient(rgba(148,163,184,.35) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,.35) 1px, transparent 1px); background-size: 28px 28px; mask-image: linear-gradient(90deg, rgba(0,0,0,.8), transparent 76%); }
        .portal-banner-content { position: relative; z-index: 1; max-width: 720px; }
        .portal-banner-kicker { color: #93c5fd; font-size: .76rem; font-weight: 760; letter-spacing: .12em; margin: 0 0 .65rem; }
        .portal-banner h1 { color: #fff; font-size: clamp(2.2rem, 4vw, 4rem); line-height: 1.05; letter-spacing: -.05em; margin: 0; }
        .portal-banner-lead { color: #dbeafe; font-size: 1.06rem; line-height: 1.6; max-width: 42rem; margin: .85rem 0 .45rem; }
        .portal-banner-copy { color: #bfdbfe; font-size: .82rem; line-height: 1.6; max-width: 38rem; margin: 0; }
        .portal-section { margin-top: 1.4rem; }
        .portal-section-head { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: .75rem; }
        .portal-section-head h2 { color: #0f172a; font-size: 1.45rem; letter-spacing: -.025em; margin: 0; }
        .portal-section-head p { color: #64748b; font-size: .82rem; line-height: 1.5; margin: .3rem 0 0; }
        .portal-section-label { color: #2563eb; font-size: .68rem; font-weight: 760; letter-spacing: .12em; white-space: nowrap; }
        .portal-ai-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
        .portal-ai-card { position: relative; overflow: hidden; min-height: 330px; border: 1px solid rgba(129,140,248,.3); border-radius: 14px; color: #fff; background: radial-gradient(circle at 92% 8%, rgba(99,102,241,.72), transparent 42%), linear-gradient(145deg, #0f172a, #1e1b4b); box-shadow: 0 14px 28px rgba(30,27,75,.16); transition: transform .18s ease, box-shadow .18s ease; }
        .portal-ai-card:hover { transform: translateY(-3px); box-shadow: 0 18px 32px rgba(30,27,75,.24); }
        .portal-ai-card::after { content: ""; position: absolute; width: 130px; height: 130px; right: -42px; bottom: -56px; border: 1px solid rgba(165,180,252,.2); border-radius: 50%; box-shadow: 0 0 0 16px rgba(129,140,248,.05), 0 0 0 32px rgba(129,140,248,.04); }
        .portal-news-cover { position: relative; height: 116px; overflow: hidden; background: linear-gradient(135deg, #e0e7ff, #f8fafc); }
        .portal-news-cover img { width: 100%; height: 100%; object-fit: cover; border: 0; border-radius: 0; box-shadow: none; }
        .news-illustrated { position: relative; }
        .news-illustrated small { position: absolute; right: .65rem; bottom: .5rem; color: #dbeafe; background: rgba(15,23,42,.7); border-radius: 4px; padding: .12rem .38rem; font-size: .62rem; }
        .portal-news-cover.is-logo { display: grid; place-items: center; background: radial-gradient(circle at 78% 20%, rgba(99,102,241,.28), transparent 38%), linear-gradient(135deg, #eef2ff, #dbeafe); }
        .portal-news-cover.is-logo span { color: #1e3a8a; font-size: 1.05rem; font-weight: 820; letter-spacing: .04em; }
        .portal-news-body { position: relative; z-index: 1; padding: 1rem 1.05rem 1.05rem; }
        .portal-ai-card .portal-card-tag { color: #c4b5fd; font-size: .65rem; font-weight: 760; letter-spacing: .1em; }
        .portal-ai-card h3 { color: #fff; font-size: 1.02rem; line-height: 1.45; margin: .6rem 0 .45rem; }
        .portal-ai-card p { color: #c7d2fe; font-size: .78rem; line-height: 1.6; margin: 0; }
        .portal-card-meta { display: flex; justify-content: space-between; gap: .5rem; color: #a5b4fc; font-size: .66rem; margin-top: .85rem; }
        .portal-card-meta a { color: #c4b5fd; text-decoration: none; }
        .portal-card-meta a:hover { color: #fff; text-decoration: underline; }
        .news-official-links { display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0 1rem; }
        .news-official-links a { color: #1e40af; background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 999px; padding: .45rem .85rem; font-size: .78rem; text-decoration: none; }
        .portal-ai-card a.news-image-link, .news-card a.news-image-link { display: block; margin: 0; }
        .portal-ai-card a.news-title-link, .news-card a.news-title-link { color: inherit; font-size: inherit; line-height: inherit; margin: 0; text-decoration: none; }
        .portal-ai-card a.news-title-link:hover, .news-card a.news-title-link:hover { text-decoration: underline; }
        .portal-ai-card::after { pointer-events: none; }
        .portal-dual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
        .portal-public-card { border: 1px solid #dbe3ee; border-radius: 14px; background: rgba(255,255,255,.84); padding: 1.1rem; box-shadow: var(--soft-shadow); }
        .portal-public-card.appointment { position: relative; border-top: 3px solid #9f1239; background: linear-gradient(150deg, #fff, #fff8f8 68%, #f5f8fc); }
        .portal-public-card.notice { position: relative; border-top: 3px solid #b45309; background: linear-gradient(150deg, #fff, #fffdf7 72%, #f5f8fc); }
        .portal-public-card.appointment::before, .portal-public-card.notice::before { position: absolute; top: .82rem; right: 1rem; border: 1px solid currentColor; border-radius: 3px; padding: .16rem .3rem; font-size: .58rem; letter-spacing: .08em; opacity: .7; }
        .portal-public-card.appointment::before { content: "内部公示"; color: #9f1239; }
        .portal-public-card.notice::before { content: "内部通知"; color: #b45309; }
        .portal-public-card h3 { color: #1e293b; font-size: 1rem; margin: 0; }
        .portal-public-card p { color: #64748b; font-size: .76rem; line-height: 1.55; margin: .35rem 0 .7rem; }
        .portal-public-head { display: flex; align-items: center; justify-content: space-between; gap: .7rem; border-bottom: 1px solid #e5eaf1; padding-bottom: .65rem; margin-bottom: .5rem; }
        .portal-public-code { color: #64748b; font-size: .65rem; }
        .portal-table { width: 100%; border-collapse: collapse; font-size: .73rem; }
        .portal-table th, .portal-table td { border-bottom: 1px solid #edf0f4; padding: .55rem .35rem; text-align: left; }
        .portal-table th { color: #64748b; font-weight: 650; background: #f8fafc; }
        .portal-table td { color: #475569; }
        .portal-status { display: inline-block; border-radius: 999px; padding: .18rem .4rem; font-size: .61rem; }
        .portal-status.active { color: #9f1239; background: #fce7f3; }
        .portal-status.done { color: #166534; background: #dcfce7; }
        .portal-status.archived { color: #64748b; background: #f1f5f9; }
        .portal-notice-list { display: grid; gap: .45rem; }
        .portal-notice-item { display: grid; grid-template-columns: 7px 1fr auto; gap: .5rem; align-items: start; border-bottom: 1px solid #edf0f4; padding: .38rem 0; }
        .portal-notice-item:last-child { border-bottom: 0; }
        .portal-notice-dot { width: 7px; height: 7px; border-radius: 50%; background: #b45309; margin-top: .3rem; box-shadow: 0 0 0 3px rgba(180,83,9,.1); }
        .portal-notice-copy strong { display: block; color: #475569; font-size: .76rem; font-weight: 650; }
        .portal-notice-copy small { display: block; color: #94a3b8; font-size: .65rem; line-height: 1.45; margin-top: .15rem; }
        .portal-notice-date { color: #94a3b8; font-size: .64rem; white-space: nowrap; }
        .portal-product-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; }
        .portal-product-card { position: relative; overflow: hidden; min-height: 168px; border: 1px solid rgba(37,99,235,.14); border-radius: 14px; background: linear-gradient(150deg, #fff, #f3f7ff); padding: 1.05rem; box-shadow: var(--soft-shadow); transition: transform .18s ease, border-color .18s ease; }
        .portal-product-card:hover { transform: translateY(-3px); border-color: rgba(37,99,235,.35); }
        .portal-product-mark { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 10px; color: #2563eb; background: #eaf3ff; font-size: .85rem; font-weight: 760; }
        .portal-product-category { display: block; color: #64748b; font-size: .62rem; letter-spacing: .08em; margin-top: .55rem; }
        .portal-product-card h3 { color: #1e293b; font-size: .92rem; margin: .7rem 0 .38rem; }
        .portal-product-card p { color: #64748b; font-size: .73rem; line-height: 1.55; margin: 0; }
        .portal-product-card a { display: inline-block; color: #2563eb; font-size: .68rem; text-decoration: none; margin-top: .75rem; }
        .portal-disclaimer { color: #94a3b8; font-size: .66rem; text-align: right; margin: 1rem 0 .2rem; }
        .portal-footer { display: flex; align-items: start; justify-content: space-between; gap: 1rem; border-top: 1px solid rgba(148,163,184,.2); margin-top: 1.8rem; padding: 1.1rem 0 1.8rem; }
        .portal-footer strong { color: #334155; font-size: .82rem; }
        .portal-footer p { color: #94a3b8; font-size: .68rem; line-height: 1.6; margin: .25rem 0 0; }
        .portal-footer-note { color: #94a3b8; font-size: .66rem; text-align: right; }
        .employee-hero { padding: .75rem 0 1.45rem; }
        .employee-hero-wrap { display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
        .employee-hero-copy { flex: 1; min-width: 0; }
        .employee-mascot { width: 168px; height: 168px; object-fit: cover; object-position: center 28%; border-radius: 30px; box-shadow: 0 14px 32px rgba(29,29,31,.14); border: 1px solid rgba(29,29,31,.1); }
        .company-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 12px; color: #fff; background: linear-gradient(145deg, #1d1d1f 0%, #4b5563 100%); font-size: 1.08rem; font-weight: 760; box-shadow: 0 8px 20px rgba(29,29,31,.2); }
        .module-header { border-top: 1px solid var(--line); padding: 1.25rem 0 1.05rem; margin-bottom: 1.05rem; }
        .module-header h1 { color: var(--ink); font-size: 2.25rem; line-height: 1.1; margin: .15rem 0 .55rem; letter-spacing: -.025em; }
        .module-header p { color: var(--muted); line-height: 1.6; margin: 0; max-width: 46rem; }
        .public-notice { background: #fff; border: 1px solid #d9dee7; border-radius: 10px; box-shadow: 0 8px 24px rgba(30,41,59,.06); overflow: hidden; }
        .public-notice-head { background: #f4f7fb; border-bottom: 1px solid #d9dee7; padding: .9rem 1.1rem; color: #334155; font-size: .86rem; font-weight: 700; letter-spacing: .04em; }
        .public-notice-row { display: grid; grid-template-columns: 1.1fr 1fr 1fr .7fr; gap: .8rem; padding: .9rem 1.1rem; border-bottom: 1px solid #edf0f4; color: #475569; font-size: .88rem; }
        .public-notice-row:last-child { border-bottom: 0; }
        .public-notice-row strong { color: #1e293b; font-weight: 650; }
        .public-badge { color: #166534; background: #dcfce7; border-radius: 999px; padding: .2rem .52rem; font-size: .72rem; justify-self: start; }
        .public-home-grid { display: grid; grid-template-columns: 1.35fr .9fr; gap: 1rem; margin-bottom: 1rem; }
        .public-feature { min-height: 230px; border-radius: 12px; padding: 1.35rem; color: #fff; background: linear-gradient(135deg,#174a7e,#2874b2 58%,#83b9d8); box-shadow: 0 12px 28px rgba(30,64,95,.18); }
        .public-feature .tag { color: #dbeafe; font-size: .74rem; letter-spacing: .1em; }
        .public-feature h2 { color: #fff; font-size: 1.55rem; line-height: 1.3; margin: 1.1rem 0 .55rem; }
        .public-feature p { color: #e0f2fe; font-size: .86rem; line-height: 1.6; }
        .public-headlines { border: 1px solid #d9dee7; border-radius: 12px; background: #fff; padding: 1rem 1.15rem; }
        .public-headlines h3 { color: #1e293b; margin: 0 0 .65rem; font-size: 1.05rem; }
        .public-headlines li { color: #475569; font-size: .84rem; line-height: 1.5; margin: .55rem 0; }
        .news-card a { display: inline-block; color: #c4b5fd; font-size: .76rem; margin-top: .25rem; text-decoration: none; }
        .news-card a:hover { color: #fff; text-decoration: underline; }
        .news-hero { border-radius: 16px; padding: 1.35rem 1.45rem; color: #fff; background: radial-gradient(circle at 84% 10%, rgba(124,58,237,.75), transparent 38%), linear-gradient(135deg, #111827, #1e1b4b 52%, #312e81); box-shadow: 0 18px 40px rgba(30,27,75,.25); overflow: hidden; }
        .news-hero .module-kicker { color: #c4b5fd; }
        .news-hero h1 { color: #fff; margin: .2rem 0 .5rem; }
        .news-hero p { color: #e0e7ff; }
        .news-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: 1rem; }
        .news-card { overflow: hidden; min-height: 300px; border: 1px solid rgba(255,255,255,.16); border-radius: 12px; background: rgba(255,255,255,.09); backdrop-filter: blur(14px); }
        .news-cover { height: 126px; overflow: hidden; background: rgba(255,255,255,.9); }
        .news-cover img { width: 100%; height: 100%; object-fit: cover; border: 0; border-radius: 0; box-shadow: none; }
        .news-cover.is-logo { display: grid; place-items: center; background: radial-gradient(circle at 78% 20%, rgba(99,102,241,.3), transparent 38%), linear-gradient(135deg, #eef2ff, #dbeafe); }
        .news-cover.is-logo span { color: #1e3a8a; font-size: 1.08rem; font-weight: 820; letter-spacing: .04em; }
        .news-card-body { padding: .9rem; }
        .news-card .tag { color: #c4b5fd; font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
        .news-card h3 { color: #fff; font-size: .98rem; line-height: 1.4; margin: .48rem 0 .35rem; }
        .news-card p { color: #c7d2fe; font-size: .8rem; line-height: 1.5; }
        .finance-shell { background: #f8fafc; border: 1px solid #dbe3ee; border-radius: 12px; padding: 1.1rem; }
        .finance-layout { display: grid; grid-template-columns: 165px 1fr; gap: 1rem; }
        .finance-nav { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: .55rem; }
        .finance-nav div { color: #64748b; border-radius: 6px; padding: .62rem .7rem; font-size: .82rem; }
        .finance-nav div.active { color: #4338ca; background: #eef2ff; font-weight: 700; }
        .finance-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: .95rem; color: #334155; font-size: .83rem; }
        .finance-period { color: #64748b; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; padding: .35rem .58rem; }
        .finance-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; margin-bottom: 1rem; }
        .finance-kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: .75rem; }
        .finance-kpi span { display: block; color: #64748b; font-size: .72rem; }
        .finance-kpi strong { display: block; color: #0f172a; font-size: 1.15rem; margin-top: .25rem; }
        .finance-kpi em { color: #15803d; font-size: .7rem; font-style: normal; }
        .finance-table { width: 100%; border-collapse: collapse; background: #fff; font-size: .82rem; }
        .finance-table th, .finance-table td { border-bottom: 1px solid #e2e8f0; padding: .65rem .7rem; text-align: right; }
        .finance-table th:first-child, .finance-table td:first-child { text-align: left; }
        .finance-table th { color: #64748b; background: #f8fafc; font-weight: 650; }
        .finance-table td { color: #334155; }
        .employee-hero h1 { color: var(--ink); font-size: 2.55rem; font-weight: 730; line-height: 1.1; margin: 0; }
        .command-eyebrow, .section-overline, .answer-overline { color: var(--accent); font-size: .76rem; font-weight: 700; margin: 0 0 .55rem; }
        [data-testid=stImage] img { width: 100%; max-height: 300px; object-fit: cover; object-position: top; border: 1px solid rgba(29,29,31,.12); border-radius: 8px; box-shadow: var(--shadow); }
        .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1.5rem; border-top: 1px solid var(--line); padding: 1.35rem 0 .9rem; }
        .section-heading h2 { color: var(--ink); font-size: 1.45rem; line-height: 1.2; margin: 0; }
        .section-heading p { color: var(--muted); font-size: .9rem; line-height: 1.55; margin: .35rem 0 0; max-width: 31rem; }
        .section-overline { margin-bottom: .35rem !important; }
        .suggestion-caption { color: var(--muted); font-size: .82rem; margin: .15rem 0 .5rem; }
        .answer-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--soft-shadow); padding: 1.12rem 1.2rem; }
        .answer-panel p { color: var(--ink); line-height: 1.65; margin: .35rem 0; }
        .answer-panel ul { margin: .55rem 0 .65rem 1.1rem; padding: 0; }
        .answer-panel li { color: var(--ink); line-height: 1.65; margin: .3rem 0; }
        .answer-overline { margin-bottom: .55rem !important; }
        [data-testid=stChatMessage] { background: transparent; border-top: 1px solid var(--line); padding: 1.2rem 0; }
        [data-testid=stExpander] { border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.62); }
        [data-testid=stExpander] summary { color: var(--ink); font-weight: 620; }
        [data-testid=stChatInput] { background: rgba(255,255,255,.9); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--soft-shadow); }
        [data-testid=stChatInput] textarea { color: var(--ink); }
        .table-fallback { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); box-shadow: var(--soft-shadow); }
        .table-fallback table { width: 100%; border-collapse: collapse; font-size: .84rem; }
        .table-fallback th { background: rgba(0,113,227,.065); color: var(--ink); font-weight: 700; text-align: left; white-space: nowrap; }
        .table-fallback th, .table-fallback td { padding: .7rem .75rem; border-bottom: 1px solid var(--line); vertical-align: top; }
        .table-fallback tr:last-child td { border-bottom: 0; }
        .table-fallback td { color: #515154; }
        div[data-testid=stMetric] { background: var(--surface); border: 1px solid var(--line); padding: 1rem 1.05rem; border-radius: 8px; box-shadow: var(--soft-shadow); }
        div[data-testid=stMetricLabel] { color: var(--muted); font-size: .82rem; }
        div[data-testid=stMetricValue] { color: var(--ink); font-weight: 690; }
        .stButton > button { border: 1px solid var(--line); border-radius: 999px; color: var(--ink); background: rgba(255,255,255,.86); box-shadow: none; transition: background .18s ease, border-color .18s ease, transform .18s ease; }
        .stButton > button:hover { border-color: rgba(0,113,227,.38); color: var(--accent); background: var(--accent-soft); transform: translateY(-1px); }
        .stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
        .stButton > button[kind="primary"]:hover { background: #0077ed; color: #fff; }
        .stTabs [data-baseweb=tab-list] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb=tab] { color: var(--muted); font-size: .93rem; padding: .65rem .05rem; }
        .stTabs [aria-selected="true"] { color: var(--ink); font-weight: 650; }
        .stTabs [data-baseweb=tab-highlight] { background-color: var(--accent); height: 2px; }
        .stAlert { border-radius: 8px; border-color: var(--line); }
        @media (max-width: 700px) {
           .block-container { padding: 3.25rem 1rem 3rem; }
           .overview-hero { min-height: 0; padding: 1.45rem 1.25rem; border-radius: 14px; }
           .overview-hero h1 { font-size: 2.5rem; }
           .overview-hero .overview-lead { font-size: .95rem; }
           .overview-kpis { grid-template-columns: 1fr 1fr; }
           .overview-grid, .overview-products { grid-template-columns: 1fr; }
           .overview-achievements { grid-template-columns: 1fr; }
           .dashboard-topbar { align-items: flex-start; flex-direction: column; }
           .dashboard-toolbar-actions { justify-content: flex-start; }
           .dashboard-hero { grid-template-columns: 1fr; padding: 1.25rem; }
           .dashboard-hero h1 { white-space: normal; }
           .dashboard-hero-visual { display: none; }
           .dashboard-metrics, .dashboard-columns, .cockpit-grid, .activity-grid { grid-template-columns: 1fr; }
           .service-grid { grid-template-columns: 1fr 1fr; }
           .overview-triad, .personal-layout, .detail-grid { grid-template-columns: 1fr; }
           .finance-summary { grid-template-columns: 1fr 1fr; }
           .portal-brandbar { align-items: flex-start; flex-direction: column; }
           .portal-tagline { margin-left: 0; }
           .portal-utilities { flex-wrap: wrap; gap: .55rem; }
           .portal-nav { gap: .85rem; overflow-x: auto; white-space: nowrap; padding-bottom: .55rem; }
           .portal-banner { min-height: 210px; padding: 1.25rem 1.15rem; }
           .portal-banner h1 { font-size: 2.35rem; }
           .portal-banner-lead { font-size: .92rem; max-width: 19rem; }
           .portal-ai-grid, .portal-dual-grid, .portal-product-grid { grid-template-columns: 1fr; }
           .erp-header { align-items: flex-start; flex-direction: column; }
           .erp-header-tools { flex-wrap: wrap; }
           .erp-kpi-grid, .erp-shortcuts, .erp-main-grid, .erp-bottom-grid { grid-template-columns: 1fr 1fr; }
           .erp-main-grid, .erp-bottom-grid { grid-template-columns: 1fr; }
           .erp-shortcuts { grid-template-columns: 1fr 1fr; }
           div.st-key-assistant-launcher { right: .8rem !important; bottom: .8rem !important; width: 58px !important; height: 58px !important; }
           div.st-key-assistant-launcher > div, div.st-key-assistant-launcher button { width: 58px !important; height: 58px !important; min-height: 58px !important; }
           [data-baseweb="modal"] { padding: .65rem !important; }
           [data-baseweb="modal"] [role="dialog"] { width: calc(100vw - 1.3rem) !important; max-width: calc(100vw - 1.3rem) !important; max-height: calc(100vh - 1.3rem) !important; border-radius: 15px !important; }
           .portal-ai-card, .portal-public-card, .portal-product-card { min-height: 0; }
           .portal-footer { flex-direction: column; }
           .portal-footer-note { text-align: left; }
           .portal-disclaimer { text-align: left; }
           .overview-section-heading { align-items: start; flex-direction: column; gap: .35rem; }
          .employee-hero h1 { font-size: 2.15rem; }
          .employee-hero-wrap { align-items: flex-start; gap: 1rem; }
          .employee-mascot { width: 92px; height: 92px; border-radius: 20px; }
          .module-header h1 { font-size: 1.9rem; }
          .public-notice-row { grid-template-columns: 1fr 1fr; gap: .35rem .6rem; }
          .news-grid, .finance-kpis { grid-template-columns: 1fr 1fr; }
          .public-home-grid, .finance-layout { grid-template-columns: 1fr; }
          .section-heading { align-items: start; flex-direction: column; gap: .45rem; padding-top: 1.1rem; }
          .section-heading h2 { font-size: 1.28rem; }
          [data-testid=stImage] img { max-height: 208px; }
          div[data-testid=stMetric] { padding: .85rem .9rem; }
          [data-testid=stChatMessage] { padding: .9rem 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_employee_header() -> None:
    mascot_markup = ""
    if MASCOT_PATH.is_file():
        encoded_mascot = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
        mascot_markup = (
            f'<img class="employee-mascot" src="data:image/png;base64,{encoded_mascot}" '
            f'alt="{COMPANY_NAME}员工助手形象" />'
        )
    st.markdown(
        f"""
        <section class="employee-hero employee-hero-wrap">
          <div class="employee-hero-copy">
            <p class="command-eyebrow">制度问答助手</p>
            <h1>今天想了解什么？</h1>
            <p class="command-summary">用一句话描述你的工作问题，我们会给出清晰、可执行的答复。</p>
          </div>
          {mascot_markup}
        </section>
        """,
        unsafe_allow_html=True,
    )


def _news_illustration() -> str:
    """Local vector illustration for a publisher whose image CDN blocks embedding."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">
    <defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#061a38"/><stop offset="1" stop-color="#34236c"/></linearGradient>
    <radialGradient id="halo"><stop stop-color="#26d6ed" stop-opacity=".6"/><stop offset="1" stop-color="#26d6ed" stop-opacity="0"/></radialGradient>
    <linearGradient id="chip" x2="1" y2="1"><stop stop-color="#173c70"/><stop offset="1" stop-color="#5967b8"/></linearGradient></defs>
    <rect width="900" height="360" fill="url(#bg)"/><ellipse cx="470" cy="190" rx="300" ry="210" fill="url(#halo)"/>
    <g fill="none" stroke="#57e5ff" stroke-width="2" opacity=".65">
    <path d="M60 95H245L325 160H382M92 250H230L315 204H382M160 35V72L307 112L382 160M65 175H382M518 160H620L705 84H845M518 204H660L738 265H856M518 180H824M485 112V62L580 25M423 248V302H300L266 339"/>
    <circle cx="65" cy="175" r="7"/><circle cx="60" cy="95" r="5"/><circle cx="92" cy="250" r="7"/><circle cx="845" cy="84" r="6"/><circle cx="824" cy="180" r="6"/><circle cx="856" cy="265" r="7"/></g>
    <g transform="translate(362 92)"><rect x="-14" y="-14" width="204" height="204" rx="33" fill="#102d56" stroke="#50d1f2" stroke-opacity=".4"/>
    <rect width="176" height="176" rx="24" fill="url(#chip)" stroke="#9fefff" stroke-width="2"/>
    <g stroke="#b4f5ff" stroke-width="3" fill="none"><path d="M44 122V76L87 49L130 76V122L87 146Z M44 76L130 122M130 76L44 122M87 49V146"/></g>
    <g fill="#d7ffff"><circle cx="44" cy="76" r="7"/><circle cx="130" cy="76" r="7"/><circle cx="87" cy="49" r="7"/><circle cx="44" cy="122" r="7"/><circle cx="130" cy="122" r="7"/><circle cx="87" cy="146" r="7"/></g></g>
    <g fill="#bae9ff" opacity=".45"><circle cx="720" cy="135" r="3"/><circle cx="237" cy="193" r="3"/><circle cx="568" cy="282" r="3"/><circle cx="197" cy="309" r="3"/><circle cx="782" cy="315" r="3"/></g></svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode("ascii")


def _news_cover_markup(item: dict[str, str], class_name: str) -> str:
    if item.get("source") == "量子位":
        return (f'<div class="{class_name} news-illustrated"><img src="{_news_illustration()}" '
                'alt="AI 芯片与数据网络主题示意图"><small>主题示意图</small></div>')
    if item.get("image_kind") == "photo" and item.get("image"):
        return (
            f'<div class="{class_name}"><img src="{escape(item["image"], quote=True)}" '
            f'alt="{escape(item["source"])}" loading="lazy" referrerpolicy="no-referrer"></div>'
        )
    return f'<div class="{class_name} is-logo"><span>{escape(item["source"])}</span></div>'


def _news_card_markup(item: dict[str, str], *, portal: bool) -> str:
    card, cover, body, tag = (
        ("portal-ai-card", "portal-news-cover", "portal-news-body", "portal-card-tag")
        if portal else ("news-card", "news-cover", "news-card-body", "tag")
    )
    link = escape(item["link"], quote=True)
    summary = _news_summary(item.get("summary", ""))
    return (
        f'<article class="{card}"><a class="news-image-link" href="{link}" target="_blank" rel="noopener noreferrer">'
        f'{_news_cover_markup(item, cover)}</a><div class="{body}">'
        f'<span class="{tag}">{escape(item.get("region", "资讯"))} · {escape(item["category"])} · {escape(item["published"] or "日期未提供")}</span>'
        f'<h3><a class="news-title-link" href="{link}" target="_blank" rel="noopener noreferrer">{escape(item["title"])}</a></h3>'
        + (f'<p>{escape(summary)}</p>' if summary else "")
        + f'<div class="portal-card-meta"><span>{escape(item["source"])} · {escape(_news_source_type(item))}</span></div></div></article>'
    )


def _official_news_links() -> str:
    return '<nav class="news-official-links" aria-label="AI 公司官方新闻">' + "".join(
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label} · 官方新闻 ↗</a>'
        for label, url in (
            ("OpenAI", "https://openai.com/news/"),
            ("Anthropic / Claude", "https://www.anthropic.com/news"),
            ("Google / Gemini", "https://blog.google/technology/ai/"),
        )
    ) + '</nav>'


def render_company_overview() -> None:
    """Render the internal employee overview with shared navigation."""
    poster_style = ""
    if MASCOT_PATH.is_file():
        encoded_mascot = base64.b64encode(MASCOT_PATH.read_bytes()).decode("ascii")
        poster_style = f"background-image:url('data:image/png;base64,{encoded_mascot}');"
    frontier_news, frontier_error = fetch_ai_frontier_news()
    if frontier_error:
        st.info(frontier_error)
    portal_news_cards = "".join(
        _news_card_markup(item, portal=True) for item in frontier_news[:9]
    )
    if not portal_news_cards:
        portal_news_cards = '<article class="portal-ai-card"><div class="portal-news-body"><span class="portal-card-tag">连接失败</span><h3>暂时没有获取到 AI 行业资讯</h3><p>外部资讯源当前均不可达，请检查网络后刷新重试。</p></div></article>'

    st.markdown(
        f"""
        <div id="top" class="portal-shell"><div class="portal-backdrop" style="{poster_style}" aria-hidden="true"></div><nav class="portal-nav" aria-label="员工信息栏目"><a class="active" href="#top">首页</a><a href="#about">关于鲲坤</a><a href="#ai-frontier">AI 前沿</a><a href="#appointments">任命公示</a><a href="#announcements">公司公告</a><a href="#products">产品与解决方案</a></nav>
        <section class="portal-banner" id="about"><div class="portal-banner-content"><h1>{COMPANY_NAME}</h1><p class="portal-banner-lead">以人工智能连接企业运营，构建统一、智能、可信的数字化工作平台</p><p class="portal-banner-copy">内部员工信息台 · 仅供公司内部使用</p></div></section>
        <section id="ai-frontier" class="portal-section"><div class="portal-section-head"><div><h2>AI 前沿资讯</h2><p>公司官方发布与专业媒体报道 · 模型升级、产品动态与行业应用</p></div><span class="portal-section-label">多源资讯</span></div>{_official_news_links()}<div class="portal-ai-grid">{portal_news_cards}</div></section>
        <section class="portal-section portal-dual-grid"><article id="appointments" class="portal-public-card appointment"><div class="portal-public-head"><div><h3>公司任命公示</h3><p>规范发布组织任命与岗位变动信息。</p></div><span class="portal-public-code">人事任命公示〔2026〕08 号</span></div><table class="portal-table"><thead><tr><th>拟任岗位</th><th>姓名</th><th>任职部门</th><th>公示状态</th></tr></thead><tbody><tr><td>AI 产品运营负责人</td><td>林知夏</td><td>产品与增长中心</td><td><span class="portal-status active">公示中</span></td></tr><tr><td>企业服务平台主管</td><td>周予安</td><td>客户成功部</td><td><span class="portal-status done">已生效</span></td></tr><tr><td>财务数字化项目经理</td><td>沈嘉禾</td><td>财务管理部</td><td><span class="portal-status archived">已归档</span></td></tr></tbody></table></article><article id="announcements" class="portal-public-card notice"><div class="portal-public-head"><div><h3>公司公告</h3><p>集中展示公司重要通知、制度更新和企业动态。</p></div><span class="portal-public-code">公告中心 · 2026 年 8 月</span></div><div class="portal-notice-list"><div class="portal-notice-item"><i class="portal-notice-dot"></i><span class="portal-notice-copy"><strong>关于 9 月财务结账与发票提交时间的通知</strong><small>财务管理部 · 请于 9 月 3 日前完成发票提交</small></span><span class="portal-notice-date">08-26</span></div><div class="portal-notice-item"><i class="portal-notice-dot"></i><span class="portal-notice-copy"><strong>信息安全管理制度 2026 修订版</strong><small>信息安全委员会 · 制度更新</small></span><span class="portal-notice-date">08-27</span></div><div class="portal-notice-item"><i class="portal-notice-dot"></i><span class="portal-notice-copy"><strong>园区办公网络维护安排</strong><small>行政与 IT 服务 · 维护窗口 22:00–24:00</small></span><span class="portal-notice-date">08-28</span></div></div></article></section>
        <section id="products" class="portal-section"><div class="portal-section-head"><div><h2>产品与解决方案</h2><p>覆盖智能制造、半导体、电子元器件与 AI 软件技术。</p></div><span class="portal-section-label">核心产品</span></div><div class="portal-product-grid"><article class="portal-product-card"><span class="portal-product-mark">01</span><span class="portal-product-category">自动化设备</span><h3>智能装配与检测产线</h3><p>面向 3C、汽车电子与精密制造，提供柔性装配、视觉检测和数据追溯一体化设备。</p><a href="#products">了解更多 →</a></article><article class="portal-product-card"><span class="portal-product-mark">02</span><span class="portal-product-category">半导体</span><h3>晶圆级封装与测试方案</h3><p>覆盖先进封装、晶圆级测试与良率分析，支持 12/8 英寸产线协同。</p><a href="#products">了解更多 →</a></article><article class="portal-product-card"><span class="portal-product-mark">03</span><span class="portal-product-category">电子元器件</span><h3>高可靠功率器件与连接器</h3><p>提供功率器件、连接器与信号完整性方案，服务工业控制和新能源客户。</p><a href="#products">了解更多 →</a></article><article class="portal-product-card"><span class="portal-product-mark">04</span><span class="portal-product-category">AI 软件技术</span><h3>鲲坤企业智能平台</h3><p>面向集团员工的制度问答、知识检索与经营协同软件，强调证据链和安全边界。</p><a href="#products">了解更多 →</a></article></div></section>
        <p class="portal-disclaimer">页面内容仅用于内部员工信息展示 · 具体制度以最新发布版本为准</p></div>
        """,
        unsafe_allow_html=True,
    )

    render_scene_questions(
        "概览",
        ["知识库有哪些制度？", "忘记打卡后几天内可以申请考勤修正？"],
    )


def personal_task_rows(items: list[dict[str, str]]) -> str:
    """Render role-specific personal work items into the existing task-list pattern."""
    return "".join(
        (
            '<div class="task-row"><div>'
            f'<strong>{escape(item["title"])}</strong>'
            f'<small>{escape(item["meta"])}</small>'
            "</div>"
            f'<span class="task-chip {escape(item["tone"])}">{escape(item["chip"])}</span>'
            "</div>"
        )
        for item in items
    )


def render_personal_center(role: str = "employee") -> None:
    """Render local identity information and personal work items."""
    profile = role_profile(role)
    joining_date = date.fromisoformat(profile["joining_date"])
    joining_days = max((date.today() - joining_date).days, 0)
    avatar_markup = (
        f'<span class="personal-avatar personal-avatar-initial" '
        f'style="--avatar-color:{escape(profile["avatar_color"])}">{escape(profile["avatar"])}</span>'
    )
    if role == "employee" and PORTRAIT_PATH.is_file():
        encoded_portrait = base64.b64encode(PORTRAIT_PATH.read_bytes()).decode("ascii")
        avatar_markup = f'<img class="personal-avatar" src="data:image/png;base64,{encoded_portrait}" alt="{escape(profile["name"])} 工作照" />'
    st.markdown(
        f"""
        <section class="module-header"><p class="section-overline">{COMPANY_NAME} · 员工自助</p><h1>个人中心</h1><p>查看个人资料、入职信息、申请记录和只与你有关的待办事项。</p></section>
        <section class="personal-layout"><article class="personal-profile"><div class="personal-profile-head">{avatar_markup}<div><h2>{escape(profile["name"])}</h2><p>{escape(profile["department"])}</p><span class="task-chip">当前身份 · {escape(profile["label"])}</span></div></div><div class="joining-badge"><span>今天是你入职的第</span><strong>{joining_days}</strong><span>天</span></div><div class="profile-facts"><div class="profile-fact"><span>员工编号</span><strong>{escape(profile["employee_id"])}</strong></div><div class="profile-fact"><span>入职日期</span><strong>{escape(profile["joining_date"])}</strong></div><div class="profile-fact"><span>直属上级</span><strong>{escape(profile["manager"])}</strong></div><div class="profile-fact"><span>办公地点</span><strong>{escape(profile["office"])}</strong></div><div class="profile-fact"><span>在职状态</span><strong>{escape(profile["status"])}</strong></div></div></article><div class="detail-grid"><article class="detail-card"><h3>我的待办</h3><div class="task-list">{personal_task_rows(profile["todos"])}</div></article><article class="detail-card"><h3>我的申请</h3><div class="task-list">{personal_task_rows(profile["applications"])}</div></article></div></section>
        <section class="dashboard-section detail-grid"><article class="detail-card"><h3>个人资料</h3><p>联系方式、紧急联系人、收款账户等信息由员工本人维护，修改后按公司流程生效。</p><span class="dashboard-secondary">编辑资料</span></article><article class="detail-card"><h3>我的偏好</h3><p>当前主题：浅色　·　通知：工作日 09:00–18:00　·　语言：简体中文</p><span class="dashboard-secondary">管理偏好</span></article><article class="detail-card"><h3>通知设置</h3><p>消息提醒与订阅管理：制度更新提醒已开启，公告订阅已开启，营销消息已关闭。</p><span class="dashboard-secondary">管理订阅</span></article></section><p class="dashboard-note">页面使用样例人员信息 · 请勿用于真实业务决策</p>
        """,
        unsafe_allow_html=True,
    )

    render_scene_questions(
        "个人中心",
        ["年休假怎么申请？", "入职试用期评估什么时候发起？", "离职交接需要办哪些手续？"],
    )


def render_hr_services_page() -> None:
    """Render detailed HR services with approval progress in one place."""
    st.markdown(
        f"""
        <section class="module-header"><p class="section-overline">{COMPANY_NAME} · 员工服务</p><h1>人事服务</h1><p>请假、考勤、证明、入职和人事审批统一办理，进度清晰可追踪。</p></section>
        <section class="finance-summary"><div class="finance-summary-card"><span>年休假余额</span><strong>8.5 天</strong></div><div class="finance-summary-card"><span>本月考勤状态</span><strong>正常</strong></div><div class="finance-summary-card"><span>待处理人事审批</span><strong>2 项</strong></div></section>
        <section class="detail-grid"><article class="detail-card"><h3>请假与考勤</h3><p>查看年假、调休和病假余额；提交请假、补卡并查看审批结果。</p><div class="task-list"><div class="task-row"><div><strong>年休假申请</strong><small>最小申请单位 0.5 天</small></div><span class="task-chip">发起申请 →</span></div><div class="task-row"><div><strong>考勤补卡</strong><small>近 30 天打卡记录</small></div><span class="task-chip">去处理 →</span></div></div></article><article class="detail-card"><h3>人事证明</h3><p>在线申请在职、收入、工作经历和盖章证明，提交后可下载或查看进度。</p><div class="task-list"><div class="task-row"><div><strong>在职证明</strong><small>预计 1 个工作日</small></div><span class="task-chip">申请 →</span></div><div class="task-row"><div><strong>收入证明</strong><small>需填写用途和接收方</small></div><span class="task-chip">申请 →</span></div></div></article></section>
        <section class="dashboard-section detail-card"><div class="dashboard-section-title"><div><h2>审批进度</h2><p>人事类申请的当前节点</p></div><span class="dashboard-toolbar-chip">仅显示本人申请</span></div><div class="workflow-list"><div class="workflow-item"><div class="workflow-head"><strong>年休假 · 2026-09-04</strong><small>已通过</small></div><div class="workflow-track"><i class="workflow-step done"></i><i class="workflow-step done"></i><i class="workflow-step done"></i><i class="workflow-step done"></i></div></div><div class="workflow-item"><div class="workflow-head"><strong>补卡申请 · 2026-08-26</strong><small>直属上级审批</small></div><div class="workflow-track"><i class="workflow-step done"></i><i class="workflow-step current"></i><i class="workflow-step"></i><i class="workflow-step"></i></div></div></div></section><p class="dashboard-note">人事流程使用样例数据 · 实际审批节点以公司制度和权限为准</p>
        """,
        unsafe_allow_html=True,
    )

    render_scene_questions(
        "人事服务",
        ["年休假怎么申请？", "忘记打卡后几天内可以申请考勤修正？", "入职试用期评估什么时候发起？"],
    )


def render_finance_center_page() -> None:
    """Render a detailed employee finance center with invoice and reimbursement tracking."""
    st.markdown(
        f"""
        <section class="module-header"><p class="section-overline">{COMPANY_NAME} · 财务服务</p><h1>财务中心</h1><p>发票、报销、付款和财务审批集中处理，随时查看流程走到哪一步。</p></section>
        <section class="finance-summary"><div class="finance-summary-card"><span>本月可报销额度</span><strong>¥ 8,000</strong></div><div class="finance-summary-card"><span>待补充发票</span><strong>2 张</strong></div><div class="finance-summary-card"><span>报销中金额</span><strong>¥ 3,680</strong></div></section>
        <section class="detail-grid"><article class="detail-card"><h3>发票报销</h3><p>支持差旅、办公、招待等费用的发票登记、影像上传和报销单提交。</p><div class="task-list"><div class="task-row"><div><strong>扫描 / 上传发票</strong><small>支持 PDF、图片和电子发票</small></div><span class="task-chip">上传 →</span></div><div class="task-row"><div><strong>新建报销单</strong><small>自动带出发票金额与税额</small></div><span class="task-chip">发起 →</span></div></div></article><article class="detail-card"><h3>报销流程</h3><p>报销单提交后依次经过直属上级、财务审核和出纳付款。</p><div class="workflow-list"><div class="workflow-item"><div class="workflow-head"><strong>BX-20260821 · 差旅报销</strong><small>审批中</small></div><div class="workflow-track"><i class="workflow-step done"></i><i class="workflow-step current"></i><i class="workflow-step"></i><i class="workflow-step"></i></div></div><div class="workflow-item"><div class="workflow-head"><strong>BX-20260808 · 办公用品</strong><small>已付款</small></div><div class="workflow-track"><i class="workflow-step done"></i><i class="workflow-step done"></i><i class="workflow-step done"></i><i class="workflow-step done"></i></div></div></div></article></section>
        <section class="dashboard-section detail-card"><div class="dashboard-section-title"><div><h2>我的报销记录</h2><p>最近 90 天 · 样例数据</p></div><span class="dashboard-toolbar-chip">导出明细</span></div><div class="table-fallback"><table><thead><tr><th>报销单号</th><th>费用类型</th><th>金额</th><th>当前状态</th><th>更新时间</th></tr></thead><tbody><tr><td>BX-20260821</td><td>上海-北京差旅</td><td>¥ 2,480</td><td>财务审核</td><td>今天 09:35</td></tr><tr><td>BX-20260808</td><td>办公用品</td><td>¥ 1,200</td><td>已付款</td><td>08-15</td></tr><tr><td>BX-20260726</td><td>客户拜访交通</td><td>¥ 680</td><td>待补发票</td><td>08-12</td></tr></tbody></table></div></section><p class="dashboard-note">财务页面使用样例数据 · 发票与付款以财务系统最终记录为准</p>
        """,
        unsafe_allow_html=True,
    )

    render_scene_questions(
        "财务中心",
        ["差旅报销需要哪些材料？", "上海出差住宿标准是多少？", "发票抬头和税号是什么？"],
    )


def render_erp_page() -> None:
    """Render a realistic static ERP workspace for the employee portal MVP."""
    st.markdown(
        f"""
        <section class="module-header"><p class="section-overline">{COMPANY_NAME} · 企业经营协同</p><h1>ERP 系统</h1><p>统一查看采购、销售、库存与生产协同信息，让业务单据和流程状态清晰可追踪。</p></section>
        <section class="erp-shell">
          <div class="erp-header"><div><div class="erp-header-title">经营协同工作台</div><div class="erp-header-subtitle">集团制造事业群 · 数据更新时间：2026-08-29 09:42</div></div><div class="erp-header-tools"><span class="erp-org">组织：<strong>集团总部</strong>⌄</span><span class="erp-period">期间：<strong>2026 年 8 月</strong>⌄</span></div></div>
          <div class="erp-kpi-grid"><article class="erp-kpi"><span class="erp-kpi-label">本月销售订单</span><strong>¥ 2,486 万</strong><div class="erp-kpi-foot"><span class="erp-positive">↑ 12.6% 环比</span><span>32 笔</span></div></article><article class="erp-kpi"><span class="erp-kpi-label">采购执行金额</span><strong>¥ 1,738 万</strong><div class="erp-kpi-foot"><span class="erp-positive">↑ 8.4% 环比</span><span>完成率 86%</span></div></article><article class="erp-kpi"><span class="erp-kpi-label">库存总额</span><strong>¥ 4,216 万</strong><div class="erp-kpi-foot"><span class="erp-warning">↓ 3.1% 环比</span><span>周转 42 天</span></div></article><article class="erp-kpi"><span class="erp-kpi-label">生产计划达成率</span><strong>94.8%</strong><div class="erp-kpi-foot"><span class="erp-positive">↑ 2.3 个百分点</span><span>运行正常</span></div></article></div>
          <div class="erp-shortcuts"><div class="erp-shortcut"><span class="erp-shortcut-mark">采</span><div><strong>采购与供应链</strong><small>订单、到货、供应商</small></div></div><div class="erp-shortcut"><span class="erp-shortcut-mark">销</span><div><strong>销售订单</strong><small>合同、发货、回款</small></div></div><div class="erp-shortcut"><span class="erp-shortcut-mark">库</span><div><strong>库存管理</strong><small>物料、批次、盘点</small></div></div><div class="erp-shortcut"><span class="erp-shortcut-mark">产</span><div><strong>生产协同</strong><small>计划、工单、报工</small></div></div><div class="erp-shortcut"><span class="erp-shortcut-mark">财</span><div><strong>财务联动</strong><small>应收、应付、成本</small></div></div></div>
          <div class="erp-main-grid"><article class="erp-panel"><div class="erp-panel-head"><h2>近期业务单据</h2><span>查看全部 →</span></div><div style="overflow-x:auto"><table class="erp-table"><thead><tr><th>单据编号</th><th>业务类型</th><th>往来单位</th><th>金额</th><th>状态</th><th>更新时间</th></tr></thead><tbody><tr><td>SO-260828</td><td>销售订单</td><td>华东精密制造有限公司</td><td class="amount">¥ 386,000</td><td><span class="erp-status processing">备货中</span></td><td>今天 09:18</td></tr><tr><td>PO-260827</td><td>采购订单</td><td>苏州芯材电子有限公司</td><td class="amount">¥ 218,500</td><td><span class="erp-status pending">待收货</span></td><td>昨天 17:42</td></tr><tr><td>MO-260826</td><td>生产工单</td><td>装配一车间 · P-08</td><td class="amount">1,200 件</td><td><span class="erp-status processing">生产中</span></td><td>昨天 15:06</td></tr><tr><td>AR-260824</td><td>应收核销</td><td>上海智造系统集成商</td><td class="amount">¥ 96,800</td><td><span class="erp-status done">已核销</span></td><td>08-24 11:20</td></tr></tbody></table></div></article><article class="erp-panel"><div class="erp-panel-head"><h2>待处理流程</h2><span>我的协同事项</span></div><div class="erp-workflow"><div class="erp-workflow-row"><div><strong>PO-260827 · 采购订单审批</strong><small>采购管理部提交 · 需要确认交期与价格</small><div class="erp-workflow-step"><i class="done"></i><i class="current"></i><i></i><i></i></div></div><span class="erp-status pending">待处理</span></div><div class="erp-workflow-row"><div><strong>SO-260828 · 销售订单评审</strong><small>销售中心提交 · 信用额度校验中</small><div class="erp-workflow-step"><i class="done"></i><i class="done"></i><i class="current"></i><i></i></div></div><span class="erp-status processing">流转中</span></div><div class="erp-workflow-row"><div><strong>MO-260826 · 生产领料确认</strong><small>装配一车间 · 物料齐套率 98%</small><div class="erp-workflow-step"><i class="done"></i><i class="done"></i><i class="done"></i><i class="current"></i></div></div><span class="erp-status done">已完成</span></div></div></article></div>
          <div class="erp-bottom-grid"><article class="erp-panel"><div class="erp-panel-head"><h2>库存概览</h2><span>安全库存预警 2 项</span></div><div class="erp-inventory-list"><div class="erp-inventory-row"><div><strong>功率模块 · IGBT-1200V</strong><small>电子元器件 · A-03-18</small></div><div class="erp-stock-bar"><i style="width:82%"></i></div><span class="erp-stock-count">8,240 件</span></div><div class="erp-inventory-row"><div><strong>精密连接器 · KX-08</strong><small>电子元器件 · B-02-06</small></div><div class="erp-stock-bar"><i class="low" style="width:28%"></i></div><span class="erp-stock-count">1,120 件</span></div><div class="erp-inventory-row"><div><strong>晶圆载具 · FOUP-12</strong><small>半导体耗材 · C-01-02</small></div><div class="erp-stock-bar"><i style="width:64%"></i></div><span class="erp-stock-count">486 件</span></div></div></article><article class="erp-panel"><div class="erp-panel-head"><h2>生产协同</h2><span>今日计划 · 6 条</span></div><div style="overflow-x:auto"><table class="erp-table"><thead><tr><th>产线</th><th>工单</th><th>计划</th><th>完成</th><th>达成率</th></tr></thead><tbody><tr><td>SMT-01</td><td>MO-260826</td><td>2,400</td><td>2,280</td><td><span class="erp-status done">95%</span></td></tr><tr><td>装配-03</td><td>MO-260829</td><td>1,800</td><td>1,620</td><td><span class="erp-status processing">90%</span></td></tr><tr><td>封测-02</td><td>MO-260830</td><td>960</td><td>960</td><td><span class="erp-status done">100%</span></td></tr></tbody></table></div></article></div>
          <p class="erp-note">ERP 页面使用样例数据 · 具体业务单据与金额以正式系统记录为准</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_scene_questions(
        "ERP 系统",
        ["采购多少金额需要走什么审批？", "软件安装需要走什么流程？"],
    )


AI_NEWS_FEEDS = (
    {
        "source": "通义千问",
        "url": "https://qwenlm.github.io/blog/index.xml",
        "category": "模型升级",
        "image": "https://qwenlm.github.io/favicon.png",
        "region": "国内",
    },
    {
        "source": "机器之心",
        "url": "https://www.jiqizhixin.com/rss",
        "category": "AI 行业动态",
        "image": "https://www.jiqizhixin.com/favicon.ico",
        "region": "国内",
    },
    {
        "source": "量子位",
        "url": "https://www.qbitai.com/feed",
        "category": "AI 行业动态",
        "image": "https://www.qbitai.com/favicon.ico",
        "region": "国内",
    },
    {
        "source": "InfoQ 中文",
        "url": "https://www.infoq.cn/feed",
        "category": "行业应用",
        "image": "https://www.infoq.cn/favicon.ico",
        "region": "国内",
        "keywords": "人工智能|AI|大模型|智能体|模型|机器学习|深度学习|算力",
    },
    {
        "source": "OpenAI",
        "url": "https://openai.com/news/rss.xml",
        "category": "模型与产品",
        "image": "https://openai.com/favicon.ico",
        "region": "国际",
    },
    {
        "source": "Google AI",
        "url": "https://blog.google/technology/ai/rss/",
        "category": "模型与产品",
        "image": "https://www.gstatic.com/images/branding/product/2x/googleg_96dp.png",
        "region": "国际",
    },
    {
        "source": "NVIDIA",
        "url": "https://blogs.nvidia.com/blog/category/generative-ai/feed/",
        "category": "算力与应用",
        "image": "https://www.nvidia.com/favicon.ico",
        "region": "国际",
    },
    {
        "source": "Hugging Face",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "开源生态",
        "image": "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        "region": "国际",
    },
)

AI_RELEASE_REPOSITORIES = (
    {"repo": "openai/openai-python", "source": "OpenAI GitHub"},
    {"repo": "huggingface/transformers", "source": "Transformers GitHub"},
    {"repo": "vllm-project/vllm", "source": "vLLM GitHub"},
    {"repo": "langchain-ai/langchain", "source": "LangChain GitHub"},
)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _news_text(value: str | None, limit: int = 220) -> str:
    raw = unescape(value or "")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"[#*_`>|\[\]]+", " ", raw)
    cleaned = " ".join(raw.split())
    if len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "…"
    return cleaned


def _news_summary(value: str) -> str:
    text = _news_text(value)
    text = re.sub(r"(?:点击|点开)?(?:查看|阅读)原文[>→↗\s]*", "", text).strip()
    if text in {"查看来源发布的最新 AI 动态与完整信息。", "查看 DeepSeek 官方发布的模型与产品动态。"}:
        return ""
    return text


def _news_source_type(item: dict[str, str]) -> str:
    host = (urlsplit(item["link"]).hostname or "").lower()
    if host == "huggingface.co" and len(urlsplit(item["link"]).path.strip("/").split("/")) > 2:
        return "社区作者文章"
    if item["source"] in {"InfoQ 中文", "量子位", "机器之心"}:
        return "专业媒体"
    if host == "github.com":
        return "官方项目发布"
    return "官方发布"


class _ArticleMetadata(HTMLParser):
    """Read publisher-supplied metadata without executing page scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content") or ""
            if key and content:
                self.meta.setdefault(key, content)
        elif tag == "img":
            candidate = attributes.get("data-src") or attributes.get("src") or ""
            if candidate:
                self.images.append(candidate)


@st.cache_data(ttl=3600, show_spinner=False)
def _article_metadata(link: str) -> dict[str, str]:
    # Only follow articles from the configured publishers, not arbitrary feed URLs.
    host = (urlsplit(link).hostname or "").lower()
    publishers = ("openai.com", "anthropic.com", "blog.google", "nvidia.com", "huggingface.co",
                  "infoq.cn", "qbitai.com", "jiqizhixin.com", "qwenlm.github.io", "deepseek.com")
    if not any(host == domain or host.endswith("." + domain) for domain in publishers):
        return {}
    response = requests.get(link, headers={"User-Agent": "KunkunEnterpriseWorkspace/1.0 (+AI industry news reader)"}, timeout=(3, 5))
    response.raise_for_status()
    parser = _ArticleMetadata()
    parser.feed(response.content.decode("utf-8", errors="replace"))
    meta = parser.meta
    image = urljoin(response.url, meta.get("og:image") or meta.get("twitter:image") or "")
    if host.endswith("qbitai.com") and ("logo" in image.lower() or image == response.url):
        image = next((urljoin(response.url, candidate) for candidate in parser.images
                      if urlsplit(urljoin(response.url, candidate)).hostname == "i.qbitai.com"
                      and "/wp-content/uploads/" in candidate), image)
    return {
        "image": image if image != response.url and image.startswith(("http://", "https://")) else "",
        "summary": _news_summary(meta.get("og:description") or meta.get("description") or ""),
    }


def _enrich_news_item(item: dict[str, str]) -> dict[str, str]:
    result = dict(item)
    result["summary"] = _news_summary(result.get("summary", ""))
    if result.get("image_kind") != "photo" or not result["summary"]:
        try:
            metadata = _article_metadata(result["link"])
        except (requests.RequestException, ValueError, OSError):
            return result
        if metadata.get("image") and result.get("image_kind") != "photo":
            result.update(image=metadata["image"], image_kind="photo")
        if not result["summary"]:
            result["summary"] = metadata.get("summary", "")
    return result


def _news_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return str(value)[:10]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.date().isoformat()


def _entry_text(entry: ElementTree.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in entry.iter():
        if _xml_local_name(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def _entry_link(entry: ElementTree.Element) -> str:
    for child in entry.iter():
        if _xml_local_name(child.tag) != "link":
            continue
        link = (child.attrib.get("href") or child.text or "").strip()
        if link.startswith(("https://", "http://")) and child.attrib.get("rel", "alternate") == "alternate":
            return link
    return ""


def _entry_image(entry: ElementTree.Element, raw_summary: str, fallback: str) -> tuple[str, str]:
    for child in entry.iter():
        local_name = _xml_local_name(child.tag)
        media_type = child.attrib.get("type", "")
        if local_name in {"thumbnail", "content", "enclosure"}:
            candidate = (child.attrib.get("url") or child.attrib.get("href") or "").strip()
            if candidate.startswith(("https://", "http://")) and (
                local_name == "thumbnail" or media_type.startswith("image/")
                or child.attrib.get("medium") == "image"
                or bool(re.search(r"\.(?:jpe?g|png|webp|gif)(?:\?|$)", candidate, re.I))
            ):
                return candidate, "photo"
    match = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)', raw_summary, flags=re.I)
    if match:
        return unescape(match.group(1)), "photo"
    return fallback, "logo"


def _news_category(title: str, summary: str, default: str) -> str:
    text = f"{title} {summary}".lower()
    if any(word in text for word in ("release", "version", "model", "模型", "升级", "gpt", "gemini", "claude", "qwen", "llama", "deepseek")):
        return "模型升级"
    if any(word in text for word in ("github", "open source", "open-source", "开源", "framework", "library", "sdk")):
        return "开源生态"
    if any(word in text for word in ("医疗", "制造", "教育", "金融", "零售", "汽车", "行业", "industry", "health", "manufactur", "enterprise", "science", "societal", "workforce")):
        return "行业应用"
    if any(word in text for word in ("launch", "product", "agent", "api", "发布", "产品", "智能体")):
        return "产品动态"
    return default


def _parse_ai_news_feed(content: bytes, config: dict[str, str]) -> list[dict[str, str]]:
    root = ElementTree.fromstring(content)
    entries = [node for node in root.iter() if _xml_local_name(node.tag) in {"item", "entry"}]
    items: list[dict[str, str]] = []
    for entry in entries[:4]:
        title = _news_text(_entry_text(entry, "title"), limit=150)
        raw_summary = _entry_text(entry, "description", "summary", "encoded", "content")
        summary = _news_summary(raw_summary)
        link = _entry_link(entry)
        if not title or not link:
            continue
        keywords = config.get("keywords")
        if keywords and not re.search(keywords, f"{title} {summary}", flags=re.I):
            continue
        source = config["source"]
        if config.get("use_entry_source"):
            source = _news_text(_entry_text(entry, "source"), limit=40) or source
        image_markup = " ".join(child.text or "" for child in entry.iter()
                                if _xml_local_name(child.tag) in {"description", "summary", "encoded", "content"})
        image, image_kind = _entry_image(entry, image_markup, config["image"])
        items.append(
            {
                "title": title,
                "summary": summary,
                "published": _news_date(_entry_text(entry, "pubdate", "published", "updated", "date")),
                "link": link,
                "source": source,
                "category": _news_category(title, summary, config["category"]),
                "image": image,
                "image_kind": image_kind,
                "region": config.get("region", "国际"),
            }
        )
    return items


def _fetch_ai_news_feed(config: dict[str, str]) -> list[dict[str, str]]:
    response = requests.get(
        config["url"],
        headers={"User-Agent": "KunkunEnterpriseWorkspace/1.0 (+AI industry news reader)"},
        timeout=(3.5, 6),
    )
    response.raise_for_status()
    return _parse_ai_news_feed(response.content, config)


def _fetch_ai_release(config: dict[str, str]) -> list[dict[str, str]]:
    response = requests.get(
        f'https://api.github.com/repos/{config["repo"]}/releases',
        params={"per_page": 2},
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "KunkunEnterpriseWorkspace/1.0",
        },
        timeout=(3.5, 6),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    items: list[dict[str, str]] = []
    owner = config["repo"].split("/", 1)[0]
    for release in payload[:2]:
        title = _news_text(str(release.get("name") or release.get("tag_name") or ""), limit=150)
        link = str(release.get("html_url") or "")
        if not title or not link.startswith(("https://", "http://")):
            continue
        summary = _news_text(str(release.get("body") or "")) or f'{config["repo"]} 发布了新版本，点击查看变更说明。'
        items.append(
            {
                "title": f'{config["repo"]} · {title}',
                "summary": summary,
                "published": _news_date(str(release.get("published_at") or release.get("created_at") or "")),
                "link": link,
                "source": config["source"],
                "category": "开源生态",
                "image": f"https://github.com/{owner}.png?size=480",
                "image_kind": "logo",
                "region": "开源",
            }
        )
    return items


def _fetch_deepseek_news() -> list[dict[str, str]]:
    response = requests.get(
        "https://www.deepseek.com/news/",
        headers={"User-Agent": "Mozilla/5.0 KunkunEnterpriseWorkspace/1.0"},
        timeout=(3.5, 6),
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    items: list[dict[str, str]] = []
    seen_links: set[str] = set()
    for relative_link, block in re.findall(
        r'<a[^>]+href="(/news/[^"#?]+/)"[^>]*>(.*?)</a>', response.text, flags=re.I | re.S
    ):
        link = f"https://www.deepseek.com{relative_link}"
        if link in seen_links:
            continue
        title_match = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", block, flags=re.I | re.S)
        if not title_match:
            continue
        title = _news_text(title_match.group(1), limit=150)
        paragraph_values = [_news_text(value) for value in re.findall(r"<p[^>]*>(.*?)</p>", block, flags=re.I | re.S)]
        summary = next((value for value in reversed(paragraph_values) if len(value) >= 18 and "年" not in value[:12]), "查看 DeepSeek 官方发布的模型与产品动态。")
        date_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", block)
        published = ""
        if date_match:
            published = f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        image_match = re.search(r'<img[^>]+src="([^"]+)"', block, flags=re.I)
        image = "https://www.deepseek.com/favicon.ico"
        image_kind = "logo"
        if image_match:
            image = image_match.group(1)
            if image.startswith("/"):
                image = f"https://www.deepseek.com{image}"
            image_kind = "photo"
        seen_links.add(link)
        items.append(
            {
                "title": title,
                "summary": summary,
                "published": published,
                "link": link,
                "source": "DeepSeek 官方",
                "category": _news_category(title, summary, "模型升级"),
                "image": image,
                "image_kind": image_kind,
                "region": "国内",
            }
        )
        if len(items) >= 4:
            break
    return items


def _parse_anthropic_news(html: str) -> list[dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    for href, block in re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S):
        link = urljoin("https://www.anthropic.com", unescape(href))
        if urlsplit(link).hostname not in {"www.anthropic.com", "anthropic.com"} or not urlsplit(link).path.startswith("/news/"):
            continue
        title_match = re.search(r'<h[1-6][^>]*>(.*?)</h[1-6]>', block, re.S)
        if not title_match:
            title_match = re.search(r'<span[^>]*class="[^"]*__title[^"]*"[^>]*>(.*?)</span>', block, re.S)
        date_match = re.search(r'<time[^>]*>(.*?)</time>', block, re.S)
        if not title_match or not date_match:
            continue
        try:
            published = datetime.strptime(_news_text(date_match.group(1)), "%b %d, %Y").date().isoformat()
        except ValueError:
            continue
        summary_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        title = _news_text(title_match.group(1), 150)
        summary = _news_summary(summary_match.group(1)) if summary_match else ""
        items.setdefault(link, {
            "title": title, "summary": summary, "published": published, "link": link,
            "source": "Anthropic", "category": _news_category(title, summary, "公司动态"),
            "image": "", "image_kind": "logo", "region": "国际",
        })
    return sorted(items.values(), key=lambda row: row["published"], reverse=True)[:4]


def _fetch_anthropic_news() -> list[dict[str, str]]:
    response = requests.get("https://www.anthropic.com/news",
                            headers={"User-Agent": "KunkunEnterpriseWorkspace/1.0 (+AI industry news reader)"},
                            timeout=(3.5, 6))
    response.raise_for_status()
    return _parse_anthropic_news(response.content.decode("utf-8", errors="replace"))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ai_frontier_news() -> tuple[list[dict[str, str]], str | None]:
    """Aggregate AI company, product, open-source and industry news from independent public sources."""
    jobs: list[tuple[Any, dict[str, str]]] = []
    for config in AI_NEWS_FEEDS:
        jobs.append((_fetch_ai_news_feed, config))
    for config in AI_RELEASE_REPOSITORIES:
        jobs.append((_fetch_ai_release, config))
    jobs.append((_fetch_deepseek_news, {"source": "DeepSeek 官方"}))
    jobs.append((_fetch_anthropic_news, {"source": "Anthropic"}))

    collected: list[dict[str, str]] = []
    executor = ThreadPoolExecutor(max_workers=min(16, len(jobs)))
    futures = {
            executor.submit(loader) if loader in {_fetch_deepseek_news, _fetch_anthropic_news} else executor.submit(loader, config): config["source"]
            for loader, config in jobs
        }
    completed, pending = wait(futures, timeout=8.5)
    for future in completed:
        try:
            collected.extend(future.result())
        except (requests.RequestException, ElementTree.ParseError, ValueError, TypeError, OSError):
            continue
    for future in pending:
        future.cancel()
    executor.shutdown(wait=False, cancel_futures=True)

    sorted_items = sorted(collected, key=lambda row: row.get("published", ""), reverse=True)
    unique: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    source_counts: dict[str, int] = {}

    def append_if_eligible(item: dict[str, str]) -> bool:
        title_key = re.sub(r"\W+", "", item["title"].lower())
        source = item["source"]
        if not title_key or title_key in seen_titles or source_counts.get(source, 0) >= 2:
            return False
        seen_titles.add(title_key)
        source_counts[source] = source_counts.get(source, 0) + 1
        unique.append(item)
        return True

    # Keep domestic coverage, then reserve an article for each requested company.
    for region in ("国内",):
        region_items = [item for item in sorted_items if item.get("region") == region]
        region_sources: set[str] = set()
        region_added = 0
        for item in region_items:
            if item["source"] in region_sources:
                continue
            if append_if_eligible(item):
                region_sources.add(item["source"])
                region_added += 1
            if region_added >= 3:
                break
    for source in ("OpenAI", "Anthropic", "Google AI"):
        candidate = next((item for item in sorted_items if item["source"] == source), None)
        if candidate:
            append_if_eligible(candidate)
    for item in sorted_items:
        if len(unique) >= 9:
            break
        if item["source"] not in source_counts:
            append_if_eligible(item)
    for item in sorted_items:
        if len(unique) >= 9:
            break
        append_if_eligible(item)

    if unique:
        # Enrich only selected stories; one slow article cannot block other covers.
        enrichment = ThreadPoolExecutor(max_workers=9)
        details = {enrichment.submit(_enrich_news_item, item): index for index, item in enumerate(unique)}
        completed_details, pending_details = wait(details, timeout=7)
        for future in completed_details:
            try:
                unique[details[future]] = future.result()
            except (requests.RequestException, ValueError, OSError):
                pass
        for future in pending_details:
            future.cancel()
        enrichment.shutdown(wait=False, cancel_futures=True)
        return unique, None
    return [], "暂时无法连接外部资讯源，请检查网络后刷新重试。"


def render_ai_news_page() -> None:
    """Render live AI industry news aggregated from public sources."""
    news, error = fetch_ai_frontier_news()
    if error:
        st.warning(error)
    cards = [_news_card_markup(item, portal=False) for item in news]
    if not cards:
        cards.append('<article class="news-card"><div class="news-card-body"><span class="tag">连接失败</span><h3>暂时没有获取到 AI 行业资讯</h3><p>外部资讯源当前均不可达，请检查网络后刷新重试。</p></div></article>')
    st.markdown(
        f"""
        <section class="news-hero">
          <p class="module-kicker">{COMPANY_NAME} · LIVE MULTI-SOURCE</p>
          <h1>AI 行业资讯</h1>
          <p>聚合 AI 公司官方动态、模型与产品升级、开源项目发布和行业应用报道；每条资讯均保留来源、发布日期与原文链接。</p>
          {_official_news_links()}
          <div class="news-grid">{"".join(cards)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_question_composer(*, compact: bool) -> str | None:
    # Basic form widgets avoid the separately lazy-loaded ChatInput bundle.
    key = "assistant-composer" if compact else "chat-composer"
    with st.container(key=key):
        if compact:
            st.markdown(
                '<p class="assistant-dialog-intro">继续问我，或换个问题试试</p>',
                unsafe_allow_html=True,
            )
        with st.form(f"{key}-form", clear_on_submit=True, border=False):
            field, action = st.columns([8, 1], gap="small")
            with field:
                question = st.text_input(
                    "输入问题", placeholder="例如：事假怎么申请、出差怎么报销…",
                    key=f"{key}-question", label_visibility="collapsed",
                )
            with action:
                submitted = st.form_submit_button("", icon=":material/arrow_upward:", help="发送", type="primary")
    return (question.strip() or None) if submitted else None


def render_chat_tab(api_url: str, *, show_sources: bool = False, compact: bool = False) -> None:
    if not compact:
        st.markdown(
            """
            <section class="section-heading">
              <div><p class="section-overline">员工自助</p><h2>直接问，不必先找到制度。</h2></div>
              <p>用日常语言描述你的场景，系统会返回清晰、可执行的答复。</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        examples = [
            "忘记打卡后几天内可以补卡？",
            "上海出差住宿标准是多少？",
            "VPN-403 是什么意思，下一步怎么办？",
            "公司电脑安装生成式 AI 工具要走什么流程？",
        ]
        st.markdown('<p class="suggestion-caption">从这些真实工作场景开始</p>', unsafe_allow_html=True)
        columns = st.columns(2, gap="small")
        for index, example in enumerate(examples):
            if columns[index % 2].button(example, key=f"example-{index}"):
                st.session_state.pending_question = example

    if "messages" not in st.session_state:
        st.session_state.messages = []

    turns = conversation_turns_oldest_first(st.session_state.messages)
    for turn_index, (user_message, assistant_message) in enumerate(turns):
        # This anchor exists in the completed rerun, not only during the
        # request.  It lets the dialog show the newest question first while
        # preserving all earlier turns above it for manual upward scrolling.
        if compact and turn_index == len(turns) - 1:
            st.markdown('<div id="assistant-latest-turn"></div>', unsafe_allow_html=True)
        with st.chat_message(user_message["role"], avatar=assistant_user_avatar()):
            st.markdown(user_message.get("content", ""))
        if assistant_message is None:
            continue
        with st.chat_message(assistant_message["role"], avatar=str(MASCOT_PATH) if MASCOT_PATH.is_file() else "🤖"):
            answer_text = employee_answer_text(assistant_message.get("content", ""))
            st.markdown(
                format_answer_html(answer_text, show_overline=show_sources),
                unsafe_allow_html=True,
            )
            if assistant_message.get("warning"):
                st.warning(assistant_message["warning"])
            if show_sources and assistant_message.get("sources"):
                for index, source in enumerate(assistant_message["sources"], start=1):
                    render_source(source, index)
            render_feedback_controls(
                api_url,
                user_message.get("content", ""),
                answer_text,
                feedback_key_for_message(assistant_message, turn_index),
            )

    if compact and turns:
        scroll_assistant_to_latest_turn()

    typed_question = render_question_composer(compact=compact)
    question = typed_question or st.session_state.pop("pending_question", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    history = conversation_history_for_api(st.session_state.messages[:-1])
    with st.chat_message("user", avatar=assistant_user_avatar()):
        st.markdown(question)
    with st.chat_message("assistant", avatar=str(MASCOT_PATH) if MASCOT_PATH.is_file() else "🤖"):
        try:
            with st.spinner("正在检索制度并整理答案…"):
                result = api_request(
                    "POST",
                    api_url,
                    "/chat",
                    json={"question": question, "history": history},
                )
            answer_text = employee_answer_text(result["answer"])
            feedback_id = uuid.uuid4().hex
            st.markdown(
                format_answer_html(answer_text, show_overline=show_sources),
                unsafe_allow_html=True,
            )
            warning = llm_fallback_warning(result)
            if warning:
                st.warning(warning)
            if show_sources:
                for index, source in enumerate(result.get("sources", []), start=1):
                    render_source(source, index)
            render_feedback_controls(
                api_url, question, answer_text, f"message-{feedback_id}"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "question": question,
                    "feedback_id": feedback_id,
                    "sources": result.get("sources", []),
                    "warning": warning,
                }
            )
            st.rerun()
        except requests.RequestException as exc:
            message = f"问答服务暂时不可用：{exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "question": question, "feedback_id": uuid.uuid4().hex})
            st.rerun()


def render_status_tab(api_url: str) -> None:
    st.markdown(
        """
        <section class="section-heading">
          <div><p class="section-overline">知识资产</p><h2>每一份制度，都有可见状态。</h2></div>
          <p>查看入库结果、解析元数据、索引配置与反馈信号，确保知识数据处于可用状态。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    try:
        health = fetch_health(api_url)
        documents = fetch_documents(api_url)
    except requests.RequestException as exc:
        st.error(f"无法读取知识库状态：{exc}")
        return

    metrics = st.columns(4)
    metrics[0].metric("文档数", health.get("document_count", 0))
    metrics[1].metric("文本块", health.get("chunk_count", 0))
    metrics[2].metric("评测问题", health.get("evaluation_case_count", 0))
    metrics[3].metric("运行状态", "正常" if health.get("status") == "ok" else "降级可用")

    left, right = st.columns(2)
    with left:
        st.markdown("**检索配置**")
        st.write(f"Embedding：`{health.get('effective_embedding_provider', '-')}`")
        st.write(f"向量后端：`{health.get('vector_backend', '-')}`")
        st.write(f"Reranker：`{reranker_status_label(health)}`")
    with right:
        st.markdown("**回答配置**")
        st.write(f"LLM：`{health.get('effective_llm_provider', health.get('llm_provider', '-'))}`")
        st.write(f"索引时间：`{health.get('built_at') or '尚未记录'}`")
        if health.get("errors"):
            st.warning(f"降级信息：{health['errors']}")
    try:
        feedback = api_request("GET", api_url, "/feedback/summary")
        st.caption(f"反馈闭环：{feedback['total']} 条评价，点踩 {feedback['down']} 条，未解决 {feedback['unresolved']} 条")
    except requests.RequestException:
        pass

    table = [
        {
            "文档": item["title"],
            "分类": item["category"],
            "文件": item["source_path"],
            "版本": item.get("version") or "未填写",
            "生效日期": item.get("effective_date") or "未填写",
            "发布部门": item.get("department") or "未填写",
            "负责人": item.get("owner") or "未填写",
            "状态": "可用" if item["status"] == "ready" else "异常",
            "章节数": item["section_count"],
            "文本块": item["chunk_count"],
        }
        for item in documents
    ]
    render_data_table(table)

    if st.button("重新构建知识库索引", type="secondary"):
        try:
            with st.spinner("正在重新解析文档并构建索引…"):
                result = api_request("POST", api_url, "/knowledge-base/rebuild")
            fetch_health.clear()
            fetch_documents.clear()
            st.success(result["message"])
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"重建失败：{exc}")


def render_contract_review_tab(api_url: str, role: str) -> None:
    """Render the isolated contract-review MVP with explicit local roles."""
    if role not in CONTRACT_REVIEW_ROLES:
        st.markdown(
            '<section class="section-heading"><h2>无权查看合同评审</h2><p>当前身份不可用。</p></section>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        """
        <section class="section-heading">
          <div><p class="section-overline">员工工作台</p><h2>合同评审</h2></div>
          <p>定位关键条款、待确认内容与原文依据，不构成法律结论；按当前身份过滤敏感内容。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    can_upload = role in {"legal", "admin"}
    st.caption(
        f"当前身份：{role_display_name(role)}。上传时选择的是这份合同允许哪些预置身份查看，不是新增角色。"
    )

    with st.expander("上传合同", expanded=False):
        if not can_upload:
            st.info("仅法务或领导身份可上传；其他身份只能复核分配给自己的合同。")
        uploaded = st.file_uploader(
            "合同文件",
            type=["pdf", "docx", "md", "markdown", "txt"],
            key="contract-upload",
            disabled=not can_upload,
            help="支持可复制文本的 PDF、Word、Markdown 和 TXT；扫描版 PDF 暂不支持 OCR。",
        )
        default_roles = [role] if role != "admin" else ["legal"]
        allowed_roles = st.multiselect(
            "可见身份范围",
            tuple(ROLE_PROFILES),
            default=default_roles,
            format_func=role_display_name,
            key="contract-allowed-roles",
            disabled=not can_upload,
            help="只有选中的身份能在合同列表和审阅结果中看到这份合同；领导身份可查看全部合同。",
        )
        if st.button("上传合同", type="secondary", disabled=not can_upload or uploaded is None or not allowed_roles):
            try:
                suffix = Path(uploaded.name).suffix.lower()
                payload: dict[str, Any] = {
                    "filename": uploaded.name,
                    "allowed_roles": allowed_roles,
                    "actor_role": role,
                }
                file_bytes = uploaded.getvalue()
                if len(file_bytes) > 18_000_000:
                    st.error("单份合同不能超过 18 MB，请拆分后上传。")
                    return
                if suffix in {".md", ".markdown", ".txt"}:
                    payload["content"] = file_bytes.decode("utf-8-sig")
                else:
                    payload["content_base64"] = base64.b64encode(file_bytes).decode("ascii")
                api_request(
                    "POST",
                    api_url,
                    "/contracts",
                    json=payload,
                )
                st.success("合同已进入独立合同库，可在授权范围内审阅。")
                st.rerun()
            except UnicodeDecodeError:
                st.error("合同文本必须使用 UTF-8 编码。")
            except requests.RequestException as exc:
                st.error(f"上传失败：{exc}")

    try:
        contracts = fetch_contracts(api_url, role)
    except requests.RequestException as exc:
        st.error(f"无法读取合同库：{exc}")
        return
    if not contracts:
        st.info("当前角色没有可审阅的合同。")
        return

    labels = {item["document_id"]: item["title"] for item in contracts}
    document_id = st.selectbox("可审阅合同", list(labels), format_func=labels.get, key="contract-document")
    query = st.text_input("优先定位的内容（可选）", placeholder="例如：合同金额、开户行、履约保证金", key="contract-query")
    version = next(item.get("document_version", "") for item in contracts if item["document_id"] == document_id)
    scope = (role, document_id, version, query)
    if st.button("开始审阅", type="primary"):
        st.session_state.pop("contract-review-result", None)
        try:
            result = api_request(
                "POST", api_url, f"/contracts/{quote(document_id, safe='')}/review",
                json={"role": role, "query": query},
            )
            st.session_state["contract-review-result"] = {"scope": scope, "result": result}
        except requests.RequestException:
            st.error("合同不可用。")
            return
    cached = st.session_state.get("contract-review-result", {})
    if cached.get("scope") != scope:
        return
    result = cached["result"]

    fields = result.get("fields", [])
    counts = result.get("review_counts", {})
    st.subheader("本次审了什么")
    st.caption("系统只做条款定位与缺失/含糊表达初筛；签署、修改或合规判断仍由责任部门人工完成。")
    checked = counts.get("checked", len(fields))
    located = counts.get("located", sum(item.get("status") == "found" for item in fields))
    pending = counts.get("pending", max(checked - located, 0))
    metric_checked, metric_located, metric_pending = st.columns(3)
    metric_checked.metric("已检查", f"{checked} 类")
    metric_located.metric("已定位原文", f"{located} 项")
    metric_pending.metric("信息不完整/待确认", f"{pending} 项")

    st.subheader("需要人工确认")
    prompts = result.get("risk_prompts", [])
    if prompts:
        for prompt in prompts:
            st.warning(prompt["message"])
    else:
        st.info(
            f"{checked} 类条款均已定位，未发现本轮规则可识别的缺失或明显含糊表述。"
            "这不是“合同没有问题”：请逐项核对下方列出的具体要点是否写清楚。"
        )

    st.subheader("逐项定位结果")
    st.caption("人工复核记录仅保存在本次会话，未签名、未服务端归档；请下载后按内部流程交接。")
    decisions: dict[str, dict[str, str]] = {}
    for item in fields:
        status = item.get("status")
        located_label = {
            "found": "已定位，关键信息可见",
            "incomplete": "已找到相关表述，但信息不完整",
            "missing": "未定位，需补充确认",
        }.get(status, "待确认")
        with st.expander(f"{item['label']} · {located_label}", expanded=status != "found"):
            st.markdown(f"**这项在核对：** {item.get('review_focus', '相关约定是否明确。')}")
            if status == "incomplete":
                st.warning(item.get("issue") or "这不是完整定位：请核对上面的关键要点是否在合同中明确写出。")
            if item.get("evidence"):
                sections = item.get("evidence_sections", [])
                st.caption("原文位置：" + "、".join(sections) if sections else "未取得可靠页码或章节，请打开完整文件核对。")
                st.markdown("**定位到的原文：**")
                st.write(item["evidence"])
            else:
                st.warning("未在已解析文本中定位到这一类条款，请回看完整合同并与责任部门确认。")
            field_key = f"contract-decision-{role}-{document_id}-{result.get('document_version', '')}-{item['field']}"
            decision = st.selectbox("人工复核", ("待复核", "已确认", "需补充"), key=field_key)
            note = st.text_area("处理备注", max_chars=2000, key=f"{field_key}-note")
            decisions[item["field"]] = {"status": decision, "note": note}
    st.download_button(
        "下载核对记录", export_contract_review(result, decisions),
        file_name="合同条款核对记录.md", mime="text/markdown",
    )
    st.caption("自动定位可能遗漏、截断或误选相邻条款；“已定位”不等于约定完整或已通过审阅。")
    related_sources = result.get("related_policy_sources", [])
    if related_sources:
        st.subheader("关联制度检索结果（需人工核验适用性）")
        for index, source in enumerate(related_sources, start=1):
            with st.expander(f"{index}. {source['title']} · {source['section']}"):
                st.text(source["content_preview"])


def format_evaluation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep refusal cases readable instead of presenting them as failed hits."""
    if row.get("should_refuse"):
        refusal_correct = row.get("refusal_correct", row.get("failure_type") == "passed")
        rank = "不应召回" if refusal_correct else "误召回"
        hit_at_3 = "拒答通过" if refusal_correct else "拒答失败"
    else:
        rank = str(row.get("rank")) if row.get("rank") else "未召回"
        hit_at_3 = "通过" if row.get("hit_at_3") else "失败"
    return {
        "案例": row["case_id"],
        "问题": row["question"],
        "预期文档": row.get("expected_source") or "无（应拒答）",
        "排名": rank,
        "Hit@3": hit_at_3,
        "失败类型": row["failure_type"],
    }


def render_evaluation_tab(api_url: str) -> None:
    st.markdown(
        """
        <section class="section-heading">
          <div><p class="section-overline">质量验证</p><h2>让每次回答，都经得起复盘。</h2></div>
          <p>用固定员工问题检验正确制度是否被召回，并将无答案场景与有效答案分开衡量。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button("运行固定评测", type="primary"):
        try:
            with st.spinner("正在逐条检索评测问题…"):
                st.session_state.evaluation_report = api_request("POST", api_url, "/evaluation/run")
        except requests.RequestException as exc:
            st.error(f"评测执行失败：{exc}")

    report = st.session_state.get("evaluation_report")
    if not report:
        st.info("点击按钮后会显示整体指标、每条问题排名和失败案例。")
        return

    metrics = st.columns(4)
    metrics[0].metric("案例数", report["total_cases"])
    metrics[1].metric("Hit@1", f"{report['hit_at_1']:.1%}")
    metrics[2].metric("Hit@3", f"{report['hit_at_3']:.1%}")
    metrics[3].metric("MRR", f"{report['mrr']:.3f}")
    extra = st.columns(5)
    extra[0].metric("拒答准确率", f"{report.get('refusal_accuracy', 0):.1%}")
    extra[1].metric("引用覆盖率", f"{report.get('citation_coverage', 0):.1%}")
    extra[2].metric("文档覆盖率", f"{report.get('document_coverage', 0):.1%}")
    extra[3].metric("P50(ms)", report.get("latency_p50_ms", 0))
    extra[4].metric("P95(ms)", report.get("latency_p95_ms", 0))

    rows = [
        format_evaluation_row(row)
        for row in report["cases"]
    ]
    render_data_table(rows)

    failed = [row for row in report["cases"] if row["failure_type"] != "passed"]
    if failed:
        st.error(f"发现 {len(failed)} 条失败案例，可据此调整语料、切块或检索参数。")
        for row in failed:
            with st.expander(f"{row['case_id']}｜{row['question']}"):
                st.write(f"预期文档：`{row['expected_source']}`")
                st.write(f"实际排序：{row['ranked_sources']}")
                st.write(f"失败类型：`{row['failure_type']}`")
    else:
        st.success("固定案例全部通过：有答案案例命中前三名，无答案案例均正确拒答。")


def main() -> None:
    st.set_page_config(page_title=f"{COMPANY_NAME}企业 AI 协作工作台", page_icon="K", layout="wide")
    inject_workspace_style()
    api_url = DEFAULT_API_URL
    sidebar_logo = f'<span class="company-mark">鲲</span>'
    if SIDEBAR_LOGO_PATH.is_file():
        encoded_logo = base64.b64encode(SIDEBAR_LOGO_PATH.read_bytes()).decode("ascii")
        sidebar_logo = f'<img class="sidebar-logo" src="data:image/png;base64,{encoded_logo}" alt="{COMPANY_NAME} Logo" />'

    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand-row">{sidebar_logo}<div><div class="sidebar-brand">{COMPANY_NAME}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">员工工作台</div>', unsafe_allow_html=True)
        session_state = getattr(st, "session_state", {})
        selected_role = str(session_state.get("active_role", "employee"))
        session_state.setdefault("_assistant_role", selected_role)
        visible_employee_pages = employee_pages_for_role(selected_role)
        if session_state.get("employee_nav") is not None and session_state.get("employee_nav") not in visible_employee_pages:
            session_state["employee_nav"] = "概览"
        # Existing browser sessions may still point at the former operations entry.
        if session_state.get("operations_nav") is not None and session_state.get("operations_nav") not in OPERATIONS_PAGES:
            session_state["operations_nav"] = None
            session_state["employee_nav"] = "合同评审" if selected_role in CONTRACT_REVIEW_ROLES else "概览"
        page = st.radio("员工工作台", visible_employee_pages, label_visibility="collapsed", key="employee_nav", on_change=_select_employee_nav)
        st.markdown('<div class="sidebar-label">知识库运营</div>', unsafe_allow_html=True)
        operations_page = st.radio("知识库运营", OPERATIONS_PAGES, label_visibility="collapsed", key="operations_nav", index=None, on_change=_select_operations_nav)
        st.divider()
        role = render_sidebar_user_card(selected_role)

    active_page = operations_page or page
    if operations_page == "知识库状态":
        render_status_tab(api_url)
    elif operations_page == "基础评测":
        render_evaluation_tab(api_url)
    elif page == "合同评审":
        render_contract_review_tab(api_url, role)
    elif page == "概览":
        render_company_overview()
    elif page == "人事服务":
        render_hr_services_page()
    elif page == "财务中心":
        render_finance_center_page()
    elif page == "ERP 系统":
        render_erp_page()
    elif page == "个人中心":
        render_personal_center(role)
    render_global_assistant(api_url, current_page=active_page)


if __name__ == "__main__":
    main()
