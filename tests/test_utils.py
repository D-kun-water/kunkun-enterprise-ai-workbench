from app.utils import normalize_text, split_sentences, tokenize_zh


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  年假  \n\n  申请 ") == "年假\n申请"


def test_split_sentences_joins_pdf_wrapped_lines_without_punctuation() -> None:
    text = "VPN-403 表示远程访问被策略拒绝，常见原因包括权限未获批、MFA\n未完成、终端不合规。\n2. 请通过 IT-SERVICE 提交工单。"

    assert split_sentences(text) == [
        "VPN-403 表示远程访问被策略拒绝，常见原因包括权限未获批、MFA未完成、终端不合规。",
        "2. 请通过 IT-SERVICE 提交工单。",
    ]


def test_tokenize_zh_keeps_codes_and_numbers() -> None:
    tokens = tokenize_zh("VPN-403 报错，端口 11434")
    assert "vpn" in tokens
    assert "403" in tokens
    assert "11434" in tokens
