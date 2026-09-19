"""Shared text and file utilities."""

from __future__ import annotations

import re
from pathlib import Path

import jieba


def normalize_text(text: str) -> str:
    """Collapse redundant whitespace while retaining meaningful line breaks."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF text layers can contain invisible control characters (notably NUL)
    # around symbols such as inequality signs. They must not reach chunks or
    # index persistence because they break JSON/text consumers downstream.
    normalized = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", normalized)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)


def tokenize_zh(text: str) -> list[str]:
    """Tokenize Chinese text and keep English codes and numbers searchable."""

    normalized = normalize_text(text).lower().replace("_", " ").replace("-", " ")
    tokens: list[str] = []
    for raw_token in jieba.lcut(normalized):
        token = raw_token.strip()
        if not token:
            continue
        parts = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", token)
        tokens.extend(parts)
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split Chinese policy text into compact answer candidates."""

    normalized = normalize_text(text)
    lines = normalized.splitlines()
    if len(lines) > 1:
        merged_lines: list[str] = []
        buffer = ""
        for line in lines:
            starts_new_item = bool(
                re.match(r"^(?:[-*•]\s*|\d+[.、)]\s*|[一二三四五六七八九十]+[、.])", line)
            )
            if buffer and (starts_new_item or re.search(r"[。！？!?；;]$", buffer)):
                merged_lines.append(buffer)
                buffer = line
            elif buffer:
                buffer += line
            else:
                buffer = line
        if buffer:
            merged_lines.append(buffer)
        normalized = "\n".join(merged_lines)
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", normalized)
    return [part.strip(" -\t") for part in parts if part.strip(" -\t")]


def ensure_directory(path: Path) -> Path:
    """Create a directory when missing and return it for convenient chaining."""

    path.mkdir(parents=True, exist_ok=True)
    return path
