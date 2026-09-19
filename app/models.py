"""Embedding and language-model provider adapters."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np
import requests

from app.utils import normalize_text, tokenize_zh


class LocalHashEmbeddings:
    """Small deterministic embedding model for API-free local validation.

    It uses feature hashing over Chinese tokens and character bigrams. The
    vectors are useful for validating the retrieval pipeline, but are not
    presented as a replacement for a production semantic embedding model.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be greater than zero")
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        normalized = normalize_text(text).lower()
        compact = "".join(normalized.split())
        bigrams = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
        features = tokenize_zh(normalized) + bigrams
        vector = np.zeros(self.dim, dtype=np.float32)

        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += sign

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class ChatClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaEmbeddings:
    """Minimal Ollama embeddings adapter using its local HTTP API."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise RuntimeError("Ollama 未返回 embedding")
        return [float(value) for value in embedding]


class OpenAICompatibleEmbeddings:
    """Embedding adapter for OpenAI-compatible endpoints."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise RuntimeError("Embedding 接口返回数量与输入不一致")
        return [[float(value) for value in item["embedding"]] for item in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama 未返回回答")
        return content


class OpenAICompatibleChatClient:
    """Chat adapter usable with DeepSeek and other compatible services."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI 兼容接口未返回回答")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI 兼容接口未返回有效回答")
        return content.strip()


def build_embedding_model(settings: Any):
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return LocalHashEmbeddings()
    if provider == "ollama":
        return OllamaEmbeddings(settings.ollama_base_url, settings.embedding_model)
    if provider in {"openai", "openai_compatible"}:
        if not settings.openai_base_url:
            raise ValueError("使用 OpenAI 兼容 Embedding 时必须配置 OPENAI_BASE_URL")
        return OpenAICompatibleEmbeddings(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.embedding_model,
        )
    raise ValueError(f"不支持的 Embedding 提供方：{settings.embedding_provider}")


def build_chat_client(settings: Any) -> ChatClient | None:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaChatClient(settings.ollama_base_url, settings.llm_model)
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("使用 DeepSeek 时必须配置 DEEPSEEK_API_KEY")
        return OpenAICompatibleChatClient(settings.deepseek_base_url, settings.deepseek_api_key, settings.deepseek_model)
    if provider in {"openai", "openai_compatible"}:
        if not settings.openai_api_key:
            raise ValueError("使用 OpenAI 时必须配置 OPENAI_API_KEY")
        return OpenAICompatibleChatClient(settings.openai_base_url, settings.openai_api_key, settings.openai_model)
    raise ValueError(f"不支持的大模型提供方：{settings.llm_provider}")
