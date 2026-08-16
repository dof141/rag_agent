from collections.abc import Mapping
from typing import Any

import requests

from app.conf.embedding_config import EmbeddingConfig
from app.embedding.interface import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingResult,
    EmbeddingTimeoutError,
    validate_texts,
)


class SiliconFlowEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, http_client: Any = None):
        if not config.api_key:
            raise EmbeddingConfigurationError("SiliconFlow embedding 缺少 API Key")
        if config.dimension <= 0 or config.batch_size <= 0 or config.request_timeout <= 0:
            raise EmbeddingConfigurationError("SiliconFlow embedding 数值配置无效")
        self._config = config
        self._http_client = http_client or requests
        self._url = f"{config.base_url.rstrip('/')}/embeddings"

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        validate_texts(texts)
        dense: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            dense.extend(self._embed_batch(texts[start : start + self._config.batch_size]))
        return {"dense": dense}

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._http_client.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._config.model,
                    "input": texts,
                    "encoding_format": "float",
                },
                timeout=self._config.request_timeout,
            )
        except requests.Timeout as exc:
            raise EmbeddingTimeoutError("SiliconFlow embedding 请求超时") from exc
        except Exception as exc:
            raise EmbeddingRequestError("SiliconFlow embedding 请求失败") from exc

        status_code = getattr(response, "status_code", None)
        if status_code in (401, 403):
            raise EmbeddingAuthenticationError("SiliconFlow embedding 认证失败")
        if status_code == 429:
            raise EmbeddingRateLimitError("SiliconFlow embedding 请求被限流")
        if status_code is None or not 200 <= status_code < 300:
            raise EmbeddingRequestError(f"SiliconFlow embedding 返回 HTTP {status_code}")
        try:
            payload = response.json()
        except Exception as exc:
            raise EmbeddingResponseError("SiliconFlow embedding 响应不是有效 JSON") from exc
        return self._parse_response(payload, expected_count=len(texts))

    def _parse_response(self, payload: Any, *, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, Mapping):
            raise EmbeddingResponseError("SiliconFlow embedding 响应必须是对象")
        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingResponseError("SiliconFlow embedding 响应缺少 data")
        if len(data) != expected_count:
            raise EmbeddingResponseError("SiliconFlow embedding 响应数量与请求文本数量不匹配")

        ordered: list[Any] = [None] * expected_count
        for item in data:
            if not isinstance(item, Mapping):
                raise EmbeddingResponseError("SiliconFlow embedding 响应条目格式无效")
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingResponseError("SiliconFlow embedding 响应缺少有效 index") from exc
            if index < 0 or index >= expected_count or ordered[index] is not None:
                raise EmbeddingResponseError("SiliconFlow embedding 响应包含重复或越界 index")
            ordered[index] = item
        if any(item is None for item in ordered):
            raise EmbeddingResponseError("SiliconFlow embedding 响应缺少部分 index")

        dense_vectors: list[list[float]] = []
        for position, item in enumerate(ordered):
            dense = item.get("embedding")
            if not isinstance(dense, list):
                raise EmbeddingResponseError("SiliconFlow embedding 响应缺少 embedding")
            if len(dense) != self._config.dimension:
                raise EmbeddingResponseError(
                    f"SiliconFlow embedding 第 {position} 项稠密向量维度错误"
                )
            try:
                dense_vectors.append([float(value) for value in dense])
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError("SiliconFlow embedding 向量值格式无效") from exc
        return dense_vectors
