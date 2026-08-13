from collections.abc import Mapping
from typing import Any

from app.conf.embedding_config import EmbeddingConfig
from app.embedding.interface import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingResult,
    validate_texts,
)


_ENDPOINT = "services/embeddings/text-embedding/text-embedding"


class DashScopeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, http_client: Any = None):
        if not config.api_key:
            raise EmbeddingConfigurationError(
                "DashScope embedding 缺少 DASHSCOPE_API_KEY"
            )
        if config.batch_size <= 0:
            raise EmbeddingConfigurationError("EMBEDDING_BATCH_SIZE 必须大于 0")
        if config.dimension <= 0:
            raise EmbeddingConfigurationError("EMBEDDING_DIMENSION 必须大于 0")
        if config.output_type != "dense&sparse":
            raise EmbeddingConfigurationError(
                "DashScope 混合检索要求 EMBEDDING_OUTPUT_TYPE=dense&sparse"
            )

        if http_client is None:
            import requests

            http_client = requests

        self._config = config
        self._http_client = http_client
        self._url = f"{config.base_url.rstrip('/')}/{_ENDPOINT}"

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        validate_texts(texts)
        result: EmbeddingResult = {"dense": [], "sparse": []}

        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start : start + self._config.batch_size]
            batch_result = self._embed_batch(batch)
            result["dense"].extend(batch_result["dense"])
            result["sparse"].extend(batch_result["sparse"])

        return result

    def _embed_batch(self, texts: list[str]) -> EmbeddingResult:
        body = {
            "model": self._config.model,
            "input": {"texts": texts},
            "parameters": {
                "dimension": self._config.dimension,
                "output_type": self._config.output_type,
            },
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._http_client.post(
                self._url,
                headers=headers,
                json=body,
                timeout=self._config.request_timeout,
            )
        except Exception as exc:
            raise EmbeddingRequestError(
                f"DashScope embedding 请求失败: {exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)
        if status_code is None or not 200 <= status_code < 300:
            detail = getattr(response, "text", "")
            raise EmbeddingRequestError(
                f"DashScope embedding 请求返回 HTTP {status_code}: {detail}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise EmbeddingResponseError(
                "DashScope embedding 响应不是有效 JSON"
            ) from exc

        return self._parse_response(payload, expected_count=len(texts))

    def _parse_response(
        self,
        payload: Any,
        *,
        expected_count: int,
    ) -> EmbeddingResult:
        if not isinstance(payload, Mapping):
            raise EmbeddingResponseError("DashScope embedding 响应必须是对象")
        output = payload.get("output")
        if not isinstance(output, Mapping):
            raise EmbeddingResponseError("DashScope embedding 响应缺少 output")
        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError(
                "DashScope embedding 响应缺少 output.embeddings"
            )
        if len(embeddings) != expected_count:
            raise EmbeddingResponseError(
                "DashScope embedding 响应数量与请求文本数量不匹配: "
                f"期望 {expected_count}，实际 {len(embeddings)}"
            )

        ordered = self._restore_input_order(embeddings, expected_count)
        dense_vectors: list[list[float]] = []
        sparse_vectors: list[dict[int, float]] = []

        for position, item in enumerate(ordered):
            if not isinstance(item, Mapping):
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项不是对象"
                )
            dense = item.get("embedding")
            sparse = item.get("sparse_embedding")
            if not isinstance(dense, list):
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项缺少 embedding"
                )
            if not isinstance(sparse, list):
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项缺少 sparse_embedding"
                )
            if len(dense) != self._config.dimension:
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项稠密向量维度错误: "
                    f"期望 {self._config.dimension}，实际 {len(dense)}"
                )

            try:
                dense_vectors.append([float(value) for value in dense])
                sparse_vector = {
                    int(entry["index"]): float(entry["value"])
                    for entry in sparse
                    if isinstance(entry, Mapping)
                    and "index" in entry
                    and "value" in entry
                }
            except (TypeError, ValueError, KeyError) as exc:
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项向量值格式无效"
                ) from exc

            if len(sparse_vector) != len(sparse):
                raise EmbeddingResponseError(
                    f"DashScope embedding 第 {position} 项稀疏向量格式无效"
                )
            sparse_vectors.append(sparse_vector)

        return {"dense": dense_vectors, "sparse": sparse_vectors}

    @staticmethod
    def _restore_input_order(
        embeddings: list[Any],
        expected_count: int,
    ) -> list[Any]:
        has_indexes = [
            isinstance(item, Mapping) and "text_index" in item
            for item in embeddings
        ]
        if not any(has_indexes):
            return embeddings
        if not all(has_indexes):
            raise EmbeddingResponseError(
                "DashScope embedding 响应中的 text_index 不完整"
            )

        ordered: list[Any] = [None] * expected_count
        for item in embeddings:
            try:
                index = int(item["text_index"])
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError(
                    "DashScope embedding 响应包含无效 text_index"
                ) from exc
            if index < 0 or index >= expected_count or ordered[index] is not None:
                raise EmbeddingResponseError(
                    "DashScope embedding 响应包含重复或越界的 text_index"
                )
            ordered[index] = item

        if any(item is None for item in ordered):
            raise EmbeddingResponseError(
                "DashScope embedding 响应缺少部分 text_index"
            )
        return ordered
