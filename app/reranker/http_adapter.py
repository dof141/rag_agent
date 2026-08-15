import math
from typing import Any

from app.conf.reranker_config import RerankerConfig
from app.reranker.interface import (
    RerankItem,
    RerankerConfigurationError,
    RerankerRequestError,
    RerankerResponseError,
)


class HttpRerankerProvider:
    def __init__(self, config: RerankerConfig, http_client: Any = None):
        if not config.api_key:
            raise RerankerConfigurationError(
                "远程重排序需要配置 RERANKER_API_KEY 或 SILICONFLOW_API_KEY"
            )
        self._config = config
        if http_client is None:
            import requests

            http_client = requests
        self._http_client = http_client

    def rerank(self, query: str, documents: list[str]) -> list[RerankItem]:
        if not query or not query.strip():
            raise RerankerRequestError("重排序 query 不能为空")
        if not documents:
            return []

        try:
            response = self._http_client.post(
                f"{self._config.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._config.model,
                    "query": query,
                    "documents": documents,
                    "return_documents": False,
                    "top_n": len(documents),
                },
                timeout=self._config.request_timeout,
            )
        except Exception as exc:
            raise RerankerRequestError(f"远程重排序请求失败: {exc}") from exc

        if not 200 <= response.status_code < 300:
            detail = (getattr(response, "text", "") or "")[:500]
            raise RerankerRequestError(
                f"远程重排序请求返回 HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise RerankerResponseError("远程重排序响应不是有效 JSON") from exc

        return self._parse_results(payload, len(documents))

    @staticmethod
    def _parse_results(payload: Any, document_count: int) -> list[RerankItem]:
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RerankerResponseError("远程重排序响应缺少 results 列表")

        results = payload["results"]
        if len(results) != document_count:
            raise RerankerResponseError(
                f"远程重排序结果数量不一致: expected={document_count}, actual={len(results)}"
            )

        seen_indexes: set[int] = set()
        items: list[RerankItem] = []
        for result in results:
            if not isinstance(result, dict):
                raise RerankerResponseError("远程重排序 result 必须是对象")

            index = result.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RerankerResponseError("远程重排序 result.index 必须是整数")
            if index < 0 or index >= document_count:
                raise RerankerResponseError(f"远程重排序 result.index 越界: {index}")
            if index in seen_indexes:
                raise RerankerResponseError(f"远程重排序 result.index 重复: {index}")

            raw_score = result.get("relevance_score")
            if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                raise RerankerResponseError(
                    "远程重排序 result 缺少有效 relevance_score"
                )
            score = float(raw_score)
            if not math.isfinite(score):
                raise RerankerResponseError("远程重排序 relevance_score 必须是有限数值")

            seen_indexes.add(index)
            items.append(RerankItem(index=index, score=score))

        if seen_indexes != set(range(document_count)):
            raise RerankerResponseError("远程重排序结果未完整覆盖输入索引")

        items.sort(key=lambda item: item.score if item.score is not None else float("-inf"), reverse=True)
        return items
