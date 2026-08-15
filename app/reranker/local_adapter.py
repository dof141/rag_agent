import math
import threading
from typing import Any

from app.conf.reranker_config import RerankerConfig
from app.reranker.interface import RerankItem, RerankerRequestError, RerankerResponseError


class LocalBgeRerankerProvider:
    _BATCH_SIZE = 4

    def __init__(self, config: RerankerConfig, model_factory: Any = None):
        self._config = config
        self._model_factory = model_factory
        self._model: Any = None
        self._model_lock = threading.Lock()

    def get_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_factory = self._model_factory
            if model_factory is None:
                from FlagEmbedding import FlagReranker

                model_factory = FlagReranker

            self._model = model_factory(
                model_name_or_path=(
                    self._config.bge_reranker_large
                    or self._config.model
                    or "BAAI/bge-reranker-v2-m3"
                ),
                device=self._config.bge_reranker_device,
                use_fp16=self._config.bge_reranker_fp16,
            )
            return self._model

    def rerank(self, query: str, documents: list[str]) -> list[RerankItem]:
        if not query or not query.strip():
            raise RerankerRequestError("重排序 query 不能为空")
        if not documents:
            return []

        pairs = [[query, document] for document in documents]
        scores: list[float] = []
        try:
            model = self.get_model()
            for start in range(0, len(pairs), self._BATCH_SIZE):
                batch_scores = model.compute_score(
                    pairs[start : start + self._BATCH_SIZE],
                    normalize=True,
                )
                if isinstance(batch_scores, (int, float)):
                    batch_scores = [batch_scores]
                scores.extend(float(score) for score in batch_scores)
        except Exception as exc:
            raise RerankerRequestError(f"本地重排序计算失败: {exc}") from exc

        if len(scores) != len(documents):
            raise RerankerResponseError(
                f"本地重排序结果数量不一致: expected={len(documents)}, actual={len(scores)}"
            )
        if any(not math.isfinite(score) for score in scores):
            raise RerankerResponseError("本地重排序分数必须是有限数值")

        items = [
            RerankItem(index=index, score=score)
            for index, score in enumerate(scores)
        ]
        items.sort(key=lambda item: item.score if item.score is not None else float("-inf"), reverse=True)
        return items
