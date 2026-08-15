from app.conf.reranker_config import RerankerConfig, reranker_config
from app.core.logger import logger
from app.reranker.factory import get_reranker_provider
from app.reranker.interface import RerankItem, RerankOutcome, RerankerProvider


RERANKER_DEGRADED_CODE = "reranker_degraded"
RERANKER_DEGRADED_MESSAGE = "重排序服务暂时不可用，本次回答已使用原始检索顺序生成"
_FALLBACK_LIMIT = 10


def rerank_texts(
    query: str,
    documents: list[str],
    *,
    config: RerankerConfig | None = None,
    provider: RerankerProvider | None = None,
) -> RerankOutcome:
    selected_config = config or reranker_config
    selected_documents = list(documents[: selected_config.max_documents])
    if not selected_documents:
        return RerankOutcome(items=[])

    try:
        selected_provider = provider or get_reranker_provider(selected_config)
        items = selected_provider.rerank(query, selected_documents)
        return RerankOutcome(items=items)
    except Exception as exc:
        logger.warning(f"重排序失败，降级为原始检索顺序: {exc}")
        return RerankOutcome(
            items=[
                RerankItem(index=index, score=None)
                for index in range(min(len(selected_documents), _FALLBACK_LIMIT))
            ],
            degraded=True,
            warning_code=RERANKER_DEGRADED_CODE,
            warning_message=RERANKER_DEGRADED_MESSAGE,
        )
