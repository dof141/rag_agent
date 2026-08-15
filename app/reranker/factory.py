import threading

from app.conf.reranker_config import RerankerConfig, reranker_config
from app.reranker.interface import RerankerConfigurationError, RerankerProvider


_provider_cache: dict[RerankerConfig, RerankerProvider] = {}
_provider_lock = threading.Lock()


def get_reranker_provider(
    config: RerankerConfig | None = None,
) -> RerankerProvider:
    selected_config = config or reranker_config
    with _provider_lock:
        cached = _provider_cache.get(selected_config)
        if cached is not None:
            return cached

        adapter = selected_config.adapter.strip().lower()
        if adapter == "http":
            from app.reranker.http_adapter import HttpRerankerProvider

            provider: RerankerProvider = HttpRerankerProvider(selected_config)
        elif adapter == "local":
            from app.reranker.local_adapter import LocalBgeRerankerProvider

            provider = LocalBgeRerankerProvider(selected_config)
        else:
            raise RerankerConfigurationError(
                f"不支持的 RERANKER_ADAPTER: {selected_config.adapter}"
            )

        _provider_cache[selected_config] = provider
        return provider


def clear_reranker_provider_cache() -> None:
    with _provider_lock:
        _provider_cache.clear()
