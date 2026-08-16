import threading

from app.conf.embedding_config import EmbeddingConfig, embedding_config
from app.embedding.interface import EmbeddingConfigurationError, EmbeddingProvider


_provider_cache: dict[EmbeddingConfig, EmbeddingProvider] = {}
_provider_lock = threading.Lock()


def get_embedding_provider(
    config: EmbeddingConfig | None = None,
) -> EmbeddingProvider:
    selected_config = config or embedding_config
    with _provider_lock:
        cached = _provider_cache.get(selected_config)
        if cached is not None:
            return cached

        provider = create_embedding_provider(selected_config)

        _provider_cache[selected_config] = provider
        return provider


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    adapter = config.adapter.strip().lower()
    if adapter == "dashscope":
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider

        return DashScopeEmbeddingProvider(config)
    if adapter in {"local", "local_bge_m3"}:
        from app.embedding.local_adapter import LocalBgeM3EmbeddingProvider

        return LocalBgeM3EmbeddingProvider(config)
    if adapter == "siliconflow":
        from app.embedding.siliconflow_adapter import SiliconFlowEmbeddingAdapter

        return SiliconFlowEmbeddingAdapter(config)
    raise EmbeddingConfigurationError(f"不支持的 EMBEDDING_ADAPTER: {config.adapter}")


def clear_embedding_provider_cache() -> None:
    with _provider_lock:
        _provider_cache.clear()
