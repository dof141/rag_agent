from functools import lru_cache

from app.conf.embedding_config import embedding_config
from app.embedding.factory import get_embedding_provider
from app.embedding.interface import EmbeddingResult, validate_texts


def generate_embeddings(texts: list[str]) -> EmbeddingResult:
    """Compatibility facade for callers that need hybrid embeddings."""
    validate_texts(texts)
    return get_embedding_provider().embed_documents(texts)


@lru_cache(maxsize=1)
def get_bge_m3_ef():
    """Return the legacy local BGE-M3 model without importing it eagerly."""
    from app.embedding.local_adapter import LocalBgeM3EmbeddingProvider

    provider = LocalBgeM3EmbeddingProvider(embedding_config)
    return provider.get_model()
