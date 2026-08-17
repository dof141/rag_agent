from dataclasses import dataclass, field


@dataclass(frozen=True)
class QdrantVectorStoreConfig:
    url: str
    api_key: str = field(repr=False)
    item_collection: str = "rag_item_names_v1"
    chunks_collection: str = "rag_chunks_v1"
    dimension: int = 1024
    cloud_inference: bool = True
    bm25_model: str = "Qdrant/bm25"
    batch_size: int = 16
    request_timeout: float = 30.0


@dataclass(frozen=True)
class MilvusVectorStoreConfig:
    url: str
    token: str | None = field(default=None, repr=False)
    item_collection: str = "item_names"
    chunks_collection: str = "rag_chunks"
    dimension: int = 1024
    batch_size: int = 64
