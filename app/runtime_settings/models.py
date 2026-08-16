from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.conf.embedding_config import EmbeddingConfig
from app.vector_store.config import MilvusVectorStoreConfig, QdrantVectorStoreConfig


class SecretStatus(BaseModel):
    configured: bool
    masked: str | None = None


class RuntimeSettingsUpdate(BaseModel):
    embedding_provider: Literal["siliconflow", "local_bge_m3"]
    embedding_base_url: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    embedding_batch_size: int = Field(gt=0, le=128)
    embedding_timeout: float = Field(gt=0, le=300)
    embedding_api_key: str | None = None
    vector_store_type: Literal["qdrant", "milvus"]
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_item_collection: str | None = "rag_item_names_v1"
    qdrant_chunks_collection: str | None = "rag_chunks_v1"
    qdrant_cloud_inference: bool = True
    milvus_url: str | None = None
    milvus_token: str | None = None
    milvus_item_collection: str | None = None
    milvus_chunks_collection: str | None = None


class RuntimeSettingsResponse(BaseModel):
    embedding_provider: Literal["siliconflow", "local_bge_m3"]
    embedding_base_url: str
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int
    embedding_timeout: float
    embedding_api_key: SecretStatus
    vector_store_type: Literal["qdrant", "milvus"]
    qdrant_url: str | None
    qdrant_api_key: SecretStatus
    qdrant_item_collection: str | None
    qdrant_chunks_collection: str | None
    qdrant_cloud_inference: bool
    milvus_url: str | None
    milvus_token: SecretStatus
    milvus_item_collection: str | None
    milvus_chunks_collection: str | None
    version: int
    updated_at: str


@dataclass(frozen=True)
class UserRuntimeSnapshot:
    user_id: str
    version: int
    embedding_config: EmbeddingConfig
    vector_store_type: Literal["qdrant", "milvus"]
    qdrant: QdrantVectorStoreConfig | None
    milvus: MilvusVectorStoreConfig | None
