from app.embedding.factory import create_embedding_provider
from app.reranker import rerank_texts
from app.retrieval.interface import Retrieval
from app.retrieval.local_adapter import RetrievalModule
from app.retrieval.milvus_adapter import MilvusVectorSearch
from app.retrieval.qdrant_adapter import QdrantVectorSearch
from app.runtime_settings.service import RuntimeSettingsConfigurationError


def create_retrieval(
    snapshot,
    *,
    embedding_factory=create_embedding_provider,
    qdrant_factory=QdrantVectorSearch,
    milvus_factory=MilvusVectorSearch,
    reranker=rerank_texts,
) -> Retrieval:
    if not snapshot.user_id.strip():
        raise RuntimeSettingsConfigurationError("查询运行时缺少用户标识")

    combination = (snapshot.embedding_config.adapter, snapshot.vector_store_type)
    if combination == ("siliconflow", "qdrant") and snapshot.qdrant is not None:
        vector_config = snapshot.qdrant
        vector_factory = qdrant_factory
    elif combination == ("local_bge_m3", "milvus") and snapshot.milvus is not None:
        vector_config = snapshot.milvus
        vector_factory = milvus_factory
    else:
        raise RuntimeSettingsConfigurationError("查询运行配置缺失或组合不受支持")

    if (
        snapshot.embedding_config.dimension <= 0
        or vector_config.dimension <= 0
    ):
        raise RuntimeSettingsConfigurationError("查询运行配置的向量维度必须为正数")
    if snapshot.embedding_config.dimension != vector_config.dimension:
        raise RuntimeSettingsConfigurationError("查询运行配置的向量维度不一致")

    embedding = embedding_factory(snapshot.embedding_config)
    vector_search = vector_factory(vector_config, snapshot.user_id)
    return RetrievalModule(
        embedding,
        vector_search,
        reranker,
        expected_dimension=vector_config.dimension,
    )
