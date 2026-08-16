from app.vector_store.interface import VectorStoreConfigurationError
from app.vector_store.milvus_adapter import MilvusVectorStore
from app.vector_store.qdrant_adapter import QdrantVectorStore


def create_vector_store(
    snapshot,
    *,
    qdrant_factory=QdrantVectorStore,
    milvus_factory=MilvusVectorStore,
):
    if snapshot.vector_store_type == "qdrant" and snapshot.qdrant is not None:
        return qdrant_factory(snapshot.qdrant)
    if snapshot.vector_store_type == "milvus" and snapshot.milvus is not None:
        return milvus_factory(snapshot.milvus)
    raise VectorStoreConfigurationError("向量库配置缺失或不受支持")
