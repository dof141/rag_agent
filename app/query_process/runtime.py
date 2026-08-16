from dataclasses import dataclass

from app.embedding.factory import get_embedding_provider
from app.reranker import rerank_texts
from app.retrieval.factory import create_retrieval
from app.retrieval.interface import Retrieval
from app.retrieval.milvus_adapter import MilvusVectorSearch
from app.retrieval.qdrant_adapter import QdrantVectorSearch


@dataclass(frozen=True)
class QueryRuntime:
    user_id: str
    settings_version: int
    retrieval: Retrieval


def create_query_runtime(
    snapshot,
    *,
    embedding_factory=get_embedding_provider,
    qdrant_factory=QdrantVectorSearch,
    milvus_factory=MilvusVectorSearch,
    reranker=rerank_texts,
) -> QueryRuntime:
    retrieval = create_retrieval(
        snapshot,
        embedding_factory=embedding_factory,
        qdrant_factory=qdrant_factory,
        milvus_factory=milvus_factory,
        reranker=reranker,
    )
    return QueryRuntime(
        user_id=snapshot.user_id,
        settings_version=snapshot.version,
        retrieval=retrieval,
    )
