from collections.abc import Iterable

from app.vector_store.config import QdrantVectorStoreConfig
from app.vector_store.document_id import build_point_id
from app.vector_store.interface import (
    VectorDocument,
    VectorImportResult,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
)


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        config: QdrantVectorStoreConfig,
        *,
        client=None,
        models_module=None,
    ):
        _validate_config(config)
        self._config = config
        if client is None or models_module is None:
            from qdrant_client import QdrantClient, models

            models_module = models if models_module is None else models_module
            client = client or QdrantClient(
                url=config.url,
                api_key=config.api_key,
                prefer_grpc=False,
                cloud_inference=config.cloud_inference,
            )
        self._client = client
        self._models = models_module

    def import_document(self, document: VectorDocument) -> VectorImportResult:
        try:
            document.validate(expected_dimension=self._config.dimension, require_sparse=False)
            self._ensure_collection(self._config.item_collection)
            self._ensure_collection(self._config.chunks_collection)
            document_filter = self._document_filter(document)
            self._client.delete(
                collection_name=self._config.item_collection,
                points_selector=document_filter,
                wait=True,
            )
            self._client.delete(
                collection_name=self._config.chunks_collection,
                points_selector=document_filter,
                wait=True,
            )
            self._client.upsert(
                collection_name=self._config.item_collection,
                points=[self._item_point(document)],
                wait=True,
            )
            chunk_points = [self._chunk_point(document, chunk) for chunk in document.chunks]
            for batch in _batched(chunk_points, self._config.batch_size):
                self._client.upsert(
                    collection_name=self._config.chunks_collection,
                    points=batch,
                    wait=True,
                )
            return VectorImportResult(item_count=1, chunk_count=len(document.chunks))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant 向量写入失败") from exc

    def _ensure_collection(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name):
            return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": self._models.VectorParams(
                    size=self._config.dimension,
                    distance=self._models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": self._models.SparseVectorParams(
                    modifier=self._models.Modifier.IDF,
                )
            },
        )

    def _document_filter(self, document: VectorDocument):
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="user_id",
                    match=self._models.MatchValue(value=document.user_id),
                ),
                self._models.FieldCondition(
                    key="document_id",
                    match=self._models.MatchValue(value=document.document_id),
                ),
            ]
        )

    def _item_point(self, document: VectorDocument):
        return self._point(
            document,
            role="item",
            index=0,
            text=document.item_name,
            dense=document.item_dense_vector,
            payload={
                "user_id": document.user_id,
                "document_id": document.document_id,
                "file_title": document.file_title,
                "item_name": document.item_name,
                "content": document.item_name,
            },
        )

    def _chunk_point(self, document: VectorDocument, chunk):
        return self._point(
            document,
            role="chunk",
            index=chunk.index,
            text=chunk.content,
            dense=chunk.dense_vector,
            payload={
                "user_id": document.user_id,
                "document_id": document.document_id,
                "file_title": document.file_title,
                "item_name": document.item_name,
                "content": chunk.content,
                "title": chunk.title,
                "parent_title": chunk.parent_title,
                "part": chunk.part,
                "chunk_index": chunk.index,
            },
        )

    def _point(
        self,
        document: VectorDocument,
        *,
        role: str,
        index: int,
        text: str,
        dense: tuple[float, ...],
        payload: dict,
    ):
        return self._models.PointStruct(
            id=build_point_id(document.user_id, document.document_id, role, index),
            vector={
                "dense": list(dense),
                "bm25": self._models.Document(text=text, model=self._config.bm25_model),
            },
            payload=payload,
        )


def _validate_config(config: QdrantVectorStoreConfig) -> None:
    if not config.url or not config.api_key:
        raise VectorStoreConfigurationError("Qdrant URL 或 API Key 缺失")
    if not config.item_collection or not config.chunks_collection:
        raise VectorStoreConfigurationError("Qdrant collection 配置缺失")
    if config.dimension <= 0 or config.batch_size <= 0:
        raise VectorStoreConfigurationError("Qdrant 数值配置无效")
    if not config.cloud_inference:
        raise VectorStoreConfigurationError("Qdrant BM25 导入要求启用 Cloud Inference")


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
