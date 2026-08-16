from collections.abc import Iterable

from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.vector_store.config import MilvusVectorStoreConfig
from app.vector_store.interface import (
    VectorDocument,
    VectorImportResult,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
)


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        config: MilvusVectorStoreConfig,
        *,
        client=None,
        data_type=None,
    ):
        _validate_config(config)
        self._config = config
        if client is None or data_type is None:
            from pymilvus import DataType, MilvusClient

            data_type = data_type or DataType
            client = client or MilvusClient(uri=config.url, token=config.token)
        self._client = client
        self._data_type = data_type

    def import_document(self, document: VectorDocument) -> VectorImportResult:
        try:
            document.validate(expected_dimension=self._config.dimension, require_sparse=True)
            self._ensure_item_collection()
            self._ensure_chunks_collection()
            self._client.delete(
                collection_name=self._config.item_collection,
                filter=self._item_filter(document),
            )
            self._client.delete(
                collection_name=self._config.chunks_collection,
                filter=self._chunk_filter(document),
            )
            self._client.insert(
                collection_name=self._config.item_collection,
                data=[self._item_row(document)],
            )
            chunk_rows = [self._chunk_row(document, chunk) for chunk in document.chunks]
            for batch in _batched(chunk_rows, self._config.batch_size):
                self._client.insert(
                    collection_name=self._config.chunks_collection,
                    data=batch,
                )
            self._client.flush(collection_name=self._config.item_collection)
            self._client.flush(collection_name=self._config.chunks_collection)
            self._client.load_collection(collection_name=self._config.item_collection)
            self._client.load_collection(collection_name=self._config.chunks_collection)
            return VectorImportResult(item_count=1, chunk_count=len(document.chunks))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Milvus 向量写入失败") from exc

    def rebuild_collections(self) -> None:
        self._client.delete_collection(self._config.item_collection)
        self._client.delete_collection(self._config.chunks_collection)
        self._ensure_item_collection()
        self._ensure_chunks_collection()

    def _ensure_item_collection(self) -> None:
        if self._client.has_collection(self._config.item_collection):
            return
        schema = self._base_schema(primary_field="item_id")
        self._client.create_collection(
            collection_name=self._config.item_collection,
            schema=schema,
            index_params=self._index_params(),
        )

    def _ensure_chunks_collection(self) -> None:
        if self._client.has_collection(self._config.chunks_collection):
            return
        schema = self._base_schema(primary_field="chunk_id")
        schema.add_field(field_name="title", datatype=self._data_type.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="parent_title",
            datatype=self._data_type.VARCHAR,
            max_length=65535,
        )
        schema.add_field(field_name="part", datatype=self._data_type.INT8)
        schema.add_field(field_name="chunk_index", datatype=self._data_type.INT64)
        self._client.create_collection(
            collection_name=self._config.chunks_collection,
            schema=schema,
            index_params=self._index_params(),
        )

    def _base_schema(self, *, primary_field: str):
        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name=primary_field, datatype=self._data_type.INT64, is_primary=True)
        schema.add_field(field_name="user_id", datatype=self._data_type.VARCHAR, max_length=128)
        schema.add_field(field_name="document_id", datatype=self._data_type.VARCHAR, max_length=128)
        schema.add_field(field_name="file_title", datatype=self._data_type.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=self._data_type.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=self._data_type.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="dense_vector",
            datatype=self._data_type.FLOAT_VECTOR,
            dim=self._config.dimension,
        )
        schema.add_field(field_name="sparse_vector", datatype=self._data_type.SPARSE_FLOAT_VECTOR)
        return schema

    def _index_params(self):
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 300},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        return index_params

    def _chunk_filter(self, document: VectorDocument) -> str:
        user_id = escape_milvus_string(document.user_id)
        document_id = escape_milvus_string(document.document_id)
        return f'user_id == "{user_id}" and document_id == "{document_id}"'

    def _item_filter(self, document: VectorDocument) -> str:
        user_id = escape_milvus_string(document.user_id)
        item_name = escape_milvus_string(document.item_name)
        return f'user_id == "{user_id}" and item_name == "{item_name}"'

    def _item_row(self, document: VectorDocument) -> dict:
        return {
            "user_id": document.user_id,
            "document_id": document.document_id,
            "file_title": document.file_title,
            "item_name": document.item_name,
            "content": document.item_name,
            "dense_vector": list(document.item_dense_vector),
            "sparse_vector": document.item_sparse_vector,
        }

    def _chunk_row(self, document: VectorDocument, chunk) -> dict:
        return {
            "user_id": document.user_id,
            "document_id": document.document_id,
            "file_title": document.file_title,
            "item_name": document.item_name,
            "content": chunk.content,
            "title": chunk.title,
            "parent_title": chunk.parent_title,
            "part": chunk.part,
            "chunk_index": chunk.index,
            "dense_vector": list(chunk.dense_vector),
            "sparse_vector": chunk.sparse_vector,
        }


def _validate_config(config: MilvusVectorStoreConfig) -> None:
    if not config.url:
        raise VectorStoreConfigurationError("Milvus URL 缺失")
    if not config.item_collection or not config.chunks_collection:
        raise VectorStoreConfigurationError("Milvus collection 配置缺失")
    if config.dimension <= 0 or config.batch_size <= 0:
        raise VectorStoreConfigurationError("Milvus 数值配置无效")


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
