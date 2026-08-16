from abc import ABC, abstractmethod
from dataclasses import dataclass


SparseVector = dict[int, float]


class VectorStoreError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "vector_store"):
        super().__init__(message)
        self.stage = stage


class VectorStoreConfigurationError(VectorStoreError):
    pass


@dataclass(frozen=True)
class VectorChunk:
    index: int
    content: str
    title: str
    parent_title: str
    part: int
    dense_vector: tuple[float, ...]
    sparse_vector: SparseVector | None


@dataclass(frozen=True)
class VectorDocument:
    user_id: str
    document_id: str
    file_title: str
    item_name: str
    item_dense_vector: tuple[float, ...]
    item_sparse_vector: SparseVector | None
    chunks: tuple[VectorChunk, ...]

    def validate(self, *, expected_dimension: int, require_sparse: bool) -> None:
        if not self.user_id or not self.document_id or not self.item_name or not self.chunks:
            raise ValueError("向量文档缺少必填字段")
        vectors = [self.item_dense_vector, *(chunk.dense_vector for chunk in self.chunks)]
        if any(len(vector) != expected_dimension for vector in vectors):
            raise ValueError("dense 向量维度与配置不一致")
        sparse = [self.item_sparse_vector, *(chunk.sparse_vector for chunk in self.chunks)]
        if require_sparse and any(vector is None for vector in sparse):
            raise ValueError("Milvus 导入要求 sparse 向量")
        if len({chunk.index for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("chunk index 必须唯一")


@dataclass(frozen=True)
class VectorImportResult:
    item_count: int
    chunk_count: int


class VectorStore(ABC):
    @abstractmethod
    def import_document(self, document: VectorDocument) -> VectorImportResult:
        """幂等替换一个用户的一份文档。"""
