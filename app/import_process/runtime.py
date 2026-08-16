from dataclasses import dataclass

from app.embedding.interface import EmbeddingProvider
from app.vector_store.interface import VectorStore


@dataclass(frozen=True)
class ImportRuntime:
    embedding: EmbeddingProvider
    vector_store: VectorStore
