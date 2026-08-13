import threading
from typing import Any

from app.conf.embedding_config import EmbeddingConfig
from app.core.logger import logger
from app.embedding.interface import EmbeddingProvider, EmbeddingResult, validate_texts


class LocalBgeM3EmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, model_factory: Any = None):
        self._config = config
        self._model_factory = model_factory
        self._model: Any = None
        self._model_lock = threading.Lock()

    def get_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_factory = self._model_factory
            if model_factory is None:
                from pymilvus.model.hybrid import BGEM3EmbeddingFunction

                model_factory = BGEM3EmbeddingFunction

            model_name = (
                self._config.bge_m3_path
                or self._config.bge_m3
                or "BAAI/bge-m3"
            )
            logger.info(f"开始初始化本地 BGE-M3 模型: {model_name}")
            self._model = model_factory(
                model_name=model_name,
                device=self._config.bge_device or "cpu",
                use_fp16=self._config.bge_fp16,
                normalize_embeddings=True,
            )
            return self._model

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        validate_texts(texts)
        embeddings = self.get_model().encode_documents(texts)
        sparse_matrix = embeddings["sparse"]
        sparse_vectors: list[dict[int, float]] = []

        for index in range(len(texts)):
            start = sparse_matrix.indptr[index]
            end = sparse_matrix.indptr[index + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            values = sparse_matrix.data[start:end].tolist()
            sparse_vectors.append(
                {
                    int(sparse_index): float(value)
                    for sparse_index, value in zip(indices, values)
                }
            )

        dense_vectors = [
            [float(value) for value in vector.tolist()]
            for vector in embeddings["dense"]
        ]
        return {"dense": dense_vectors, "sparse": sparse_vectors}
