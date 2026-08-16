"""手工验证 SiliconFlow Embedding 写入 Qdrant Cloud。"""

import os
import time

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


class QdrantHybridProbe:
    """调用远程向量模型，并将测试向量写入远程 Qdrant。"""

    TEST_DOCUMENTS = (
        {
            "id": 1,
            "title": "根的判别式",
            "content": "一元二次方程可通过判别式判断实数根的情况。",
        },
        {
            "id": 2,
            "title": "根与系数的关系",
            "content": "一元二次方程两根之和等于负的二次项系数分之一项系数。",
        },
        {
            "id": 3,
            "title": "勾股定理",
            "content": "直角三角形两条直角边平方和等于斜边平方。",
        },
    )

    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "").strip()
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        self.siliconflow_api_key = (
            os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("RERANKER_API_KEY", "")
        ).strip()
        self.siliconflow_base_url = os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        self.embedding_model = os.getenv(
            "SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.embedding_dimension = int(
            os.getenv("SILICONFLOW_EMBEDDING_DIMENSION", "1024")
        )
        self.embedding_timeout = float(
            os.getenv("SILICONFLOW_EMBEDDING_TIMEOUT", "20")
        )
        self.bm25_model = os.getenv("QDRANT_BM25_MODEL", "Qdrant/bm25")
        self.collection_name = os.getenv(
            "QDRANT_PROBE_COLLECTION", "rag_remote_vector_probe"
        )

    def run(self) -> None:
        """生成 dense 向量，写入 Qdrant，并输出实际写入数量。"""
        self._validate_config()
        texts = [document["content"] for document in self.TEST_DOCUMENTS]

        started_at = time.perf_counter()
        dense_vectors = self._request_embeddings(texts)
        logger.info(
            "远程 Embedding 调用成功：model={}，数量={}，维度={}，耗时={:.2f}s",
            self.embedding_model,
            len(dense_vectors),
            self.embedding_dimension,
            time.perf_counter() - started_at,
        )

        client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            cloud_inference=True,
            timeout=30,
        )
        try:
            self._recreate_collection(client)
            self._upsert_documents(client, dense_vectors)
            count = client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
            logger.info(
                "Qdrant 写入成功：collection={}，point 数={}",
                self.collection_name,
                count,
            )
        finally:
            client.close()

    def _validate_config(self) -> None:
        missing = [
            name
            for name, value in (
                ("QDRANT_URL", self.qdrant_url),
                ("QDRANT_API_KEY", self.qdrant_api_key),
                ("SILICONFLOW_API_KEY", self.siliconflow_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"请先在 .env 中配置：{', '.join(missing)}")

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.siliconflow_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.siliconflow_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.embedding_model,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=self.embedding_timeout,
        )
        response.raise_for_status()

        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        vectors = [row["embedding"] for row in rows]
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding 返回数量不符：期望 {len(texts)}，实际 {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    "Embedding 维度不符："
                    f"期望 {self.embedding_dimension}，实际 {len(vector)}"
                )
        return vectors

    def _recreate_collection(self, client: QdrantClient) -> None:
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25_sparse_vector": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        logger.info("Qdrant collection 已创建：{}", self.collection_name)

    def _upsert_documents(
        self,
        client: QdrantClient,
        dense_vectors: list[list[float]],
    ) -> None:
        points = []
        for document, dense_vector in zip(
            self.TEST_DOCUMENTS, dense_vectors, strict=True
        ):
            points.append(
                models.PointStruct(
                    id=document["id"],
                    vector={
                        "dense_vector": dense_vector,
                        "bm25_sparse_vector": models.Document(
                            text=document["content"],
                            model=self.bm25_model,
                        ),
                    },
                    payload={
                        "title": document["title"],
                        "content": document["content"],
                    },
                )
            )

        client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )


if __name__ == "__main__":
    QdrantHybridProbe().run()
