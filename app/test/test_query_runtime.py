import unittest
from dataclasses import FrozenInstanceError, fields

from app.conf.embedding_config import EmbeddingConfig
from app.runtime_settings.models import UserRuntimeSnapshot
from app.runtime_settings.service import RuntimeSettingsConfigurationError
from app.vector_store.config import MilvusVectorStoreConfig, QdrantVectorStoreConfig


class QueryRuntimeFactoryTest(unittest.TestCase):
    def embedding_config(self, adapter="siliconflow", output_type="dense"):
        return EmbeddingConfig(
            adapter=adapter,
            model="embedding-model",
            dimension=3,
            output_type=output_type,
            batch_size=4,
            request_timeout=10,
            base_url="https://embedding.example",
            api_key="embedding-secret",
            bge_m3_path=None,
            bge_m3=None,
            bge_device="cpu",
            bge_fp16=False,
        )

    def qdrant_config(self):
        return QdrantVectorStoreConfig(
            url="https://qdrant.example",
            api_key="qdrant-secret",
            item_collection="items",
            chunks_collection="chunks",
            dimension=3,
        )

    def test_creates_qdrant_runtime_from_user_snapshot(self):
        from app.query_process.runtime import create_query_runtime

        embedding_config = self.embedding_config()
        qdrant_config = self.qdrant_config()
        snapshot = UserRuntimeSnapshot(
            user_id="user-a",
            version=7,
            embedding_config=embedding_config,
            vector_store_type="qdrant",
            qdrant=qdrant_config,
            milvus=None,
        )
        embedding = object()
        vector_search = object()
        calls = []

        def embedding_factory(config):
            calls.append(("embedding", config))
            return embedding

        def qdrant_factory(config, user_id):
            calls.append(("qdrant", config, user_id))
            return vector_search

        runtime = create_query_runtime(
            snapshot,
            embedding_factory=embedding_factory,
            qdrant_factory=qdrant_factory,
            reranker=object(),
        )

        self.assertEqual(runtime.user_id, "user-a")
        self.assertEqual(runtime.settings_version, 7)
        self.assertIs(runtime.retrieval.vector_search, vector_search)
        self.assertEqual(
            calls,
            [
                ("embedding", embedding_config),
                ("qdrant", qdrant_config, "user-a"),
            ],
        )

    def test_creates_milvus_runtime_from_user_snapshot(self):
        from app.query_process.runtime import create_query_runtime

        embedding_config = self.embedding_config(
            adapter="local_bge_m3",
            output_type="dense&sparse",
        )
        milvus_config = MilvusVectorStoreConfig(
            url="https://milvus.example",
            token="milvus-secret",
            item_collection="items",
            chunks_collection="chunks",
            dimension=3,
        )
        snapshot = UserRuntimeSnapshot(
            user_id="user-b",
            version=9,
            embedding_config=embedding_config,
            vector_store_type="milvus",
            qdrant=None,
            milvus=milvus_config,
        )
        vector_search = object()
        calls = []

        runtime = create_query_runtime(
            snapshot,
            embedding_factory=lambda config: calls.append(("embedding", config)) or object(),
            qdrant_factory=lambda *_args: self.fail("qdrant factory must not be called"),
            milvus_factory=lambda config, user_id: calls.append(
                ("milvus", config, user_id)
            )
            or vector_search,
            reranker=object(),
        )

        self.assertEqual(runtime.user_id, "user-b")
        self.assertEqual(runtime.settings_version, 9)
        self.assertIs(runtime.retrieval.vector_search, vector_search)
        self.assertEqual(
            calls,
            [
                ("embedding", embedding_config),
                ("milvus", milvus_config, "user-b"),
            ],
        )

    def test_query_runtime_is_frozen_and_has_only_public_runtime_fields(self):
        from app.query_process.runtime import QueryRuntime

        runtime = QueryRuntime(user_id="user-a", settings_version=3, retrieval=object())

        self.assertEqual(
            [field.name for field in fields(QueryRuntime)],
            ["user_id", "settings_version", "retrieval"],
        )
        with self.assertRaises(FrozenInstanceError):
            runtime.user_id = "other-user"

    def test_rejects_invalid_snapshots_without_exposing_secrets(self):
        from app.query_process.runtime import create_query_runtime

        invalid_snapshots = [
            UserRuntimeSnapshot(
                user_id="",
                version=1,
                embedding_config=self.embedding_config(),
                vector_store_type="qdrant",
                qdrant=self.qdrant_config(),
                milvus=None,
            ),
            UserRuntimeSnapshot(
                user_id="   ",
                version=1,
                embedding_config=self.embedding_config(),
                vector_store_type="qdrant",
                qdrant=self.qdrant_config(),
                milvus=None,
            ),
            UserRuntimeSnapshot(
                user_id="user-a",
                version=1,
                embedding_config=self.embedding_config(adapter="local_bge_m3"),
                vector_store_type="qdrant",
                qdrant=self.qdrant_config(),
                milvus=None,
            ),
            UserRuntimeSnapshot(
                user_id="user-a",
                version=1,
                embedding_config=self.embedding_config(),
                vector_store_type="qdrant",
                qdrant=None,
                milvus=None,
            ),
        ]

        for snapshot in invalid_snapshots:
            with self.subTest(snapshot=snapshot.vector_store_type):
                with self.assertRaises(RuntimeSettingsConfigurationError) as raised:
                    create_query_runtime(
                        snapshot,
                        embedding_factory=lambda _config: self.fail(
                            "embedding factory must not be called"
                        ),
                        qdrant_factory=lambda *_args: self.fail(
                            "qdrant factory must not be called"
                        ),
                        milvus_factory=lambda *_args: self.fail(
                            "milvus factory must not be called"
                        ),
                    )

                message = str(raised.exception)
                self.assertNotIn("embedding-secret", message)
                self.assertNotIn("qdrant-secret", message)


if __name__ == "__main__":
    unittest.main()
