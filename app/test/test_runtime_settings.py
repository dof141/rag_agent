import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet

from app.auth.repository import UserRepository
from app.auth.security import PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase
from app.runtime_settings.crypto import SecretCipher
from app.runtime_settings.models import RuntimeSettingsUpdate
from app.runtime_settings.repository import RuntimeSettingsRepository
from app.runtime_settings.service import (
    RuntimeSettingsConfigurationError,
    RuntimeSettingsService,
)


class RuntimeSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        self.database.initialize()
        users = UserRepository(self.database, PasswordHasher())
        self.user_a = users.ensure_admin("admin", "password")
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("user-b", "admin-b", "hash", "admin", "2026-08-16T00:00:00+00:00"),
            )
        self.user_b = users.get_by_id("user-b")
        self.service = RuntimeSettingsService(
            RuntimeSettingsRepository(self.database),
            SecretCipher(Fernet.generate_key().decode("ascii")),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def siliconflow_qdrant_payload(self, **overrides):
        values = {
            "embedding_provider": "siliconflow",
            "embedding_base_url": "https://api.siliconflow.cn/v1",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": 1024,
            "embedding_batch_size": 8,
            "embedding_timeout": 30.0,
            "embedding_api_key": "sf-secret",
            "vector_store_type": "qdrant",
            "qdrant_url": "https://example.qdrant.io",
            "qdrant_api_key": "qd-secret",
            "qdrant_item_collection": "items",
            "qdrant_chunks_collection": "chunks",
            "qdrant_cloud_inference": True,
            "milvus_url": None,
            "milvus_token": None,
            "milvus_item_collection": None,
            "milvus_chunks_collection": None,
        }
        values.update(overrides)
        return RuntimeSettingsUpdate(**values)

    def test_secrets_are_encrypted_masked_and_isolated(self):
        self.service.save(self.user_a.id, self.siliconflow_qdrant_payload())

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT embedding_api_key_encrypted, qdrant_api_key_encrypted
                FROM runtime_settings
                WHERE user_id = ?
                """,
                (self.user_a.id,),
            ).fetchone()
        self.assertNotEqual(row["embedding_api_key_encrypted"], "sf-secret")
        self.assertNotEqual(row["qdrant_api_key_encrypted"], "qd-secret")

        response = self.service.get_public(self.user_a.id)
        self.assertTrue(response.embedding_api_key.configured)
        self.assertTrue(response.qdrant_api_key.configured)
        self.assertNotIn("sf-secret", response.model_dump_json())
        self.assertNotIn("qd-secret", response.model_dump_json())
        self.assertIsNone(self.service.get_public(self.user_b.id))

    def test_empty_secret_keeps_value_and_explicit_clear_removes_it(self):
        self.service.save(self.user_a.id, self.siliconflow_qdrant_payload())
        update = self.siliconflow_qdrant_payload(embedding_api_key="", qdrant_api_key="")
        self.service.save(self.user_a.id, update)
        snapshot = self.service.get_snapshot(self.user_a.id)
        self.assertEqual(snapshot.embedding_config.api_key, "sf-secret")
        self.assertEqual(snapshot.qdrant.api_key, "qd-secret")

        self.service.clear_secret(self.user_a.id, "embedding_api_key")
        with self.assertRaises(RuntimeSettingsConfigurationError):
            self.service.get_snapshot(self.user_a.id)


if __name__ == "__main__":
    unittest.main()
