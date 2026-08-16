import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet
from fastapi import FastAPI

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from app.auth.dependencies import build_current_user_dependency
from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService, PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase
from app.runtime_settings.crypto import SecretCipher
from app.runtime_settings.repository import RuntimeSettingsRepository
from app.runtime_settings.router import create_settings_router
from app.runtime_settings.service import RuntimeSettingsService


class RuntimeSettingsHttpTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        self.database.initialize()
        self.users = UserRepository(self.database, PasswordHasher())
        self.user_a = self.users.ensure_admin("admin", "password")
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("user-b", "admin-b", "hash", "admin", "2026-08-16T00:00:00+00:00"),
            )
        self.user_b = self.users.get_by_id("user-b")
        self.tokens = JwtTokenService(secret="unit-test-signing-secret", ttl_seconds=60)
        self.settings = RuntimeSettingsService(
            RuntimeSettingsRepository(self.database),
            SecretCipher(Fernet.generate_key().decode("ascii")),
        )
        current_user = build_current_user_dependency(self.users, self.tokens)
        self.app = FastAPI()
        self.app.include_router(create_settings_router(self.settings, current_user))
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def auth(self, user_id: str):
        return {"Authorization": f"Bearer {self.tokens.issue(user_id)}"}

    def payload(self, **overrides):
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
        return values

    def test_requires_auth_and_masks_saved_settings(self):
        self.assertEqual(self.client.get("/api/settings/runtime").status_code, 401)

        saved = self.client.put(
            "/api/settings/runtime",
            json=self.payload(),
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(saved.status_code, 200)
        self.assertNotIn("sf-secret", saved.text)
        self.assertTrue(saved.json()["embedding_api_key"]["configured"])

        loaded = self.client.get(
            "/api/settings/runtime",
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(loaded.status_code, 200)
        self.assertNotIn("qd-secret", loaded.text)

    def test_clear_secret_and_user_isolation(self):
        self.client.put(
            "/api/settings/runtime",
            json=self.payload(),
            headers=self.auth(self.user_a.id),
        )
        other_user = self.client.get(
            "/api/settings/runtime",
            headers=self.auth(self.user_b.id),
        )
        self.assertEqual(other_user.status_code, 200)
        self.assertIsNone(other_user.json())

        cleared = self.client.delete(
            "/api/settings/runtime/secrets/embedding_api_key",
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(cleared.status_code, 200)
        self.assertFalse(cleared.json()["embedding_api_key"]["configured"])


if __name__ == "__main__":
    unittest.main()
