import time
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, FastAPI

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from app.auth.dependencies import build_current_user_dependency
from app.auth.repository import UserRepository
from app.auth.router import create_auth_router
from app.auth.security import JwtTokenService, PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase


class AuthHttpTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        self.database.initialize()
        self.users = UserRepository(self.database, PasswordHasher())
        self.admin = self.users.ensure_admin("admin", "password")
        self.tokens = JwtTokenService(secret="unit-test-signing-secret", ttl_seconds=60)
        self.app = FastAPI()
        self.app.include_router(create_auth_router(self.users, self.tokens))
        current_user = build_current_user_dependency(self.users, self.tokens)

        @self.app.get("/protected")
        async def protected(user=Depends(current_user)):
            return {"user_id": user.id}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_login_and_current_user(self):
        app = FastAPI()
        app.include_router(create_auth_router(self.users, self.tokens))
        current_user = build_current_user_dependency(self.users, self.tokens)

        @app.get("/protected")
        async def protected(user=Depends(current_user)):
            return {"user_id": user.id}

        client = TestClient(app)
        login = client.post(
            "/api/auth/login", json={"username": "admin", "password": "password"}
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["token_type"], "bearer")
        self.assertEqual(login.json()["expires_in"], self.tokens.ttl_seconds)
        token = login.json()["access_token"]
        response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user_id": self.admin.id})

    def test_invalid_credentials_and_tokens_return_401(self):
        client = TestClient(self.app)
        self.assertEqual(
            client.post(
                "/api/auth/login", json={"username": "admin", "password": "wrong"}
            ).status_code,
            401,
        )
        self.assertEqual(client.get("/protected").status_code, 401)
        self.assertEqual(
            client.get(
                "/protected", headers={"Authorization": "Bearer altered.token.value"}
            ).status_code,
            401,
        )
        expired = self.tokens.issue("user-1", now=int(time.time()) - 120)
        self.assertEqual(
            client.get("/protected", headers={"Authorization": f"Bearer {expired}"}).status_code,
            401,
        )


if __name__ == "__main__":
    unittest.main()
