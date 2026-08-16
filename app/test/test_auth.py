import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.auth.repository import UserRepository
from app.auth.security import InvalidTokenError, JwtTokenService, PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        self.database.initialize()
        self.passwords = PasswordHasher()
        self.users = UserRepository(self.database, self.passwords)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_boot_creates_admin_and_second_boot_keeps_password(self):
        first = self.users.ensure_admin("admin", "first-password")
        second = self.users.ensure_admin("different-name", "replacement-password")

        self.assertEqual(first.id, second.id)
        self.assertEqual(first.username, "admin")
        self.assertTrue(self.users.verify_credentials("admin", "first-password"))
        self.assertFalse(self.users.verify_credentials("admin", "replacement-password"))

    def test_jwt_rejects_expired_and_tampered_tokens(self):
        tokens = JwtTokenService(secret="unit-test-signing-secret", ttl_seconds=1)
        expired = tokens.issue("user-1", now=int(time.time()) - 10)
        with self.assertRaises(InvalidTokenError):
            tokens.verify(expired, now=int(time.time()))

        valid = tokens.issue("user-1", now=int(time.time()))
        prefix, payload, signature = valid.split(".")
        tampered = f"{prefix}.{payload[:-1]}A.{signature}"
        with self.assertRaises(InvalidTokenError):
            tokens.verify(tampered, now=int(time.time()))


if __name__ == "__main__":
    unittest.main()
