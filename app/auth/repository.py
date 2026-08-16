from datetime import datetime, timezone
from uuid import uuid4

from app.auth.models import User
from app.auth.security import PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase


class UserRepository:
    def __init__(self, database: SQLiteDatabase, passwords: PasswordHasher):
        self._database = database
        self._passwords = passwords

    def ensure_admin(self, username: str, password: str) -> User:
        with self._database.connect() as connection:
            existing = connection.execute(
                "SELECT id, username, role FROM users WHERE role = ? ORDER BY created_at LIMIT 1",
                ("admin",),
            ).fetchone()
            if existing is not None:
                return self._row_to_user(existing)

            user_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO users (id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    username,
                    self._passwords.hash(password),
                    "admin",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return User(id=user_id, username=username, role="admin")

    def verify_credentials(self, username: str, password: str) -> User | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        if not self._passwords.verify(password, row["password_hash"]):
            return None
        return self._row_to_user(row)

    def get_by_id(self, user_id: str) -> User | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT id, username, role FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def get_by_username(self, username: str) -> User | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT id, username, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def _row_to_user(self, row) -> User:
        return User(id=row["id"], username=row["username"], role=row["role"])
