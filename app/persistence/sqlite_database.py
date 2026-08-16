from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


class SQLiteDatabase:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_settings (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    embedding_provider TEXT NOT NULL,
                    embedding_base_url TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_batch_size INTEGER NOT NULL,
                    embedding_timeout REAL NOT NULL,
                    embedding_api_key_encrypted TEXT,
                    vector_store_type TEXT NOT NULL,
                    qdrant_url TEXT,
                    qdrant_api_key_encrypted TEXT,
                    qdrant_item_collection TEXT,
                    qdrant_chunks_collection TEXT,
                    qdrant_cloud_inference INTEGER NOT NULL DEFAULT 1,
                    milvus_url TEXT,
                    milvus_token_encrypted TEXT,
                    milvus_item_collection TEXT,
                    milvus_chunks_collection TEXT,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
