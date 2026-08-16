from datetime import datetime, timezone
from typing import Any

from app.persistence.sqlite_database import SQLiteDatabase


SETTINGS_COLUMNS = (
    "embedding_provider",
    "embedding_base_url",
    "embedding_model",
    "embedding_dimension",
    "embedding_batch_size",
    "embedding_timeout",
    "embedding_api_key_encrypted",
    "vector_store_type",
    "qdrant_url",
    "qdrant_api_key_encrypted",
    "qdrant_item_collection",
    "qdrant_chunks_collection",
    "qdrant_cloud_inference",
    "milvus_url",
    "milvus_token_encrypted",
    "milvus_item_collection",
    "milvus_chunks_collection",
)

SECRET_COLUMNS = {
    "embedding_api_key": "embedding_api_key_encrypted",
    "qdrant_api_key": "qdrant_api_key_encrypted",
    "milvus_token": "milvus_token_encrypted",
}


class RuntimeSettingsRepository:
    def __init__(self, database: SQLiteDatabase):
        self._database = database

    def get(self, user_id: str):
        with self._database.connect() as connection:
            return connection.execute(
                "SELECT * FROM runtime_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()

    def upsert(self, user_id: str, values: dict[str, Any]):
        old = self.get(user_id)
        version = 1 if old is None else int(old["version"]) + 1
        updated_at = datetime.now(timezone.utc).isoformat()
        row = {
            **{column: values.get(column) for column in SETTINGS_COLUMNS},
            "user_id": user_id,
            "version": version,
            "updated_at": updated_at,
        }
        row["qdrant_cloud_inference"] = 1 if row["qdrant_cloud_inference"] else 0
        columns = ("user_id", *SETTINGS_COLUMNS, "version", "updated_at")
        placeholders = ", ".join("?" for _ in columns)
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in (*SETTINGS_COLUMNS, "version", "updated_at")
        )
        with self._database.connect() as connection:
            connection.execute(
                f"""
                INSERT INTO runtime_settings ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(user_id) DO UPDATE SET {assignments}
                """,
                tuple(row[column] for column in columns),
            )
        return self.get(user_id)

    def clear_secret(self, user_id: str, secret_name: str):
        column = SECRET_COLUMNS[secret_name]
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                f"""
                UPDATE runtime_settings
                SET {column} = NULL,
                    version = version + 1,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (updated_at, user_id),
            )
        return self.get(user_id)
