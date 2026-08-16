from app.conf.embedding_config import EmbeddingConfig
from app.runtime_settings.crypto import SecretCipher
from app.runtime_settings.models import (
    RuntimeSettingsResponse,
    RuntimeSettingsUpdate,
    SecretStatus,
    UserRuntimeSnapshot,
)
from app.runtime_settings.repository import RuntimeSettingsRepository, SECRET_COLUMNS
from app.vector_store.config import MilvusVectorStoreConfig, QdrantVectorStoreConfig


SUPPORTED_COMBINATIONS = {
    ("siliconflow", "qdrant"),
    ("local_bge_m3", "milvus"),
}


class RuntimeSettingsConfigurationError(ValueError):
    pass


class RuntimeSettingsService:
    def __init__(self, repository: RuntimeSettingsRepository, cipher: SecretCipher):
        self._repository = repository
        self._cipher = cipher

    def save(self, user_id: str, payload: RuntimeSettingsUpdate) -> RuntimeSettingsResponse:
        old = self._repository.get(user_id)
        values = {
            "embedding_provider": payload.embedding_provider,
            "embedding_base_url": payload.embedding_base_url,
            "embedding_model": payload.embedding_model,
            "embedding_dimension": payload.embedding_dimension,
            "embedding_batch_size": payload.embedding_batch_size,
            "embedding_timeout": payload.embedding_timeout,
            "embedding_api_key_encrypted": self._secret_value(
                payload.embedding_api_key, old, "embedding_api_key_encrypted"
            ),
            "vector_store_type": payload.vector_store_type,
            "qdrant_url": payload.qdrant_url,
            "qdrant_api_key_encrypted": self._secret_value(
                payload.qdrant_api_key, old, "qdrant_api_key_encrypted"
            ),
            "qdrant_item_collection": payload.qdrant_item_collection,
            "qdrant_chunks_collection": payload.qdrant_chunks_collection,
            "qdrant_cloud_inference": payload.qdrant_cloud_inference,
            "milvus_url": payload.milvus_url,
            "milvus_token_encrypted": self._secret_value(
                payload.milvus_token, old, "milvus_token_encrypted"
            ),
            "milvus_item_collection": payload.milvus_item_collection,
            "milvus_chunks_collection": payload.milvus_chunks_collection,
        }
        self._validate_combination(
            values,
            self._cipher.decrypt(values["embedding_api_key_encrypted"]),
            self._cipher.decrypt(values["qdrant_api_key_encrypted"]),
            self._cipher.decrypt(values["milvus_token_encrypted"]),
        )
        row = self._repository.upsert(user_id, values)
        return self._to_response(row)

    def get_public(self, user_id: str) -> RuntimeSettingsResponse | None:
        row = self._repository.get(user_id)
        if row is None:
            return None
        return self._to_response(row)

    def get_snapshot(self, user_id: str) -> UserRuntimeSnapshot:
        row = self._repository.get(user_id)
        if row is None:
            raise RuntimeSettingsConfigurationError("运行配置不存在")

        embedding_api_key = self._cipher.decrypt(row["embedding_api_key_encrypted"])
        qdrant_api_key = self._cipher.decrypt(row["qdrant_api_key_encrypted"])
        milvus_token = self._cipher.decrypt(row["milvus_token_encrypted"])
        self._validate_combination(row, embedding_api_key, qdrant_api_key, milvus_token)

        embedding_config = EmbeddingConfig(
            adapter=row["embedding_provider"],
            model=row["embedding_model"],
            dimension=int(row["embedding_dimension"]),
            output_type="dense" if row["vector_store_type"] == "qdrant" else "dense&sparse",
            batch_size=int(row["embedding_batch_size"]),
            request_timeout=float(row["embedding_timeout"]),
            base_url=row["embedding_base_url"],
            api_key=embedding_api_key,
            bge_m3_path=None,
            bge_m3=row["embedding_model"],
            bge_device="cpu",
            bge_fp16=False,
        )
        qdrant = None
        milvus = None
        if row["vector_store_type"] == "qdrant":
            qdrant = QdrantVectorStoreConfig(
                url=row["qdrant_url"],
                api_key=qdrant_api_key,
                item_collection=row["qdrant_item_collection"],
                chunks_collection=row["qdrant_chunks_collection"],
                dimension=int(row["embedding_dimension"]),
                cloud_inference=bool(row["qdrant_cloud_inference"]),
            )
        if row["vector_store_type"] == "milvus":
            milvus = MilvusVectorStoreConfig(
                url=row["milvus_url"],
                token=milvus_token,
                item_collection=row["milvus_item_collection"],
                chunks_collection=row["milvus_chunks_collection"],
                dimension=int(row["embedding_dimension"]),
            )
        return UserRuntimeSnapshot(
            user_id=user_id,
            version=int(row["version"]),
            embedding_config=embedding_config,
            vector_store_type=row["vector_store_type"],
            qdrant=qdrant,
            milvus=milvus,
        )

    def clear_secret(self, user_id: str, secret_name: str) -> RuntimeSettingsResponse:
        if secret_name not in SECRET_COLUMNS:
            raise ValueError("未知密钥字段")
        row = self._repository.clear_secret(user_id, secret_name)
        if row is None:
            raise RuntimeSettingsConfigurationError("运行配置不存在")
        return self._to_response(row)

    def _secret_value(self, value: str | None, old, column: str) -> str | None:
        if value == "":
            return None if old is None else old[column]
        if value is None:
            return None if old is None else old[column]
        return self._cipher.encrypt(value)

    def _to_response(self, row) -> RuntimeSettingsResponse:
        return RuntimeSettingsResponse(
            embedding_provider=row["embedding_provider"],
            embedding_base_url=row["embedding_base_url"],
            embedding_model=row["embedding_model"],
            embedding_dimension=int(row["embedding_dimension"]),
            embedding_batch_size=int(row["embedding_batch_size"]),
            embedding_timeout=float(row["embedding_timeout"]),
            embedding_api_key=self._secret_status(row["embedding_api_key_encrypted"]),
            vector_store_type=row["vector_store_type"],
            qdrant_url=row["qdrant_url"],
            qdrant_api_key=self._secret_status(row["qdrant_api_key_encrypted"]),
            qdrant_item_collection=row["qdrant_item_collection"],
            qdrant_chunks_collection=row["qdrant_chunks_collection"],
            qdrant_cloud_inference=bool(row["qdrant_cloud_inference"]),
            milvus_url=row["milvus_url"],
            milvus_token=self._secret_status(row["milvus_token_encrypted"]),
            milvus_item_collection=row["milvus_item_collection"],
            milvus_chunks_collection=row["milvus_chunks_collection"],
            version=int(row["version"]),
            updated_at=row["updated_at"],
        )

    def _secret_status(self, encrypted: str | None) -> SecretStatus:
        value = self._cipher.decrypt(encrypted)
        return SecretStatus(configured=value is not None, masked=self._cipher.mask(value))

    def _validate_combination(
        self,
        row,
        embedding_api_key: str | None,
        qdrant_api_key: str | None,
        milvus_token: str | None,
    ) -> None:
        combination = (row["embedding_provider"], row["vector_store_type"])
        if combination not in SUPPORTED_COMBINATIONS:
            raise RuntimeSettingsConfigurationError("运行配置组合不受支持")
        if combination == ("siliconflow", "qdrant"):
            required = (
                embedding_api_key,
                row["embedding_base_url"],
                qdrant_api_key,
                row["qdrant_url"],
                row["qdrant_item_collection"],
                row["qdrant_chunks_collection"],
            )
            if not all(required) or not bool(row["qdrant_cloud_inference"]):
                raise RuntimeSettingsConfigurationError("SiliconFlow + Qdrant 配置不完整")
        if combination == ("local_bge_m3", "milvus"):
            required = (
                row["milvus_url"],
                row["milvus_item_collection"],
                row["milvus_chunks_collection"],
            )
            if not all(required):
                raise RuntimeSettingsConfigurationError("local_bge_m3 + Milvus 配置不完整")
