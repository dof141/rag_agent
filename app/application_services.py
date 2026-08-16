import os
from dataclasses import dataclass
from pathlib import Path

from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService, PasswordHasher
from app.embedding.factory import create_embedding_provider
from app.import_process.runtime import ImportRuntime
from app.import_process.task_repository import MongoTaskRepository
from app.persistence.sqlite_database import SQLiteDatabase
from app.runtime_settings.crypto import SecretCipher
from app.runtime_settings.repository import RuntimeSettingsRepository
from app.runtime_settings.service import RuntimeSettingsService
from app.vector_store.factory import create_vector_store


REQUIRED_CONFIG = (
    "RAG_SQLITE_PATH",
    "RAG_ADMIN_USERNAME",
    "RAG_ADMIN_PASSWORD",
    "RAG_JWT_SECRET",
    "RAG_JWT_TTL_SECONDS",
    "RAG_SETTINGS_MASTER_KEY",
)


class ApplicationConfigurationError(ValueError):
    pass


@dataclass
class ApplicationServices:
    database: SQLiteDatabase
    users: UserRepository
    tokens: JwtTokenService
    settings: RuntimeSettingsService
    task_repository: MongoTaskRepository
    runtime_factory: object
    output_root: Path
    admin_username: str
    admin_password: str

    def initialize_database_only(self) -> None:
        self.database.initialize()

    def initialize(self) -> None:
        self.database.initialize()
        self.users.ensure_admin(self.admin_username, self.admin_password)


def create_import_runtime(snapshot) -> ImportRuntime:
    embedding = create_embedding_provider(snapshot.embedding_config)
    vector_store = create_vector_store(snapshot)
    return ImportRuntime(embedding=embedding, vector_store=vector_store)


def create_application_services(config: dict[str, str | None]) -> ApplicationServices:
    missing = [name for name in REQUIRED_CONFIG if not config.get(name)]
    if missing:
        raise ApplicationConfigurationError("缺少应用配置：" + ", ".join(missing))

    database = SQLiteDatabase(Path(config["RAG_SQLITE_PATH"]))
    passwords = PasswordHasher()
    users = UserRepository(database, passwords)
    tokens = JwtTokenService(
        secret=config["RAG_JWT_SECRET"],
        ttl_seconds=int(config["RAG_JWT_TTL_SECONDS"]),
    )
    settings = RuntimeSettingsService(
        RuntimeSettingsRepository(database),
        SecretCipher(config["RAG_SETTINGS_MASTER_KEY"]),
    )
    return ApplicationServices(
        database=database,
        users=users,
        tokens=tokens,
        settings=settings,
        task_repository=MongoTaskRepository(),
        runtime_factory=create_import_runtime,
        output_root=Path(config.get("RAG_OUTPUT_ROOT") or Path.cwd() / "output"),
        admin_username=config["RAG_ADMIN_USERNAME"],
        admin_password=config["RAG_ADMIN_PASSWORD"],
    )


def create_application_services_from_env() -> ApplicationServices:
    return create_application_services({name: os.getenv(name) for name in REQUIRED_CONFIG})
