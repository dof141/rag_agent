import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService, PasswordHasher
from app.import_process.api.file_import_service import create_import_router
from app.import_process.errors import ImportTaskError
from app.import_process.task_repository import TaskRepositoryError
from app.persistence.sqlite_database import SQLiteDatabase


class FakeSettings:
    def __init__(self):
        self.current = None

    def get_snapshot(self, user_id):
        if self.current is None:
            raise ValueError("配置缺失")
        return self.current


class FakeTaskRepository:
    def __init__(self):
        self.tasks = {}

    def upsert(self, task_id, metadata):
        self.tasks[task_id] = {**self.tasks.get(task_id, {}), **metadata}

    def get(self, task_id):
        return self.tasks.get(task_id)


class RecordingRuntimeFactory:
    def __init__(self):
        self.created_versions = []
        self.runtime = SimpleNamespace()
        self.runtimes = []

    def __call__(self, snapshot):
        self.created_versions.append(snapshot.version)
        self.runtime = SimpleNamespace(settings_version=snapshot.version)
        self.runtimes.append(self.runtime)
        return self.runtime


class ImportApiAuthTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        database.initialize()
        self.users = UserRepository(database, PasswordHasher())
        self.user_a = self.users.ensure_admin("admin", "password")
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("user-b", "admin-b", "hash", "admin", "2026-08-16T00:00:00+00:00"),
            )
        self.user_b = self.users.get_by_id("user-b")
        self.tokens = JwtTokenService("unit-test-signing-secret", 60)
        self.settings = FakeSettings()
        self.tasks = FakeTaskRepository()
        self.runtime_factory = RecordingRuntimeFactory()
        self.status_patchers = [
            patch("app.import_process.api.file_import_service.add_running_task"),
            patch("app.import_process.api.file_import_service.add_done_task"),
            patch("app.import_process.api.file_import_service.update_task_status"),
            patch("app.import_process.api.file_import_service.set_task_result"),
        ]
        for patcher in self.status_patchers:
            patcher.start()
        self.services = SimpleNamespace(
            users=self.users,
            tokens=self.tokens,
            settings=self.settings,
            task_repository=self.tasks,
            runtime_factory=self.runtime_factory,
            output_root=Path(self.temp_dir.name) / "output",
        )
        app = FastAPI()
        app.include_router(create_import_router(self.services))
        self.client = TestClient(app)

    def tearDown(self):
        for patcher in reversed(self.status_patchers):
            patcher.stop()
        self.temp_dir.cleanup()

    def auth(self, user_id):
        return {"Authorization": f"Bearer {self.tokens.issue(user_id)}"}

    def snapshot(self, version=1):
        return SimpleNamespace(version=version)

    def test_upload_requires_auth_and_configuration_before_file_write(self):
        unauthorized = self.client.post(
            "/upload",
            files={"files": ("manual.md", b"# manual", "text/markdown")},
        )
        self.assertEqual(unauthorized.status_code, 401)

        missing_config = self.client.post(
            "/upload",
            files={"files": ("manual.md", b"# manual", "text/markdown")},
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(missing_config.status_code, 409)
        self.assertEqual(self.tasks.tasks, {})
        self.assertFalse(self.services.output_root.exists())

    def test_upload_uses_jwt_user_and_status_hides_other_users_tasks(self):
        self.settings.current = self.snapshot(version=1)
        response = self.client.post(
            "/upload",
            files={"files": ("../manual.md", b"# manual", "text/markdown")},
            data={"user_id": "forged"},
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_ids"][0]
        self.assertEqual(self.tasks.tasks[task_id]["user_id"], self.user_a.id)
        self.assertEqual(self.runtime_factory.created_versions, [1])

        hidden = self.client.get(f"/status/{task_id}", headers=self.auth(self.user_b.id))
        self.assertEqual(hidden.status_code, 404)

    def test_multi_file_upload_creates_one_runtime_per_task(self):
        self.settings.current = self.snapshot(version=1)

        with patch(
            "app.import_process.api.file_import_service.run_graph_task"
        ) as run_graph_task:
            response = self.client.post(
                "/upload",
                files=[
                    ("files", ("one.md", b"# one", "text/markdown")),
                    ("files", ("two.md", b"# two", "text/markdown")),
                ],
                headers=self.auth(self.user_a.id),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime_factory.created_versions, [1, 1])
        self.assertIsNot(
            self.runtime_factory.runtimes[0],
            self.runtime_factory.runtimes[1],
        )
        self.assertEqual(run_graph_task.call_count, 2)

    def test_upload_returns_503_when_task_persistence_fails(self):
        class FailingTaskRepository:
            def upsert(self, task_id, metadata):
                raise RuntimeError("database unavailable")

            def get(self, task_id):
                return None

        self.settings.current = self.snapshot(version=1)
        self.services.task_repository = FailingTaskRepository()

        with patch(
            "app.import_process.api.file_import_service.run_graph_task"
        ) as run_graph_task:
            response = self.client.post(
                "/upload",
                files={"files": ("manual.md", b"# manual", "text/markdown")},
                headers=self.auth(self.user_a.id),
            )

        self.assertEqual(response.status_code, 503)
        run_graph_task.assert_not_called()

    def test_partial_multi_file_failure_never_leaves_processing_task(self):
        class FailAfterFirstWriteRepository:
            def __init__(self):
                self.tasks = {}
                self.calls = 0

            def upsert(self, task_id, metadata):
                self.calls += 1
                if self.calls > 1:
                    raise TaskRepositoryError("database unavailable")
                self.tasks[task_id] = dict(metadata)

            def get(self, task_id):
                return self.tasks.get(task_id)

        repository = FailAfterFirstWriteRepository()
        self.settings.current = self.snapshot(version=1)
        self.services.task_repository = repository

        with patch(
            "app.import_process.api.file_import_service.run_graph_task"
        ) as run_graph_task:
            response = self.client.post(
                "/upload",
                files=[
                    ("files", ("one.md", b"# one", "text/markdown")),
                    ("files", ("two.md", b"# two", "text/markdown")),
                ],
                headers=self.auth(self.user_a.id),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(repository.tasks), 1)
        first_task = next(iter(repository.tasks.values()))
        self.assertEqual(first_task["status"], "pending")
        run_graph_task.assert_not_called()

    def test_status_returns_503_when_task_store_is_unavailable(self):
        class FailingReadRepository:
            def get(self, task_id):
                raise TaskRepositoryError("database unavailable")

        self.services.task_repository = FailingReadRepository()

        response = self.client.get(
            "/status/task-a",
            headers=self.auth(self.user_a.id),
        )

        self.assertEqual(response.status_code, 503)

    def test_status_reuses_strict_repository_snapshot_without_second_query(self):
        task_id = "task-strict-snapshot"
        self.tasks.upsert(
            task_id,
            {
                "user_id": self.user_a.id,
                "status": "failed",
                "error": "文档解析失败",
                "failed_stage": "document_parse",
                "done_list": ["upload_file"],
            },
        )

        with patch(
            "app.utils.task_utils.mongo_get_task",
            side_effect=AssertionError("must not query Mongo twice"),
        ):
            response = self.client.get(
                f"/status/{task_id}",
                headers=self.auth(self.user_a.id),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(response.json()["failed_stage"], "document_parse")

    def test_retry_updates_settings_version_before_scheduling(self):
        source = Path(self.temp_dir.name) / "manual.md"
        source.write_text("# manual", encoding="utf-8")
        self.tasks.upsert(
            "task-retry",
            {
                "user_id": self.user_a.id,
                "document_id": "document-a",
                "local_dir": self.temp_dir.name,
                "local_file_path": str(source),
                "settings_version": 1,
                "status": "failed",
            },
        )
        self.settings.current = self.snapshot(version=2)

        with patch(
            "app.import_process.api.file_import_service.run_graph_task"
        ) as run_graph_task:
            response = self.client.post(
                "/retry/task-retry",
                headers=self.auth(self.user_a.id),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.tasks.tasks["task-retry"]["settings_version"], 2)
        self.assertEqual(self.tasks.tasks["task-retry"]["status"], "pending")
        self.assertEqual(self.runtime_factory.created_versions, [2])
        run_graph_task.assert_called_once()

    def test_run_graph_task_maps_public_import_errors(self):
        from app.import_process.api.file_import_service import run_graph_task

        def failing_graph(runtime):
            class Graph:
                def stream(self, state):
                    raise ImportTaskError("vector_store", "向量库写入失败")

            return Graph()

        task_id = "task-a"
        self.tasks.upsert(task_id, {"user_id": self.user_a.id})
        run_graph_task(
            task_id,
            str(self.services.output_root),
            "manual.md",
            self.user_a.id,
            "doc-a",
            self.runtime_factory.runtime,
            task_repository=self.tasks,
            graph_builder=failing_graph,
        )

        self.assertEqual(self.tasks.tasks[task_id]["status"], "failed")
        self.assertEqual(self.tasks.tasks[task_id]["failed_stage"], "vector_store")
        self.assertEqual(self.tasks.tasks[task_id]["error"], "向量库写入失败")

    def test_run_graph_task_stops_before_graph_when_status_sync_fails(self):
        from app.import_process.api.file_import_service import run_graph_task

        graph_built = []

        def graph_builder(runtime):
            graph_built.append(runtime)
            raise AssertionError("graph must not be built")

        def update_status(_task_id, _status, *, persist=True, **_kwargs):
            if persist:
                raise TaskRepositoryError("database unavailable")

        with patch(
            "app.import_process.api.file_import_service.update_task_status",
            side_effect=update_status,
        ):
            run_graph_task(
                "task-persistence",
                str(self.services.output_root),
                "manual.md",
                self.user_a.id,
                "doc-a",
                self.runtime_factory.runtime,
                task_repository=self.tasks,
                graph_builder=graph_builder,
            )

        self.assertEqual(graph_built, [])


if __name__ == "__main__":
    unittest.main()
