import unittest
import warnings
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient

from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService, PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase


class FakeQueryEngine:
    def __init__(self):
        self.ask_error = None
        self.confirm_error = None
        self.event_error = None

    async def ask(self, user, request):
        if self.ask_error:
            raise self.ask_error
        from app.query_process.engine import QueryResponse

        return QueryResponse(
            request_id="request-1",
            session_id=request.session_id or "generated-session",
            status="processing" if request.is_stream else "final",
            answer=None if request.is_stream else "answer",
        )

    async def confirm(self, user, request):
        if self.confirm_error:
            raise self.confirm_error
        from app.query_process.engine import QueryResponse

        return QueryResponse(
            request_id="request-2",
            session_id=request.session_id,
            status="processing",
        )

    async def events(self, user, request_id):
        if self.event_error:
            raise self.event_error
        from app.query_process.engine import QueryEvent

        async def iterator():
            yield QueryEvent("final", {"answer": "answer", "warnings": []})

        return iterator()


class QueryHttpTest(unittest.TestCase):
    def setUp(self):
        from app.query_process.api.router import create_query_router

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
        self.engine = FakeQueryEngine()
        services = type("Services", (), {"users": self.users, "tokens": self.tokens})()
        app = FastAPI()
        app.include_router(create_query_router(services, engine=self.engine))
        self.client = TestClient(app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def auth(self, user_id):
        return {"Authorization": f"Bearer {self.tokens.issue(user_id)}"}

    def test_query_confirm_and_stream_require_authentication(self):
        responses = (
            self.client.post("/query", json={"query": "question"}),
            self.client.post(
                "/query/confirm",
                json={
                    "session_id": "session-1",
                    "pending_request_id": "pending-1",
                    "candidate_id": "course-a",
                },
            ),
            self.client.get("/stream/request-1"),
        )
        self.assertEqual([response.status_code for response in responses], [401, 401, 401])

    def test_configuration_and_runtime_failures_are_not_empty_200_answers(self):
        from app.query_process.engine import (
            ErrorPayload,
            QueryConfigurationError,
            QueryUnavailableError,
        )

        cases = (
            (
                QueryConfigurationError(
                    ErrorPayload(
                        code="query_configuration_invalid",
                        message="问答运行配置不存在或不完整",
                        retryable=False,
                    )
                ),
                409,
            ),
            (
                QueryUnavailableError(
                    ErrorPayload(
                        code="embedding_unavailable",
                        message="向量生成服务暂时不可用",
                        retryable=True,
                    )
                ),
                503,
            ),
        )
        for error, expected_status in cases:
            with self.subTest(code=error.error.code):
                self.engine.ask_error = error
                response = self.client.post(
                    "/query",
                    json={"query": "question"},
                    headers=self.auth(self.user_a.id),
                )
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json(), asdict(error.error))
                self.assertNotEqual(response.json().get("answer"), "")

    def test_cross_user_stream_and_confirm_are_404_and_invalid_candidate_is_409(self):
        from app.query_process.engine import (
            ErrorPayload,
            QueryConflictError,
            QueryNotFoundError,
        )

        not_found = QueryNotFoundError(
            ErrorPayload(
                code="query_request_not_found",
                message="问答请求不存在",
                retryable=False,
            )
        )
        self.engine.event_error = not_found
        hidden_stream = self.client.get(
            "/stream/request-1", headers=self.auth(self.user_b.id)
        )
        self.assertEqual(hidden_stream.status_code, 404)
        self.assertEqual(hidden_stream.json(), asdict(not_found.error))

        self.engine.confirm_error = not_found
        hidden_confirm = self.client.post(
            "/query/confirm",
            json={
                "session_id": "session-1",
                "pending_request_id": "pending-1",
                "candidate_id": "course-a",
            },
            headers=self.auth(self.user_b.id),
        )
        self.assertEqual(hidden_confirm.status_code, 404)

        conflict = QueryConflictError(
            ErrorPayload(
                code="query_candidate_invalid",
                message="候选项无效",
                retryable=False,
            )
        )
        self.engine.confirm_error = conflict
        invalid = self.client.post(
            "/query/confirm",
            json={
                "session_id": "session-1",
                "pending_request_id": "pending-1",
                "candidate_id": "invalid",
            },
            headers=self.auth(self.user_a.id),
        )
        self.assertEqual(invalid.status_code, 409)
        self.assertEqual(invalid.json(), asdict(conflict.error))

    def test_stream_serializes_structured_sse_terminal(self):
        response = self.client.get(
            "/stream/request-1", headers=self.auth(self.user_a.id)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: final", response.text)
        self.assertIn('"answer": "answer"', response.text)


if __name__ == "__main__":
    unittest.main()
