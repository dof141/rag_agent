import asyncio
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import patch

from app.auth.models import User
from app.runtime_settings.service import RuntimeSettingsConfigurationError


class FakeSettings:
    def __init__(self, snapshot=None, error=None):
        self.snapshot = snapshot
        self.error = error
        self.user_ids = []

    def get_snapshot(self, user_id):
        self.user_ids.append(user_id)
        if self.error is not None:
            raise self.error
        return self.snapshot


class FakeGraph:
    def __init__(self, results):
        self.results = list(results)
        self.invocations = []
        self.snapshot_values = {}
        self.state_error = None

    async def ainvoke(self, graph_input, config):
        self.invocations.append((graph_input, config))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result.get("__interrupt__"):
            self.snapshot_values = {
                "request_id": graph_input.get("request_id"),
                "awaiting_confirmation": True,
                "candidate_items": result.get("candidate_items", []),
            }
        else:
            self.snapshot_values = dict(result)
        return result

    async def aget_state(self, config):
        if self.state_error is not None:
            raise self.state_error
        return SimpleNamespace(values=dict(self.snapshot_values))


class RecordingGraphBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.calls = []

    def __call__(self, runtime, checkpointer):
        self.calls.append((runtime, checkpointer))
        return self.graph


class HangingGraph(FakeGraph):
    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()

    async def ainvoke(self, graph_input, config):
        self.started.set()
        await asyncio.Event().wait()


class ControlledGraph(FakeGraph):
    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, graph_input, config):
        self.started.set()
        await self.release.wait()
        return {"answer": "answer", "warnings": []}


class QueryEngineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.query_process.engine import QueryEngine

        self.QueryEngine = QueryEngine
        self.user_a = User(id="user-a", username="a", role="admin")
        self.user_b = User(id="user-b", username="b", role="admin")
        self.snapshot = SimpleNamespace(user_id="user-a", version=7)
        self.runtime = SimpleNamespace(user_id="user-a", settings_version=7)

    def services(self, graph, *, settings=None):
        settings = settings or FakeSettings(self.snapshot)
        runtime_snapshots = []

        def runtime_factory(snapshot):
            runtime_snapshots.append(snapshot)
            return self.runtime

        services = SimpleNamespace(
            settings=settings,
            query_runtime_factory=runtime_factory,
        )
        services.runtime_snapshots = runtime_snapshots
        return services

    async def test_ask_freezes_runtime_qualifies_thread_and_reads_answer_by_request_id(self):
        from app.query_process.engine import QueryRequest

        graph = FakeGraph([{"answer": "graph answer", "warnings": []}])
        builder = RecordingGraphBuilder(graph)
        services = self.services(graph)
        engine = self.QueryEngine(
            services,
            graph_builder=builder,
            request_id_factory=lambda: "request-1",
        )

        with patch(
            "app.query_process.engine.get_task_result",
            return_value="stored answer",
        ) as get_result:
            response = await engine.ask(
                self.user_a,
                QueryRequest(query="question", session_id="session-1"),
            )

        self.assertEqual(response.answer, "stored answer")
        self.assertEqual(response.status, "final")
        self.assertEqual(services.settings.user_ids, ["user-a"])
        self.assertEqual(services.runtime_snapshots, [self.snapshot])
        self.assertIs(builder.calls[0][0], self.runtime)
        self.assertEqual(
            graph.invocations[0][1]["configurable"]["thread_id"],
            "user-a:session-1",
        )
        get_result.assert_called_once_with("request-1", "answer", "graph answer")

    async def test_configuration_failure_is_stable_and_does_not_build_graph(self):
        from app.query_process.engine import QueryConfigurationError, QueryRequest

        settings = FakeSettings(
            error=RuntimeSettingsConfigurationError("secret-provider-details")
        )
        graph = FakeGraph([])
        builder = RecordingGraphBuilder(graph)
        engine = self.QueryEngine(self.services(graph, settings=settings), graph_builder=builder)

        with self.assertRaises(QueryConfigurationError) as caught:
            await engine.ask(self.user_a, QueryRequest(query="question"))

        self.assertEqual(caught.exception.error.code, "query_configuration_invalid")
        self.assertNotIn("secret-provider-details", caught.exception.error.message)
        self.assertEqual(builder.calls, [])

    async def test_unknown_graph_build_failure_is_sanitized_as_unavailable(self):
        from app.query_process.engine import QueryRequest, QueryUnavailableError

        def broken_builder(runtime, checkpointer):
            raise RuntimeError("builder secret detail")

        engine = self.QueryEngine(self.services(FakeGraph([])), graph_builder=broken_builder)

        with self.assertRaises(QueryUnavailableError) as caught:
            await engine.ask(self.user_a, QueryRequest(query="question"))

        self.assertEqual(caught.exception.error.code, "query_internal_error")
        self.assertNotIn("secret", caught.exception.error.message)

    async def test_confirm_hides_owner_and_session_then_rejects_invalid_candidate(self):
        from app.query_process.engine import (
            ConfirmRequest,
            QueryConflictError,
            QueryNotFoundError,
            QueryRequest,
        )

        candidates = [{"id": "course-a", "item_name": "Course A"}]
        interrupt = SimpleNamespace(value={"candidates": candidates})
        graph = FakeGraph(
            [
                {
                    "__interrupt__": [interrupt],
                    "candidate_items": candidates,
                    "awaiting_confirmation": True,
                },
                {"answer": "confirmed answer", "warnings": []},
            ]
        )
        ids = iter(["pending-1", "confirmed-1", "confirmed-2"])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: next(ids),
        )
        pending = await engine.ask(
            self.user_a,
            QueryRequest(query="question", session_id="session-1"),
        )
        self.assertEqual(pending.status, "confirmation_required")

        for user, session_id in (
            (self.user_b, "session-1"),
            (self.user_a, "other-session"),
        ):
            with self.subTest(user=user.id, session=session_id):
                with self.assertRaises(QueryNotFoundError):
                    await engine.confirm(
                        user,
                        ConfirmRequest(
                            session_id=session_id,
                            pending_request_id="pending-1",
                            candidate_id="course-a",
                        ),
                    )

        with self.assertRaises(QueryConflictError):
            await engine.confirm(
                self.user_a,
                ConfirmRequest(
                    session_id="session-1",
                    pending_request_id="pending-1",
                    candidate_id="not-a-candidate",
                ),
            )

        confirmed = await engine.confirm(
            self.user_a,
            ConfirmRequest(
                session_id="session-1",
                pending_request_id="pending-1",
                candidate_id="course-a",
            ),
        )
        self.assertEqual(confirmed.request_id, "confirmed-1")
        self.assertEqual(confirmed.status, "processing")

        with self.assertRaises(QueryNotFoundError):
            await engine.confirm(
                self.user_a,
                ConfirmRequest(
                    session_id="session-1",
                    pending_request_id="pending-1",
                    candidate_id="course-a",
                ),
            )

        await asyncio.sleep(0)
        command = graph.invocations[1][0]
        self.assertEqual(command.resume, "course-a")
        self.assertEqual(command.update["request_id"], "confirmed-1")
        self.assertEqual(command.update["is_stream"], True)

    async def test_confirm_requires_matching_checkpoint_request_and_candidates(self):
        from app.query_process.engine import ConfirmRequest, QueryConflictError, QueryRequest

        candidates = [{"id": "course-a", "item_name": "Course A"}]
        graph = FakeGraph(
            [
                {
                    "__interrupt__": [SimpleNamespace(value={"candidates": candidates})],
                    "candidate_items": candidates,
                    "awaiting_confirmation": True,
                }
            ]
        )
        ids = iter(["pending-1", "confirmed-1", "confirmed-2"])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: next(ids),
        )
        await engine.ask(
            self.user_a,
            QueryRequest(query="question", session_id="session-1"),
        )

        graph.snapshot_values["request_id"] = "newer-request"
        with self.assertRaises(QueryConflictError):
            await engine.confirm(
                self.user_a,
                ConfirmRequest("session-1", "pending-1", "course-a"),
            )

        graph.snapshot_values["request_id"] = "pending-1"
        graph.snapshot_values["candidate_items"] = []
        with self.assertRaises(QueryConflictError):
            await engine.confirm(
                self.user_a,
                ConfirmRequest("session-1", "pending-1", "course-a"),
            )

    async def test_stream_failure_emits_one_sanitized_error_terminal(self):
        from app.query_process.engine import QueryRequest

        graph = FakeGraph([RuntimeError("https://provider.invalid?api_key=secret")])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )

        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", session_id="session-1", is_stream=True),
        )
        events = await engine.events(self.user_a, response.request_id)
        received = [event async for event in events]

        terminal = [event for event in received if event.type in {"final", "confirmation_required", "error"}]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].type, "error")
        self.assertEqual(terminal[0].data["code"], "query_internal_error")
        self.assertNotIn("provider.invalid", terminal[0].data["message"])
        self.assertNotIn("secret", terminal[0].data["message"])

    async def test_wrapped_langchain_failure_is_classified_as_llm_unavailable(self):
        from langchain_core.exceptions import LangChainException

        from app.query_process.engine import QueryRequest

        wrapped = Exception("outer provider detail")
        wrapped.__cause__ = LangChainException("inner provider detail")
        graph = FakeGraph([wrapped])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )
        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", is_stream=True),
        )
        events = await engine.events(self.user_a, response.request_id)
        received = [event async for event in events]

        self.assertEqual(received[-1].data["code"], "llm_unavailable")
        self.assertTrue(received[-1].data["retryable"])

    async def test_confirm_checkpoint_failure_is_sanitized_as_unavailable(self):
        from app.query_process.engine import (
            ConfirmRequest,
            QueryRequest,
            QueryUnavailableError,
        )

        candidates = [{"id": "course-a", "item_name": "Course A"}]
        graph = FakeGraph(
            [
                {
                    "__interrupt__": [SimpleNamespace(value={"candidates": candidates})],
                    "candidate_items": candidates,
                    "awaiting_confirmation": True,
                }
            ]
        )
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "pending-1",
        )
        await engine.ask(
            self.user_a,
            QueryRequest(query="question", session_id="session-1"),
        )
        graph.state_error = RuntimeError("checkpoint secret detail")

        with self.assertRaises(QueryUnavailableError) as caught:
            await engine.confirm(
                self.user_a,
                ConfirmRequest(
                    session_id="session-1",
                    pending_request_id="pending-1",
                    candidate_id="course-a",
                ),
            )

        self.assertEqual(caught.exception.error.code, "query_internal_error")
        self.assertNotIn("secret", caught.exception.error.message)

    async def test_unknown_and_cross_user_streams_are_indistinguishable(self):
        from app.query_process.engine import QueryNotFoundError, QueryRequest

        graph = FakeGraph([{"answer": "answer", "warnings": []}])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )
        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", is_stream=True),
        )

        for user, request_id in (
            (self.user_a, "missing"),
            (self.user_b, response.request_id),
        ):
            with self.subTest(user=user.id, request_id=request_id):
                with self.assertRaises(QueryNotFoundError):
                    await engine.events(user, request_id)

    async def test_stream_can_only_be_subscribed_once(self):
        from app.query_process.engine import QueryNotFoundError, QueryRequest

        graph = FakeGraph([{"answer": "answer", "warnings": []}])
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )
        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", is_stream=True),
        )
        events = await engine.events(self.user_a, response.request_id)
        self.assertEqual([event.type async for event in events], ["final"])

        with self.assertRaises(QueryNotFoundError):
            await engine.events(self.user_a, response.request_id)
        self.assertNotIn(response.request_id, engine._requests)

    async def test_disconnected_stream_can_reconnect_and_replay_terminal(self):
        from app.query_process.engine import QueryRequest

        graph = ControlledGraph()
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )
        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", is_stream=True),
        )
        await graph.started.wait()
        first_events = await engine.events(self.user_a, response.request_id)
        waiting = asyncio.create_task(anext(first_events))
        await asyncio.sleep(0)
        waiting.cancel()
        with suppress(asyncio.CancelledError):
            await waiting

        graph.release.set()
        for _attempt in range(100):
            record = engine._requests[response.request_id]
            if record.terminal is not None:
                break
            await asyncio.sleep(0.01)

        reconnected = await engine.events(self.user_a, response.request_id)
        received = [event async for event in reconnected]
        self.assertEqual([event.type for event in received], ["final"])
        self.assertNotIn(response.request_id, engine._requests)

    async def test_expired_pending_request_is_hidden_and_reclaimed(self):
        from app.query_process.engine import ConfirmRequest, QueryNotFoundError, QueryRequest
        from app.utils.sse_utils import get_sse_queue

        now = [100.0]
        candidates = [{"id": "course-a", "item_name": "Course A"}]
        graph = FakeGraph(
            [
                {
                    "__interrupt__": [SimpleNamespace(value={"candidates": candidates})],
                    "candidate_items": candidates,
                    "awaiting_confirmation": True,
                }
            ]
        )
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "pending-1",
            request_ttl_seconds=10,
            time_factory=lambda: now[0],
        )
        await engine.ask(
            self.user_a,
            QueryRequest(query="question", session_id="session-1", is_stream=True),
        )
        now[0] = 111.0

        with self.assertRaises(QueryNotFoundError):
            await engine.confirm(
                self.user_a,
                ConfirmRequest("session-1", "pending-1", "course-a"),
            )
        self.assertNotIn("pending-1", engine._requests)
        self.assertIsNone(get_sse_queue("pending-1"))

    async def test_close_cancels_background_tasks_and_removes_stream_queue(self):
        from app.query_process.engine import QueryRequest
        from app.utils.sse_utils import get_sse_queue

        graph = HangingGraph()
        engine = self.QueryEngine(
            self.services(graph),
            graph_builder=RecordingGraphBuilder(graph),
            request_id_factory=lambda: "request-1",
        )
        response = await engine.ask(
            self.user_a,
            QueryRequest(query="question", is_stream=True),
        )
        await graph.started.wait()

        await engine.close()

        self.assertEqual(engine._tasks, set())
        self.assertIsNone(get_sse_queue(response.request_id))


if __name__ == "__main__":
    unittest.main()
