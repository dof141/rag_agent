import asyncio
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.auth.models import User
from app.embedding.interface import EmbeddingConfigurationError, EmbeddingError
from app.query_process.agent.main_graph import build_query_graph
from app.query_process.agent.state import create_query_default_state
from app.reranker.interface import RerankerError
from app.retrieval.vector_search import VectorSearchError
from app.runtime_settings.service import RuntimeSettingsConfigurationError
from app.utils.sse_utils import create_sse_queue, push_to_session, remove_sse_queue
from app.utils.task_utils import get_task_result
from app.vector_store.interface import VectorStoreConfigurationError, VectorStoreError


QueryStatus = Literal["processing", "final", "confirmation_required", "error"]
DEFAULT_REQUEST_TTL_SECONDS = 15 * 60
EVENT_QUEUE_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class QueryRequest:
    query: str
    session_id: str | None = None
    is_stream: bool = False


@dataclass(frozen=True)
class ConfirmRequest:
    session_id: str
    pending_request_id: str
    candidate_id: str


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class QueryEvent:
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class QueryResponse:
    request_id: str
    session_id: str
    status: QueryStatus
    answer: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error: ErrorPayload | None = None


class QueryEngineError(RuntimeError):
    def __init__(self, error: ErrorPayload):
        super().__init__(error.message)
        self.error = error


class QueryConfigurationError(QueryEngineError):
    pass


class QueryNotFoundError(QueryEngineError):
    pass


class QueryConflictError(QueryEngineError):
    pass


class QueryUnavailableError(QueryEngineError):
    pass


@dataclass
class _RequestRecord:
    request_id: str
    user_id: str
    session_id: str
    graph: Any
    runtime: Any
    is_stream: bool
    event_queue: queue.Queue | None = None
    status: QueryStatus = "processing"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    terminal: QueryResponse | None = None
    confirmation_consumed: bool = False
    confirmation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    events_claimed: bool = False
    events_consumed: bool = False
    delivery_interrupted: bool = False
    expires_at: float = 0.0
    task: asyncio.Task | None = field(default=None, repr=False)


class QueryEngine:
    def __init__(
        self,
        services,
        *,
        graph_builder: Callable[[Any, Any], Any] = build_query_graph,
        checkpointer=None,
        request_id_factory: Callable[[], str] | None = None,
        request_ttl_seconds: float = DEFAULT_REQUEST_TTL_SECONDS,
        time_factory: Callable[[], float] = time.monotonic,
    ):
        self._services = services
        self._graph_builder = graph_builder
        self._checkpointer = checkpointer or InMemorySaver()
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))
        self._request_ttl_seconds = request_ttl_seconds
        self._time = time_factory
        self._requests: dict[str, _RequestRecord] = {}
        self._tasks: set[asyncio.Task] = set()

    async def ask(self, user: User, request: QueryRequest) -> QueryResponse:
        self._purge_expired()
        session_id = request.session_id or str(uuid.uuid4())
        try:
            snapshot = self._services.settings.get_snapshot(user.id)
            runtime = self._services.query_runtime_factory(snapshot)
            graph = self._graph_builder(runtime, self._checkpointer)
        except (
            RuntimeSettingsConfigurationError,
            EmbeddingConfigurationError,
            VectorStoreConfigurationError,
            ValueError,
        ) as exc:
            raise QueryConfigurationError(_configuration_error()) from exc
        except Exception as exc:
            raise QueryUnavailableError(_public_runtime_error(exc)) from exc

        request_id = self._new_request_id()
        event_queue = create_sse_queue(request_id) if request.is_stream else None
        record = _RequestRecord(
            request_id=request_id,
            user_id=user.id,
            session_id=session_id,
            graph=graph,
            runtime=runtime,
            is_stream=request.is_stream,
            event_queue=event_queue,
            expires_at=self._time() + self._request_ttl_seconds,
        )
        self._requests[request_id] = record
        graph_input = create_query_default_state(
            request_id=request_id,
            session_id=session_id,
            original_query=request.query,
            is_stream=request.is_stream,
        )

        if request.is_stream:
            self._start(record, self._run_background(record, graph_input))
            return self._response(record, status="processing")
        return await self._run(record, graph_input)

    async def confirm(self, user: User, request: ConfirmRequest) -> QueryResponse:
        self._purge_expired()
        pending = self._owned_request(
            user,
            request.pending_request_id,
            session_id=request.session_id,
        )
        await self._consume_confirmation(pending, request.candidate_id)
        request_id = self._new_request_id()
        record = _RequestRecord(
            request_id=request_id,
            user_id=user.id,
            session_id=request.session_id,
            graph=pending.graph,
            runtime=pending.runtime,
            is_stream=True,
            event_queue=create_sse_queue(request_id),
            expires_at=self._time() + self._request_ttl_seconds,
        )
        self._requests[request_id] = record
        command = Command(
            resume=request.candidate_id,
            update={"request_id": request_id, "is_stream": True},
        )
        self._remove_record(pending.request_id, cancel_task=False)
        self._start(record, self._run_background(record, command))
        return self._response(record, status="processing")

    async def _consume_confirmation(
        self,
        pending: _RequestRecord,
        candidate_id: str,
    ) -> None:
        async with pending.confirmation_lock:
            if pending.status != "confirmation_required" or pending.confirmation_consumed:
                raise QueryConflictError(
                    ErrorPayload(
                        code="query_not_waiting_confirmation",
                        message="当前问答请求不在等待确认状态",
                        retryable=False,
                    )
                )
            try:
                snapshot = await pending.graph.aget_state(self._graph_config(pending))
            except Exception as exc:
                raise QueryUnavailableError(_public_runtime_error(exc)) from exc
            values = snapshot.values if snapshot is not None else {}
            if (
                not values.get("awaiting_confirmation")
                or values.get("request_id") != pending.request_id
            ):
                raise QueryConflictError(
                    ErrorPayload(
                        code="query_not_waiting_confirmation",
                        message="当前问答请求不在等待确认状态",
                        retryable=False,
                    )
                )

            candidates = self._normalize_candidates(values.get("candidate_items"))
            if candidate_id not in {str(candidate["id"]) for candidate in candidates}:
                raise QueryConflictError(
                    ErrorPayload(
                        code="query_candidate_invalid",
                        message="候选项无效",
                        retryable=False,
                    )
                )
            pending.confirmation_consumed = True

    async def events(
        self,
        user: User,
        request_id: str,
    ) -> AsyncIterator[QueryEvent]:
        self._purge_expired()
        record = self._owned_request(user, request_id)
        if record.events_claimed or record.events_consumed:
            raise QueryNotFoundError(
                ErrorPayload(
                    code="query_request_not_found",
                    message="问答请求不存在",
                    retryable=False,
                )
            )
        record.events_claimed = True
        if record.delivery_interrupted and record.terminal is not None:
            terminal = record.terminal

            async def replay_terminal():
                delivered = True
                try:
                    yield QueryEvent(terminal.status, self._terminal_data(terminal))
                finally:
                    self._finish_delivery(record, delivered)

            return replay_terminal()
        if record.event_queue is None:
            terminal = record.terminal

            async def completed_iterator():
                delivered = terminal is not None
                try:
                    if terminal is not None:
                        yield QueryEvent(terminal.status, self._terminal_data(terminal))
                finally:
                    self._finish_delivery(record, delivered)

            return completed_iterator()

        event_queue = record.event_queue

        async def iterator():
            terminal_delivered = False
            try:
                while True:
                    try:
                        message = await asyncio.to_thread(
                            event_queue.get,
                            True,
                            EVENT_QUEUE_POLL_SECONDS,
                        )
                    except queue.Empty:
                        continue
                    event_type = message.get("event", "error")
                    if event_type == "__close__":
                        return
                    data = message.get("data") or {}
                    terminal_delivered = event_type in {
                        "final",
                        "confirmation_required",
                        "error",
                    }
                    yield QueryEvent(event_type, data)
                    if terminal_delivered:
                        return
            finally:
                self._finish_delivery(record, terminal_delivered)

        return iterator()

    async def close(self) -> None:
        for record in self._requests.values():
            if record.event_queue is not None:
                push_to_session(record.request_id, "__close__", {})
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        for request_id in list(self._requests):
            self._remove_record(request_id, cancel_task=False)

    async def _run_background(self, record: _RequestRecord, graph_input: Any) -> None:
        try:
            await self._run(record, graph_input)
        except QueryEngineError:
            return

    async def _run(self, record: _RequestRecord, graph_input: Any) -> QueryResponse:
        try:
            result = await record.graph.ainvoke(
                graph_input,
                config=self._graph_config(record),
            )
            candidates = self._extract_candidates(result)
            if candidates:
                response = self._response(
                    record,
                    status="confirmation_required",
                    candidates=candidates,
                )
                record.candidates = candidates
            else:
                graph_answer = result.get("answer") or ""
                answer = (
                    graph_answer
                    if record.is_stream
                    else get_task_result(record.request_id, "answer", graph_answer)
                )
                if not answer:
                    raise QueryUnavailableError(
                        ErrorPayload(
                            code="query_empty_answer",
                            message="问答流程未产生有效答案",
                            retryable=True,
                        )
                    )
                response = self._response(
                    record,
                    status="final",
                    answer=answer,
                    warnings=list(result.get("warnings") or []),
                )
            record.status = response.status
            record.terminal = response
            self._enqueue_terminal(record, response)
            if record.events_consumed and response.status in {"final", "error"}:
                self._remove_record(record.request_id, cancel_task=False)
            return response
        except QueryEngineError as exc:
            self._store_error(record, exc.error)
            raise
        except Exception as exc:
            error = _public_runtime_error(exc)
            self._store_error(record, error)
            raise QueryUnavailableError(error) from exc

    def _store_error(self, record: _RequestRecord, error: ErrorPayload) -> None:
        response = self._response(record, status="error", error=error)
        record.status = "error"
        record.terminal = response
        self._enqueue_terminal(record, response)
        if record.events_consumed:
            self._remove_record(record.request_id, cancel_task=False)

    def _enqueue_terminal(self, record: _RequestRecord, response: QueryResponse) -> None:
        if record.event_queue is None:
            return
        if not self._queue_contains_terminal(record.event_queue):
            push_to_session(
                record.request_id,
                response.status,
                self._terminal_data(response),
            )
        push_to_session(record.request_id, "__close__", {})

    @staticmethod
    def _queue_contains_terminal(event_queue: queue.Queue) -> bool:
        with event_queue.mutex:
            return any(
                message.get("event") in {"final", "confirmation_required", "error"}
                for message in event_queue.queue
            )

    @staticmethod
    def _terminal_data(response: QueryResponse) -> dict[str, Any]:
        if response.error is not None:
            return {
                "code": response.error.code,
                "message": response.error.message,
                "retryable": response.error.retryable,
            }
        if response.status == "confirmation_required":
            return {
                "request_id": response.request_id,
                "session_id": response.session_id,
                "candidates": response.candidates,
            }
        return {
            "request_id": response.request_id,
            "session_id": response.session_id,
            "answer": response.answer,
            "warnings": response.warnings,
        }

    def _owned_request(
        self,
        user: User,
        request_id: str,
        *,
        session_id: str | None = None,
    ) -> _RequestRecord:
        record = self._requests.get(request_id)
        if (
            record is None
            or record.user_id != user.id
            or (session_id is not None and record.session_id != session_id)
        ):
            raise QueryNotFoundError(
                ErrorPayload(
                    code="query_request_not_found",
                    message="问答请求不存在",
                    retryable=False,
                )
            )
        return record

    def _new_request_id(self) -> str:
        request_id = self._request_id_factory()
        while request_id in self._requests:
            request_id = self._request_id_factory()
        return request_id

    @staticmethod
    def _graph_config(record: _RequestRecord) -> dict[str, dict[str, str]]:
        return {
            "configurable": {
                "thread_id": f"{record.user_id}:{record.session_id}",
            }
        }

    def _start(self, record: _RequestRecord, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        record.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _finish_delivery(
        self,
        record: _RequestRecord,
        terminal_delivered: bool,
    ) -> None:
        record.events_claimed = False
        if not terminal_delivered:
            record.delivery_interrupted = True
            return
        record.events_consumed = True
        if record.event_queue is not None:
            remove_sse_queue(record.request_id)
            record.event_queue = None
        if record.terminal is not None and record.terminal.status in {"final", "error"}:
            self._remove_record(record.request_id, cancel_task=False)

    def _purge_expired(self) -> None:
        now = self._time()
        for request_id, record in list(self._requests.items()):
            if record.expires_at <= now:
                self._remove_record(request_id)

    def _remove_record(self, request_id: str, *, cancel_task: bool = True) -> None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return
        if cancel_task and record.task is not None and not record.task.done():
            record.task.cancel()
        if record.event_queue is not None:
            push_to_session(record.request_id, "__close__", {})
            remove_sse_queue(record.request_id)
            record.event_queue = None
        record.events_claimed = False
        record.events_consumed = True

    @staticmethod
    def _extract_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
        interrupts = result.get("__interrupt__") or []
        raw_candidates = result.get("candidate_items") or []
        if interrupts:
            payload = getattr(interrupts[0], "value", {}) or {}
            raw_candidates = payload.get("candidates") or raw_candidates
        if not interrupts and not result.get("awaiting_confirmation"):
            return []
        return QueryEngine._normalize_candidates(raw_candidates)

    @staticmethod
    def _normalize_candidates(candidates) -> list[dict[str, Any]]:
        normalized = []
        for candidate in candidates or []:
            if isinstance(candidate, dict):
                candidate_id = candidate.get("id", candidate.get("item_name"))
                if candidate_id is None:
                    continue
                normalized.append({**candidate, "id": str(candidate_id)})
            else:
                value = str(candidate)
                normalized.append({"id": value, "item_name": value})
        return normalized

    @staticmethod
    def _response(
        record: _RequestRecord,
        *,
        status: QueryStatus,
        answer: str | None = None,
        candidates: list[dict[str, Any]] | None = None,
        warnings: list[dict[str, Any]] | None = None,
        error: ErrorPayload | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            request_id=record.request_id,
            session_id=record.session_id,
            status=status,
            answer=answer,
            candidates=candidates or [],
            warnings=warnings or [],
            error=error,
        )


def _configuration_error() -> ErrorPayload:
    return ErrorPayload(
        code="query_configuration_invalid",
        message="问答运行配置不存在或不完整",
        retryable=False,
    )


def _public_runtime_error(exc: Exception) -> ErrorPayload:
    if isinstance(exc, EmbeddingError):
        return ErrorPayload(
            code="embedding_unavailable",
            message="向量生成服务暂时不可用",
            retryable=True,
        )
    if isinstance(exc, (VectorSearchError, VectorStoreError)):
        return ErrorPayload(
            code="vector_store_unavailable",
            message="知识库检索服务暂时不可用",
            retryable=True,
        )
    if isinstance(exc, RerankerError):
        return ErrorPayload(
            code="reranker_unavailable",
            message="重排序服务暂时不可用",
            retryable=True,
        )
    if any(_is_llm_error(item) for item in _exception_chain(exc)):
        return ErrorPayload(
            code="llm_unavailable",
            message="回答生成服务暂时不可用",
            retryable=True,
        )
    return ErrorPayload(
        code="query_internal_error",
        message="问答流程执行失败",
        retryable=False,
    )


def _exception_chain(exc: Exception):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_llm_error(exc: BaseException) -> bool:
    module_name = type(exc).__module__.lower()
    class_name = type(exc).__name__.lower()
    if any(name in module_name for name in ("openai", "langchain", "app.lm")):
        return True
    if "llm" in class_name:
        return True
    traceback = exc.__traceback__
    while traceback is not None:
        frame_module = str(traceback.tb_frame.f_globals.get("__name__", "")).lower()
        if frame_module.startswith("app.lm"):
            return True
        traceback = traceback.tb_next
    return False
