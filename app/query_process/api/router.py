import json
from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth.dependencies import build_current_user_dependency
from app.query_process.api.query_server import ConfirmRequest, QueryRequest
from app.query_process.engine import (
    ConfirmRequest as EngineConfirmRequest,
    QueryConfigurationError,
    QueryConflictError,
    QueryEngine,
    QueryNotFoundError,
    QueryRequest as EngineQueryRequest,
    QueryUnavailableError,
)


def create_query_router(services, *, engine: QueryEngine | None = None) -> APIRouter:
    router = APIRouter()
    query_engine = engine or QueryEngine(services)
    current_user = build_current_user_dependency(services.users, services.tokens)
    close_engine = getattr(query_engine, "close", None)
    if close_engine is not None:
        router.add_event_handler("shutdown", close_engine)

    @router.post("/query", summary="问答提问")
    async def query(request: QueryRequest, user=Depends(current_user)):
        try:
            response = await query_engine.ask(
                user,
                EngineQueryRequest(
                    query=request.query,
                    session_id=request.session_id,
                    is_stream=request.is_stream,
                ),
            )
            return asdict(response)
        except QueryConfigurationError as exc:
            return _error_response(exc, 409)
        except QueryUnavailableError as exc:
            return _error_response(exc, 503)

    @router.post("/query/confirm", summary="人工确认歧义实体")
    async def confirm(request: ConfirmRequest, user=Depends(current_user)):
        try:
            response = await query_engine.confirm(
                user,
                EngineConfirmRequest(
                    session_id=request.session_id,
                    pending_request_id=request.pending_request_id,
                    candidate_id=request.candidate_id,
                ),
            )
            return asdict(response)
        except QueryNotFoundError as exc:
            return _error_response(exc, 404)
        except QueryConflictError as exc:
            return _error_response(exc, 409)
        except QueryUnavailableError as exc:
            return _error_response(exc, 503)

    @router.get("/stream/{request_id}", summary="SSE 流式推送到前端")
    async def stream(request_id: str, user=Depends(current_user)):
        try:
            events = await query_engine.events(user, request_id)
        except QueryNotFoundError as exc:
            return _error_response(exc, 404)

        async def encode_events():
            async for event in events:
                payload = json.dumps(event.data, ensure_ascii=False)
                yield f"event: {event.type}\ndata: {payload}\n\n"

        return StreamingResponse(
            encode_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return router


def _error_response(exc, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=asdict(exc.error))
