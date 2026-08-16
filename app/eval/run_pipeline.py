"""通过正式的用户级问答运行时生成离线评测样本。"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from contextlib import aclosing
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.application_services import create_application_services_from_env
from app.auth.models import User
from app.core.logger import logger
from app.query_process.engine import (
    ConfirmRequest,
    QueryEngine,
    QueryEngineError,
    QueryRequest,
)


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden_set.jsonl"
DEFAULT_OUT = EVAL_DIR / "reports" / "pipeline_outputs.jsonl"
TERMINAL_EVENTS = {"final", "confirmation_required", "error"}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "app" and parts[1] == "eval":
        p = Path(*parts[2:]) if len(parts) > 2 else Path()
    return (EVAL_DIR / p).resolve()


def _contexts_from_sources(sources: Any, max_contexts: int) -> list[str]:
    contexts: list[str] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        text = str(source.get("content") or source.get("text") or "").strip()
        if text:
            contexts.append(text)
        if len(contexts) >= max_contexts:
            break
    return contexts


async def _consume_events(
    engine: QueryEngine,
    user: User,
    request_id: str,
    *,
    max_contexts: int,
) -> dict[str, Any]:
    answer_parts: list[str] = []
    warnings: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    events = await engine.events(user, request_id)
    async with aclosing(events):
        async for event in events:
            if event.type == "delta":
                answer_parts.append(str(event.data.get("delta") or ""))
            elif event.type == "warning":
                warnings.append(dict(event.data))
            if event.type in TERMINAL_EVENTS:
                terminal = {"type": event.type, "data": dict(event.data)}
                break

    if terminal is None:
        return {
            "type": "error",
            "data": {
                "code": "evaluation_stream_incomplete",
                "message": "问答流未返回终态",
                "retryable": True,
            },
            "answer": "".join(answer_parts),
            "warnings": warnings,
            "contexts": [],
        }

    data = terminal["data"]
    terminal["answer"] = str(data.get("answer") or "".join(answer_parts))
    terminal["warnings"] = list(data.get("warnings") or warnings)
    terminal["contexts"] = _contexts_from_sources(data.get("sources"), max_contexts)
    return terminal


def _failed_sample(
    question: str,
    *,
    session_id: str,
    request_id: str = "",
    code: str = "evaluation_internal_error",
    message: str = "评测样本执行失败",
) -> dict[str, Any]:
    return {
        "user_input": question,
        "retrieved_contexts": [],
        "response": "",
        "meta": {
            "original_query": question,
            "session_id": session_id,
            "request_id": request_id,
            "status": "failure",
            "error_code": code,
            "error": message,
            "num_contexts": 0,
            "warnings": [],
        },
    }


async def run_one(
    question: str,
    *,
    engine: QueryEngine,
    user: User,
    session_id: str | None = None,
    max_contexts: int = 10,
) -> dict[str, Any]:
    """执行一条样本；失败也返回结构化结果，不丢弃样本。"""
    session_id = session_id or f"eval-{uuid.uuid4().hex[:12]}"
    request_id = ""
    selected_candidate_id = ""
    try:
        response = await engine.ask(
            user,
            QueryRequest(query=question, session_id=session_id, is_stream=True),
        )
        request_id = response.request_id
        terminal = await _consume_events(
            engine, user, request_id, max_contexts=max_contexts
        )

        if terminal["type"] == "confirmation_required":
            candidates = terminal["data"].get("candidates") or []
            if not candidates:
                return _failed_sample(
                    question,
                    session_id=session_id,
                    request_id=request_id,
                    code="evaluation_candidates_missing",
                    message="问答流程要求确认，但没有返回候选项",
                )
            selected_candidate_id = str(candidates[0]["id"])
            resumed = await engine.confirm(
                user,
                ConfirmRequest(
                    session_id=session_id,
                    pending_request_id=request_id,
                    candidate_id=selected_candidate_id,
                ),
            )
            request_id = resumed.request_id
            terminal = await _consume_events(
                engine, user, request_id, max_contexts=max_contexts
            )

        if terminal["type"] == "error":
            data = terminal["data"]
            return _failed_sample(
                question,
                session_id=session_id,
                request_id=request_id,
                code=str(data.get("code") or "query_failed"),
                message=str(data.get("message") or "问答流程执行失败"),
            )

        contexts = terminal["contexts"]
        answer = terminal["answer"]
        status = "success" if contexts else "empty_evidence"
        return {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "meta": {
                "original_query": question,
                "session_id": session_id,
                "request_id": request_id,
                "status": status,
                "error_code": "",
                "error": "",
                "num_contexts": len(contexts),
                "selected_candidate_id": selected_candidate_id,
                "warnings": terminal["warnings"],
            },
        }
    except QueryEngineError as exc:
        return _failed_sample(
            question,
            session_id=session_id,
            request_id=request_id,
            code=exc.error.code,
            message=exc.error.message,
        )
    except Exception:
        logger.exception("[eval] run_one failed: {}", question)
        return _failed_sample(
            question,
            session_id=session_id,
            request_id=request_id,
        )


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, int | float]:
    counts = {"success": 0, "failure": 0, "empty_evidence": 0}
    for sample in samples:
        status = (sample.get("meta") or {}).get("status")
        if status not in counts:
            status = "failure"
        counts[status] += 1
    total = len(samples)
    return {
        "total": total,
        **counts,
        "success_rate": counts["success"] / total if total else 0.0,
        "failure_rate": counts["failure"] / total if total else 0.0,
        "empty_evidence_rate": counts["empty_evidence"] / total if total else 0.0,
    }


async def build_dataset(
    golden_path: Path,
    out_path: Path,
    *,
    engine: QueryEngine,
    user: User,
    max_contexts: int = 10,
    runner: Callable[..., Awaitable[dict[str, Any]]] = run_one,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output:
        for index, line in enumerate(
            golden_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            gold = json.loads(line)
            question = str(gold["question"])
            conversation_id = str(
                gold.get("conversation_id") or f"sample-{index}"
            )
            session_id = f"eval:{conversation_id}"
            try:
                sample = await runner(
                    question,
                    engine=engine,
                    user=user,
                    session_id=session_id,
                    max_contexts=max_contexts,
                )
            except Exception:
                logger.exception("[eval] sample runner crashed: {}", question)
                sample = _failed_sample(question, session_id=session_id)

            sample["reference"] = gold.get("ground_truth", "")
            sample["meta"]["expected_item_names"] = gold.get("item_names", [])
            sample["meta"]["topic"] = gold.get("topic", "")
            sample["meta"]["turn"] = gold.get("turn")
            rows.append(sample)
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output.flush()

    summary = summarize_samples(rows)
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[eval] wrote {} samples -> {}", len(rows), out_path)
    logger.info("[eval] summary={}", summary)
    return rows, summary


async def _async_main(args) -> None:
    golden_path = resolve_path(args.golden)
    out_path = resolve_path(args.out)
    if not golden_path.exists():
        raise FileNotFoundError(f"金标文件不存在: {golden_path}")

    services = create_application_services_from_env()
    services.initialize_database_only()
    user = services.users.get_by_id(args.user_id)
    if user is None:
        raise SystemExit(f"评测用户不存在: {args.user_id}")

    engine = QueryEngine(services)
    try:
        await build_dataset(
            golden_path,
            out_path,
            engine=engine,
            user=user,
            max_contexts=args.max_contexts,
        )
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run user-scoped query pipeline")
    parser.add_argument("--user-id", required=True, help="使用该用户的冻结问答配置")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-contexts", type=int, default=10)
    asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
