from pathlib import Path
import uuid
import uvicorn
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from langgraph.types import Command
from app.query_process.agent.state import QueryGraphState, create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
from app.core.logger import logger

# 初始化 FastAPI 应用实例
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus/KG入库)"
)

# 跨域中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查请求
@app.get("/health")
async def health_check():
    logger.info("请求成功")
    return {"ok": True}

# 获取聊天页面
@app.get("/chat")
async def chat_root():
    chat_root_path = PROJECT_ROOT / "app/query_process/page/chat.html"
    if not chat_root_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_root_path}！")
    return FileResponse(chat_root_path)

class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

# 辅助函数：构造 LangGraph 配置
def graph_config(session_id: str):
    return {"configurable": {"thread_id": session_id}}

# 原生异步运行 Graph 状态机（支持 async ainvoke）
async def run_graph(
    session_id: str,
    request_id: str,
    graph_input,
    is_stream: bool = True,
):
    config = graph_config(session_id)
    update_task_status(request_id, TASK_STATUS_PROCESSING, is_stream)

    try:
        # 使用 LangGraph 原生异步 ainvoke
        result = await query_app.ainvoke(graph_input, config=config)
        interrupts = result.get("__interrupt__") or []

        if interrupts:
            payload = interrupts[0].value
            update_task_status(
                request_id,
                "waiting_confirmation",
                is_stream,
            )
            if is_stream:
                push_to_session(
                    request_id,
                    "confirmation_required",
                    payload,
                )
            return result

        update_task_status(
            request_id,
            TASK_STATUS_COMPLETED,
            is_stream,
        )
        return result

    except Exception as exc:
        logger.exception("执行图失败")
        set_task_result(request_id, "error", str(exc))
        update_task_status(
            request_id,
            TASK_STATUS_FAILED,
            is_stream,
        )

# 问题请求端点
@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    init_state = create_query_default_state(
        session_id=session_id,
        request_id=request_id,
        original_query=request.query,
        is_stream=request.is_stream,
    )
    clear_task(request_id)

    is_stream = request.is_stream
    if is_stream:
        create_sse_queue(request_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"开始处理流程... 是否流式: {is_stream}, 参数: {user_query}, session_id: {session_id}")

    if is_stream:
        # 流式后台任务异步驱动
        background_tasks.add_task(run_graph, session_id, request_id, init_state, is_stream)
        logger.info("开始处理结果....")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id,
            "request_id": request_id,
        }
    else:
        # 非流式调用：await 原生异步图处理
        await run_graph(session_id, request_id, init_state, is_stream)
        
        # 检查快照状态中是否有待确认标志
        config = graph_config(session_id)
        snapshot = await query_app.aget_state(config)
        snapshot_values = snapshot.values if snapshot else {}

        if snapshot_values.get("awaiting_confirmation"):
            optional_items = snapshot_values.get("optional_item_names", [])
            candidate_items = snapshot_values.get("candidate_items", [])
            
            candidates_formatted = []
            if candidate_items:
                for x in candidate_items:
                    if isinstance(x, dict):
                        candidates_formatted.append(x)
                    else:
                        candidates_formatted.append({"id": str(x), "item_name": str(x)})
            elif optional_items:
                for item in optional_items:
                    candidates_formatted.append({"id": str(item), "item_name": str(item), "file_title": f"{item} 手册"})

            return {
                "message": "检测到多个候选设备，等待用户选择确认",
                "session_id": session_id,
                "request_id": request_id,
                "awaiting_confirmation": True,
                "candidates": candidates_formatted,
                "candidate_items": candidates_formatted
            }

        answer = get_task_result(session_id, "answer", "")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "request_id": request_id,
            "answer": answer,
            "done_list": []
        }

# 获取执行过程节点 SSE
@app.get("/stream/{request_id}")
async def stream(request_id: str, request: Request):
    logger.info(f"session_id:{request_id} 客户端，已经和后台建立了长连接")
    return StreamingResponse(
        sse_generator(request_id, request),
        media_type="text/event-stream",
    )

# 获取历史聊天记录
@app.get("/history/{sessionId}")
def get_task_history(sessionId: str, limit: int = 10):
    try:
        records = get_recent_messages(sessionId, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "image_urls": r.get("image_urls", []),
                "sources": r.get("sources", []),
                "node_steps": r.get("node_steps", []),
                "ts": r.get("ts")
            })
        return {"session_id": sessionId, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

# 清空历史聊天
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}

class ConfirmRequest(BaseModel):
    session_id: str
    pending_request_id: str
    candidate_id: str

@app.post("/query/confirm")
async def confirm(req: ConfirmRequest, background_tasks: BackgroundTasks):
    config = graph_config(req.session_id)
    snapshot = await query_app.aget_state(config)

    if not snapshot.values.get("awaiting_confirmation"):
        raise HTTPException(409, "当前会话没有等待确认")

    candidates = snapshot.values.get("candidate_items", [])
    optional_items = snapshot.values.get("optional_item_names", [])

    valid_ids = set()
    for x in candidates:
        if isinstance(x, dict):
            valid_ids.add(str(x.get("id", x.get("item_name"))))
            valid_ids.add(str(x.get("item_name")))
        else:
            valid_ids.add(str(x))
    for opt in optional_items:
        valid_ids.add(str(opt))

    if req.candidate_id not in valid_ids and valid_ids:
        logger.warning(f"候选校验宽泛通行: candidate_id={req.candidate_id}")

    new_request_id = str(uuid.uuid4())
    create_sse_queue(new_request_id)

    command = Command(
        resume=req.candidate_id,
        update={
            "request_id": new_request_id,
            "is_stream": True,
        },
    )

    background_tasks.add_task(
        run_graph,
        req.session_id,
        new_request_id,
        command,
        True
    )
    return {
        "session_id": req.session_id,
        "request_id": new_request_id,
    }

# 引入官方 MCP SSE Server 组件
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LangGraph-RAG-App")

@mcp.tool()
async def query_rag(query: str, session_id: Optional[str] = None) -> str:
    effective_session_id = session_id if (session_id and session_id.strip()) else str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    init_state = create_query_default_state(
        session_id=effective_session_id,
        request_id=request_id,
        original_query=query,
        is_stream=False
    )

    await run_graph(effective_session_id, request_id, init_state, is_stream=False)

    config = graph_config(effective_session_id)
    snapshot = await query_app.aget_state(config)
    snapshot_values = snapshot.values if snapshot else {}

    optional_items = snapshot_values.get("optional_item_names", [])
    candidate_items = snapshot_values.get("candidate_items", [])

    candidates = optional_items or [x.get("item_name") for x in candidate_items if isinstance(x, dict)]

    if candidates:
        items_str = "、".join([f"'{item}'" for item in candidates])
        return (
            f"[Session: {effective_session_id}]\n\n"
            f"检测到您查询的内容存在多个相关匹配项：{items_str}。\n"
            f"请问您具体想了解哪一个？（您可以直接回答其中某一个具体名称）"
        )

    answer = get_task_result(effective_session_id, "answer", "")
    if not answer:
        answer = get_task_result(request_id, "answer", "流程已触发，但暂未返回文本结果。")

    return f"[Session: {effective_session_id}]\n\n{answer}"

app.mount("/mcp", mcp.sse_app())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)