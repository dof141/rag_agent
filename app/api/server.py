import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# 引入项目现有的导入服务逻辑与查询服务逻辑
from app.import_process.api.file_import_service import upload_file, get_task_progress
from app.query_process.api.query_server import (
    query, stream, get_task_history, clear_chat_history, confirm,
    QueryRequest, ConfirmRequest, mcp
)
from app.clients.kb_admin_service import list_kb_items, get_kb_chunks, delete_kb_item, get_kb_stats
from app.clients.mongo_history_utils_new import get_all_sessions_summary, delete_session, clear_history
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger

class RAGServerManager:
    """
    RAG Agent 统一应用服务端管理器 (Unified Single-Port Server)
    功能：
    1. 整合文件导入服务 (8001) 与 查询检索服务 (8002) 降维合并至单端口 (8000)
    2. 扩充向量知识库管理 (Milvus/MinIO) 与 历史会话管理 (MongoDB) 接口
    3. 自动托管打包后的 Vue 3 统一前端 SPA (frontend/dist)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="RAG Agent Unified System",
            description="整合智能问答、文档导入、向量库管理与历史记录的统一服务",
            version="0.2.0"
        )
        self._setup_middleware()
        self._setup_routes()
        self._setup_static_frontend()

    def _setup_middleware(self):
        """配置跨域中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """注册全量业务与管理 API"""
        app = self.app

        # -----------------------------
        # 1. 基础健康检查
        # -----------------------------
        @app.get("/health", summary="服务健康状态")
        async def health_check():
            return {"status": "online", "code": 200}

        # -----------------------------
        # 2. 文档导入 API (原 8001 服务)
        # -----------------------------
        app.add_api_route("/upload", upload_file, methods=["POST"], summary="上传文件并触发导入图")
        app.add_api_route("/status/{task_id}", get_task_progress, methods=["GET"], summary="任务进度轮询")

        # -----------------------------
        # 3. 问答检索与断点确认 API (原 8002 服务)
        # -----------------------------
        app.add_api_route("/query", query, methods=["POST"], summary="问答提问")
        app.add_api_route("/stream/{request_id}", stream, methods=["GET"], summary="SSE 流式推送到前端")
        app.add_api_route("/query/confirm", confirm, methods=["POST"], summary="人工确认歧义实体")

        # -----------------------------
        # 4. 向量知识库与切片管理 API [新增]
        # -----------------------------
        @app.get("/api/kb/items", summary="获取所有设备主体")
        async def get_items(keyword: str = ""):
            return {"code": 200, "data": list_kb_items(keyword)}

        @app.get("/api/kb/chunks", summary="获取指定设备的切片")
        async def get_chunks(item_name: str, keyword: str = ""):
            return {"code": 200, "data": get_kb_chunks(item_name, keyword)}

        @app.delete("/api/kb/items/{item_name}", summary="物理删除设备与切片")
        async def delete_item(item_name: str):
            return delete_kb_item(item_name)

        @app.get("/api/kb/stats", summary="获取系统与存储统计")
        async def get_stats():
            return {"code": 200, "data": get_kb_stats()}

        # -----------------------------
        # 5. 历史会话管理 API [新增与增强]
        # -----------------------------
        @app.get("/api/history/sessions", summary="聚合查询所有历史 Session 概要")
        async def get_sessions():
            return {"code": 200, "data": get_all_sessions_summary()}

        @app.get("/history/{session_id}", summary="查询指定会话历史记录")
        async def get_session_history_by_id(session_id: str, limit: int = 20):
            return get_task_history(session_id, limit)

        @app.delete("/history/{session_id}", summary="删除指定会话")
        async def delete_session_by_id(session_id: str):
            count = delete_session(session_id)
            return {"code": 200, "message": "Session deleted", "deleted_count": count}

        @app.delete("/api/history/sessions", summary="清空全量历史会话")
        async def clear_all_history():
            count = clear_history("")
            return {"code": 200, "message": "All history cleared", "deleted_count": count}

        # -----------------------------
        # 6. FastMCP SSE 挂载
        # -----------------------------
        app.mount("/mcp", mcp.sse_app())

    def _setup_static_frontend(self):
        """挂载打包后的 Vue 3 统一前端静态产物 (frontend/dist)"""
        dist_path = PROJECT_ROOT / "frontend" / "dist"
        if dist_path.exists():
            logger.info(f"成功托管 Vue 3 统一前端，路径为：{dist_path}")
            self.app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")
        else:
            logger.warning(f"前端构建产物未找到 ({dist_path})，请先在 frontend 目录执行 npm run build")

    def run(self):
        """启动统一的高性能 Uvicorn 服务"""
        logger.info(f"🚀 RAG Agent 统一服务端正在启动: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


# 文件入口测试
if __name__ == "__main__":
    server = RAGServerManager(host="127.0.0.1", port=8000)
    server.run()
