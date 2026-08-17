import os
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 引入项目现有的导入服务逻辑与查询服务逻辑
from app.application_services import create_application_services_from_env
from app.auth.dependencies import build_current_user_dependency
from app.auth.router import create_auth_router
from app.import_process.api.file_import_service import create_import_router
from app.query_process.api.router import create_query_router
from app.runtime_settings.router import create_settings_router
from app.utils.task_utils import clean_interrupted_tasks_on_startup
from app.clients.kb_admin_service import list_kb_items, get_kb_chunks, delete_kb_item, get_kb_stats, delete_single_chunk, update_single_chunk

class UpdateChunkPayload(BaseModel):
    content: str
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

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, services=None):
        self.host = host
        self.port = port
        self.services = services or create_application_services_from_env()

        @asynccontextmanager
        async def lifespan(app_instance: FastAPI):
            self.services.initialize()
            clean_interrupted_tasks_on_startup()
            yield

        self.app = FastAPI(
            title="RAG Agent Unified System",
            description="整合智能问答、文档导入、向量库管理与历史记录的统一服务",
            version="0.2.0",
            lifespan=lifespan,
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
        # 2. 认证、设置与文档导入 API
        # -----------------------------
        app.include_router(create_auth_router(self.services.users, self.services.tokens))
        current_user = build_current_user_dependency(self.services.users, self.services.tokens)
        app.include_router(create_settings_router(self.services.settings, current_user))
        app.include_router(create_import_router(self.services))

        # -----------------------------
        # 3. 问答检索与断点确认 API (原 8002 服务)
        # -----------------------------
        app.include_router(create_query_router(self.services))

        # -----------------------------
        # 4. 向量知识库与切片管理 API [新增]
        # -----------------------------
        @app.get("/api/kb/items", summary="获取所有设备主体")
        async def get_items(keyword: str = ""):
            return {"code": 200, "data": list_kb_items(keyword)}

        @app.get("/api/kb/chunks", summary="获取指定设备的切片")
        async def get_chunks(item_name: str, keyword: str = ""):
            return {"code": 200, "data": get_kb_chunks(item_name, keyword)}

        @app.delete("/api/kb/chunks/{chunk_id}", summary="单条切片物理删除")
        async def delete_chunk(chunk_id: str):
            return delete_single_chunk(chunk_id)

        @app.put("/api/kb/chunks/{chunk_id}", summary="编辑更新单条切片文本并重新向量化")
        async def update_chunk(chunk_id: str, payload: UpdateChunkPayload):
            return update_single_chunk(chunk_id, payload.content)

        @app.delete("/api/kb/items/{item_name}", summary="物理删除设备与切片")
        async def delete_item(item_name: str):
            return delete_kb_item(item_name)

        @app.get("/api/kb/stats", summary="获取系统与存储统计")
        async def get_stats():
            return {"code": 200, "data": get_kb_stats()}

    def _setup_static_frontend(self):
        """挂载打包后的 Vue 3 统一前端静态产物 (frontend/dist)，支持 SPA 路由兜底"""
        dist_path = PROJECT_ROOT / "frontend" / "dist"
        resolved_dist_path = dist_path.resolve()
        index_file = (resolved_dist_path / "index.html").resolve()
        if (
            not index_file.is_relative_to(resolved_dist_path)
            or not index_file.is_file()
        ):
            raise FileNotFoundError(
                f"前端构建产物不存在: {index_file}。请先运行 frontend 的 npm run build。"
            )

        logger.info(f"成功托管 Vue 3 统一前端，路径为：{resolved_dist_path}")
        assets_path = resolved_dist_path / "assets"
        if assets_path.exists():
            resolved_assets_path = assets_path.resolve()
            if not resolved_assets_path.is_relative_to(resolved_dist_path):
                raise RuntimeError(
                    f"前端 assets 目录越出构建目录，拒绝托管: {resolved_assets_path}"
                )
            if not resolved_assets_path.is_dir():
                raise FileNotFoundError(
                    f"前端 assets 路径存在但不是目录: {resolved_assets_path}"
                )
            self.app.mount(
                "/assets",
                StaticFiles(directory=str(resolved_assets_path)),
                name="static_assets",
            )

        index_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        api_path_roots = (
            "api",
            "query",
            "upload",
            "status",
            "stream",
            "history",
            "health",
            "mcp",
            "docs",
        )

        @self.app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa_frontend(full_path: str):
            # 排除所有后端业务 API 请求，避免强行返回 HTML
            if full_path == "openapi.json" or any(
                full_path == root or full_path.startswith(f"{root}/")
                for root in api_path_roots
            ):
                raise HTTPException(status_code=404, detail="API Not Found")

            # 若请求的是具体静态文件（如 favicon.ico），直接返回
            try:
                target_file = (resolved_dist_path / full_path).resolve()
            except (OSError, RuntimeError, ValueError):
                raise HTTPException(
                    status_code=404, detail="Static file not found"
                ) from None
            if not target_file.is_relative_to(resolved_dist_path):
                raise HTTPException(status_code=404, detail="Static file not found")
            if target_file.is_file():
                if target_file == index_file:
                    return FileResponse(target_file, headers=index_headers)
                return FileResponse(target_file)

            # Vue 3 SPA 单页应用路由兜底：一律返回 index.html
            return FileResponse(index_file, headers=index_headers)

    def run(self):
        """启动统一的高性能 Uvicorn 服务"""
        logger.info(f"🚀 RAG Agent 统一服务端正在启动: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


# 文件入口测试
if __name__ == "__main__":
    server = RAGServerManager(host="127.0.0.1", port=8000)
    server.run()
