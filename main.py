"""
RAG Agent 项目统一启动入口
直接运行此文件即可在单端口 (如 8000) 启动完整的 RAG 平台（自动托管 Vue 3 统一前端与全量后端 API）
"""
from app.api.server import RAGServerManager

if __name__ == "__main__":
    # 实例化统一服务端管理器类，并在 8000 端口启动
    server = RAGServerManager(host="127.0.0.1", port=8000)
    server.run()
