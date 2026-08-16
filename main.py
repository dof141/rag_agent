import subprocess
import os
import sys
from pathlib import Path
from app.api.server import RAGServerManager

def start_frontend_dev():
    """在后台子进程中启动 Vue3 开发服务器 (npm run dev)"""
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        print("🚀 正在拉起 Vue 3 前端热重载开发服务器...")
        # Windows 环境下使用 shell=True 拉起 npm
        subprocess.Popen("npm run dev", cwd=str(frontend_dir), shell=True)

if __name__ == "__main__":
    # 如果处于开发模式，自动开启前端 npm run dev
    # start_frontend_dev()

    # 启动 8000 后端单端口服务
    server = RAGServerManager(host="127.0.0.1", port=8000)
    server.run()