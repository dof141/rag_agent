import os
import subprocess
from pathlib import Path

from app.api.server import RAGServerManager


def build_frontend():
    """构建后端即将托管的 Vue 3 前端静态产物。"""
    frontend_dir = Path(__file__).resolve().parent / "frontend"
    if not frontend_dir.is_dir():
        raise FileNotFoundError(f"前端目录不存在，无法启动服务: {frontend_dir}")

    package_json = frontend_dir / "package.json"
    if not package_json.is_file():
        raise FileNotFoundError(f"前端 package.json 不存在，无法启动服务: {package_json}")

    npm_executable = "npm.cmd" if os.name == "nt" else "npm"
    try:
        subprocess.run([npm_executable, "run", "build"], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"前端构建失败（目录: {frontend_dir}，命令: {npm_executable} run build，"
            f"退出码: {exc.returncode}），后端未启动。"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"无法执行前端构建（目录: {frontend_dir}，"
            f"命令: {npm_executable} run build），后端未启动: {exc}"
        ) from exc


if __name__ == "__main__":
    build_frontend()
    server = RAGServerManager(host="127.0.0.1", port=8000)
    server.run()
