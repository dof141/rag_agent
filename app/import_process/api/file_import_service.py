import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.auth.dependencies import build_current_user_dependency
from app.embedding.interface import EmbeddingConfigurationError
from app.import_process.agent.state import get_default_state
from app.import_process.errors import ImportTaskError
from app.import_process.task_repository import TaskRepositoryError
from app.runtime_settings.service import RuntimeSettingsConfigurationError
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    clean_interrupted_tasks_on_startup,
    clear_task,
    get_done_task_list,
    get_node_durations,
    get_running_task_list,
    get_task_result,
    get_task_status,
    get_total_duration,
    restore_task_from_snapshot,
    set_task_result,
    update_task_status,
)
from app.vector_store.document_id import build_document_id
from app.vector_store.interface import VectorStoreConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_import_router(services) -> APIRouter:
    router = APIRouter()
    current_user = build_current_user_dependency(services.users, services.tokens)

    @router.post("/upload", summary="文件上传接口")
    async def upload_file(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        user=Depends(current_user),
    ):
        try:
            snapshot = services.settings.get_snapshot(user.id)
            runtimes = [services.runtime_factory(snapshot) for _file in files]
        except (
            RuntimeSettingsConfigurationError,
            VectorStoreConfigurationError,
            EmbeddingConfigurationError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        date_based_root_dir = Path(services.output_root) / datetime.now().strftime("%Y%m%d")
        prepared_tasks = []
        for file, runtime in zip(files, runtimes):
            filename = Path(file.filename or "upload.bin").name
            task_id = str(uuid.uuid4())
            document_id = build_document_id(user.id, filename)

            task_local_dir = date_based_root_dir / task_id
            task_local_dir.mkdir(parents=True, exist_ok=True)
            local_file_abs_path = task_local_dir / filename
            with local_file_abs_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)

            try:
                services.task_repository.upsert(
                    task_id,
                    {
                        "file_name": filename,
                        "local_dir": str(task_local_dir),
                        "local_file_path": str(local_file_abs_path),
                        "status": "pending",
                        "user_id": user.id,
                        "document_id": document_id,
                        "settings_version": snapshot.version,
                        "done_list": ["upload_file"],
                        "running_list": [],
                    },
                )
            except Exception as exc:
                shutil.rmtree(task_local_dir, ignore_errors=True)
                for prepared in prepared_tasks:
                    try:
                        services.task_repository.upsert(
                            prepared["task_id"],
                            {
                                "status": "failed",
                                "error": "任务创建失败",
                                "failed_stage": "task_persistence",
                            },
                        )
                    except Exception:
                        pass
                    shutil.rmtree(prepared["task_local_dir"], ignore_errors=True)
                raise HTTPException(status_code=503, detail="任务状态服务不可用") from exc

            prepared_tasks.append(
                {
                    "task_id": task_id,
                    "task_local_dir": task_local_dir,
                    "local_file_abs_path": local_file_abs_path,
                    "document_id": document_id,
                    "runtime": runtime,
                }
            )

        for prepared in prepared_tasks:
            task_id = prepared["task_id"]
            update_task_status(task_id, "processing", persist=False)
            add_running_task(task_id, "upload_file", persist=False)
            add_done_task(task_id, "upload_file", persist=False)
            background_tasks.add_task(
                run_graph_task,
                task_id,
                str(prepared["task_local_dir"]),
                str(prepared["local_file_abs_path"]),
                user.id,
                prepared["document_id"],
                prepared["runtime"],
                task_repository=services.task_repository,
            )
        return {
            "code": 200,
            "message": f"Files uploaded successfully, total: {len(files)}",
            "task_ids": [prepared["task_id"] for prepared in prepared_tasks],
        }

    @router.get("/status/{task_id}")
    async def get_task_progress(task_id: str, user=Depends(current_user)):
        try:
            task = services.task_repository.get(task_id)
        except TaskRepositoryError as exc:
            raise HTTPException(status_code=503, detail="任务状态服务不可用") from exc
        if task is None or task.get("user_id") != user.id:
            raise HTTPException(status_code=404, detail="任务不存在")
        restore_task_from_snapshot(task_id, task)
        return build_task_status_response(task_id)

    @router.post("/retry/{task_id}")
    async def retry_task(
        task_id: str,
        background_tasks: BackgroundTasks,
        user=Depends(current_user),
    ):
        try:
            task = services.task_repository.get(task_id)
        except TaskRepositoryError as exc:
            raise HTTPException(status_code=503, detail="任务状态服务不可用") from exc
        if task is None or task.get("user_id") != user.id:
            raise HTTPException(status_code=404, detail="任务不存在")
        local_file_path = task.get("local_file_path")
        local_dir = task.get("local_dir")
        if not local_file_path or not os.path.exists(local_file_path):
            raise HTTPException(status_code=400, detail="本地源文件不存在，请重新上传文件进行导入")
        snapshot = services.settings.get_snapshot(user.id)
        runtime = services.runtime_factory(snapshot)
        try:
            services.task_repository.upsert(
                task_id,
                {
                    "status": "pending",
                    "error": "",
                    "failed_stage": "",
                    "settings_version": snapshot.version,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="任务状态服务不可用") from exc
        clear_task(task_id)
        update_task_status(task_id, "processing", persist=False)
        add_running_task(task_id, "upload_file", persist=False)
        add_done_task(task_id, "upload_file", persist=False)
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(local_dir),
            str(local_file_path),
            user.id,
            task["document_id"],
            runtime,
            task_repository=services.task_repository,
        )
        return {"code": 200, "message": "Task retry started successfully", "task_id": task_id}

    return router


def run_graph_task(
    task_id: str,
    local_dir: str,
    local_file_path: str,
    user_id: str,
    document_id: str,
    runtime,
    *,
    task_repository,
    graph_builder=None,
):
    try:
        if graph_builder is None:
            from app.import_process.agent.main_graph import build_import_graph

            graph_builder = build_import_graph
        update_task_status(task_id, "processing")
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["user_id"] = user_id
        init_state["document_id"] = document_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        for event in graph_builder(runtime).stream(init_state):
            for node_name, _node_result in event.items():
                add_done_task(task_id, node_name)
        update_task_status(task_id, "completed", persist=False)
        task_repository.upsert(task_id, {"status": "completed", "failed_stage": ""})
    except ImportTaskError as exc:
        record_task_failure(
            task_id,
            error=exc.public_message,
            failed_stage=exc.stage,
            task_repository=task_repository,
        )
    except TaskRepositoryError:
        record_task_failure(
            task_id,
            error="任务状态持久化失败",
            failed_stage="task_persistence",
            task_repository=task_repository,
        )
    except Exception:
        record_task_failure(
            task_id,
            error="文档导入失败",
            failed_stage="unknown",
            task_repository=task_repository,
        )


def record_task_failure(task_id: str, *, error: str, failed_stage: str, task_repository) -> None:
    update_task_status(task_id, "failed", persist=False)
    set_task_result(task_id, "error", error, persist=False)
    set_task_result(task_id, "failed_stage", failed_stage, persist=False)
    try:
        task_repository.upsert(
            task_id,
            {"status": "failed", "error": error, "failed_stage": failed_stage},
        )
    except Exception:
        pass


def build_task_status_response(task_id: str) -> dict:
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "node_durations": get_node_durations(task_id),
        "total_duration": get_total_duration(task_id),
        "error": get_task_result(task_id, "error"),
        "failed_stage": get_task_result(task_id, "failed_stage"),
    }


async def get_import_page():
    html_abs_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not html_abs_path.exists():
        raise HTTPException(status_code=404, detail="import.html page not found")
    return FileResponse(path=html_abs_path, media_type="text/html")


app = FastAPI(title="File Import Service")
app.include_router(APIRouter())


@app.on_event("startup")
async def _startup():
    clean_interrupted_tasks_on_startup()


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8001)
