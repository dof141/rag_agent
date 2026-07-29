import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.clients.mongo_history_utils import mongo_upsert_task, mongo_get_task
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
    set_task_result,
    get_task_result,
    get_node_durations,
    get_total_duration,
    clean_interrupted_tasks_on_startup,
    clear_task,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_work_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """服务生命周期管理：启动时自动清理被打断的历史任务状态"""
    try:
        count = clean_interrupted_tasks_on_startup()
        if count > 0:
            logger.info(f"【服务开机自愈】成功将 {count} 个被打断的历史任务重置为 failed 状态")
    except Exception as e:
        logger.error(f"【服务开机自愈】清洗任务状态失败：{e}")
    yield


# 初始化FastAPI应用实例
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus/KG入库)",
    lifespan=lifespan
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    """
    导入前端页面
    :return:
    """
    html_abs_path = PROJECT_ROOT/"app"/"import_process"/"page"/"import.html"
    # 日志记录页面访问的文件路径，方便排查文件不存在问题
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

# 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
 )
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    初始化图流程  更新节点状态 流式输出  更新节点完成
    :return:
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        # 初始化流程
        init_state = get_default_state()
        #初始化参数
        init_state['task_id'] = task_id
        init_state['local_dir'] = local_dir
        init_state['local_file_path'] = local_file_path
        #流式执行
        for event in kb_work_app.stream(init_state):
            for node_name,node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")
    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        set_task_result(task_id, "error", str(e))
        logger.exception(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)


@app.post("/upload",summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_file(background_tasks: BackgroundTasks,files: list[UploadFile] = File(...)):
    #将上传文件传入到目标目录
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")
        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        #4.创建当前运行目录
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        #创建
        os.makedirs(task_local_dir, exist_ok=True)
        #创建文件上传保存目录
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        # 写入文件
        with open(local_file_abs_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 持久化文件与任务元数据至 MongoDB
        mongo_upsert_task(task_id, {
            "file_name": file.filename,
            "local_dir": str(task_local_dir),
            "local_file_path": str(local_file_abs_path),
            "status": "processing",
        })

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(run_graph_task, task_id, str(task_local_dir), str(local_file_abs_path))
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }


@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态轮询窗口
    前端轮询此窗口，获取任务的实时处理进度
    :param task_id:
    :return:
    """
    #构建任务状态返回体
    task_status_info:Dict[str,Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "node_durations": get_node_durations(task_id),
        "total_duration": get_total_duration(task_id),
        "error": get_task_result(task_id, "error"),
    }
    #记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id} 任务状态查询，当前状态：{task_status_info['status']},已完成]"
    )
    return task_status_info


@app.post("/retry/{task_id}", summary="重试/重新运行任务接口", description="针对打断或失败的任务，从第一个节点重新从头运行全流程")
async def retry_task(task_id: str, background_tasks: BackgroundTasks):
    doc = mongo_get_task(task_id)
    if not doc:
        raise HTTPException(status_code=404, detail="任务记录不存在，无法重试")

    local_file_path = doc.get("local_file_path")
    local_dir = doc.get("local_dir")

    if not local_file_path or not os.path.exists(local_file_path):
        raise HTTPException(status_code=400, detail="本地源文件不存在，请重新上传文件进行导入")

    # 清理内存与数据库的旧异常和完成节点状态
    clear_task(task_id)
    add_running_task(task_id, "upload_file")
    add_done_task(task_id, "upload_file")

    # 重新触发 LangGraph 全流程异步任务（从第一个节点重新跑）
    background_tasks.add_task(run_graph_task, task_id, str(local_dir), str(local_file_path))
    logger.info(f"[{task_id}] 已成功触发【从头重试】，LangGraph全流程重新开始执行")
    return {
        "code": 200,
        "message": "Task retry started successfully",
        "task_id": task_id
    }


# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8001  # 服务端口
    )