import time
from typing import Dict, List, Any
from .sse_utils import push_to_session
from app.clients.mongo_history_utils import (
    mongo_upsert_task,
    mongo_get_task,
    mongo_clean_interrupted_tasks,
)

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# 任务总体耗时追踪
_tasks_start_time: Dict[str, float] = {}
_tasks_end_time: Dict[str, float] = {}

# 节点耗时追踪: task_id -> { node_name: timestamp / duration }
_tasks_node_start_time: Dict[str, Dict[str, float]] = {}
_tasks_node_durations: Dict[str, Dict[str, float]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "node_item_name_confirm": "确认学习主题",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}
    if task_id not in _tasks_node_start_time:
        _tasks_node_start_time[task_id] = {}
    if task_id not in _tasks_node_durations:
        _tasks_node_durations[task_id] = {}
    if task_id not in _tasks_start_time:
        _tasks_start_time[task_id] = time.time()


def _sync_to_mongo(task_id: str) -> None:
    """内部工具：将内存中的最新任务状态同步落盘到 MongoDB"""
    task_data = {
        "status": _tasks_status.get(task_id, ""),
        "done_list": _tasks_done_list.get(task_id, []),
        "running_list": _tasks_running_list.get(task_id, []),
        "node_durations": _tasks_node_durations.get(task_id, {}),
        "total_duration": get_total_duration(task_id),
        "error": _tasks_result.get(task_id, {}).get("error", ""),
    }
    mongo_upsert_task(task_id, task_data)


def _load_from_mongo_if_needed(task_id: str) -> None:
    """内部工具：若内存中无此 task_id 记录（如服务刚重启），则从 MongoDB 恢复缓存"""
    if task_id not in _tasks_status:
        doc = mongo_get_task(task_id)
        if doc:
            _ensure_task(task_id)
            _tasks_status[task_id] = doc.get("status", "")
            _tasks_done_list[task_id] = doc.get("done_list", [])
            _tasks_running_list[task_id] = doc.get("running_list", [])
            _tasks_node_durations[task_id] = doc.get("node_durations", {})
            _tasks_result[task_id] = {"error": doc.get("error", "")}
            total_dur = doc.get("total_duration", 0.0)
            now = time.time()
            _tasks_start_time[task_id] = now - total_dur
            _tasks_end_time[task_id] = now


def clean_interrupted_tasks_on_startup() -> int:
    """服务启动时调用：清理并修正所有在上一次运行中打断的任务状态"""
    return mongo_clean_interrupted_tasks()


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if node_name not in running:
        running.append(node_name)

    # 记录节点启动时间
    now = time.time()
    if node_name not in _tasks_node_start_time[task_id]:
        _tasks_node_start_time[task_id][node_name] = now

    _sync_to_mongo(task_id)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    # 3) 计算该节点的精确耗时 (秒)
    now = time.time()
    start_t = _tasks_node_start_time[task_id].get(node_name, now)
    duration = round(now - start_t, 2)
    _tasks_node_durations[task_id][node_name] = duration

    _sync_to_mongo(task_id)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value
    _sync_to_mongo(task_id)


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    _load_from_mongo_if_needed(task_id)
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    _load_from_mongo_if_needed(task_id)
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。
    """
    _load_from_mongo_if_needed(task_id)
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。
    """
    _load_from_mongo_if_needed(task_id)
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [_to_cn(n) for n in running]


def get_node_durations(task_id: str) -> Dict[str, float]:
    """
    获取任务各节点的执行耗时字典（格式：{node_name: duration_seconds}）
    """
    _load_from_mongo_if_needed(task_id)
    _ensure_task(task_id)
    return _tasks_node_durations.get(task_id, {})


def get_total_duration(task_id: str) -> float:
    """
    获取图任务的总消耗时长（已完成则为固定耗时，处理中则为当前动态累计耗时）
    """
    _load_from_mongo_if_needed(task_id)
    _ensure_task(task_id)
    start_t = _tasks_start_time.get(task_id, time.time())
    end_t = _tasks_end_time.get(task_id, time.time())
    return round(end_t - start_t, 2)


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """
    _tasks_status[task_id] = status_name
    if status_name in (TASK_STATUS_COMPLETED, TASK_STATUS_FAILED):
        _tasks_end_time[task_id] = time.time()
    _sync_to_mongo(task_id)
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)