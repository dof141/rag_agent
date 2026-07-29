import os
import sys

from pathlib import Path
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

MINERU_EXTS = (
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
    ".html"
)

def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    设计的state: local_file_path [ is_read_md_enabled is_read_pdf_enabled ] md_path pdf_path file_title
    未来要实现:
       1.进入节点的日志输出 【节点 + 核心参数】
         记录任务状态 【哪个任务开始了】 -》 给前端推送信息 （埋点）
       2. 参数校验 （local_file_path -> 没有传入文件 -> end  / local_dir -> 没有传入输出文件夹 -> 创建一个临时）
       3. 解析文件类型，修改state对应的参数 local_file_path -> md | pdf
          -> is_md_read_enabled True  ||   is_pdf_read_enabled True
          -> md_path = local_file_path | pdf_path = local_file_path
          -> file_tile = 读取文件名
       4.结束节点的日志输出 【节点 + 核心参数】
         记录任务状态 【哪个任务结束了】 -》 给前端推送信息 （埋点）
    """
    # 1. 进入节点的日志输出 【节点 + 核心参数】 记录任务状态（给前端推送信息）
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], function_name)

    #2.拿到文件上传内容判断不为空
    local_file_path = state["local_file_path"]
    if not local_file_path:
        logger.error(f"当前没有上传文件")
        return state
    #3.对文件类型进行判断 pdf 还是md 文件，或者是其他暂时不受支持文件
    if local_file_path.endswith(".md"):
        #为md文件
        logger.info(f"【{function_name}】文件类型校验通过：{local_file_path} → MD格式，开启MD解析流程")
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
    elif local_file_path.lower().endswith(MINERU_EXTS):
        #为pdf 文件
        logger.info(f"【{function_name}】文件类型校验通过：{local_file_path} → MINERU_EXTS 格式，开启 多模态文件解析流程")
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
    else:
        #为暂时不受支持的文件
        logger.info(f"【{function_name}】文件类型校验通过：{local_file_path} → 暂时无法解析")
        return state
    #4.传入全局文件状态
    file_name = os.path.basename(local_file_path).split(".")[0]
    state["file_title"] = file_name
    logger.info(f"【{function_name}】文件业务标识提取完成：file_title = {state['file_title']}")

    # 结束：记录节点运行状态
    add_done_task(state["task_id"], function_name)
    #结束状态
    logger.info(f">>> [{function_name}]结束了！现在的状态为：{state}")

    return state
