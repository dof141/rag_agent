import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task

"""
1.进行参数的校验 校验chunks 与 file_title 是否存在 
2.提取前top chunks 对chunks 拼接上下文
3.发给大模型总结得到 item_name
4.修改state 中chunks 》item_name
"""
# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500
ITEM_NAME_TIMEOUT_SECONDS = lm_config.item_name_timeout


def step_1_get_chunks(state: ImportGraphState):
    #检查校验chunks 与 file_title 是否存在
    chunks = state["chunks"]
    file_title = state["file_title"]
    if not chunks:
        raise ValueError(f"chunks没有值，无法继续进行,抛出异常处理")
    if not file_title:
        #尝试获取file_title
        file_title = os.path.basename(state.get("md_path"))
        logger.info(f"file_title,获取md_path 进行截取！{file_title}")
        state["file_title"] = file_title
    return chunks,file_title



def step_2_build_context(chunks):
    """
    构建上下文
    :param chunks:
    :return:
    """
    parts = [] #传入各个chunks
    total_chars = 0 #记录chunks 个数
    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_title = chunk["title"]
        chunk_content = chunk["content"]
        data = f"切片{index},标题{chunk_title},内容:{chunk_content}"
        parts.append(data)
        total_chars += len(chunk_content)
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数：{total_chars},停止拼接")
            break
    #
    context = "\n\n".join(parts)
    finally_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]
    return finally_context




def step_3_call_llm(context, file_title):
    """
    将拼接好的上下文传个大模型
    :param context:
    :param file_title:
    :return:
    """
    fallback_name = Path(file_title).stem
    try:
        prompt = _load_prompt(
            "item_name_recognition",
            file_title=file_title,
            context=context,
        )
        system_prompt = _load_prompt("product_recognition_system")
        llm = get_llm_client(
            json_mode=False,
            timeout=ITEM_NAME_TIMEOUT_SECONDS,
        )
        messages = [
            HumanMessage(content=prompt),
            SystemMessage(content=system_prompt),
        ]
        item_name = llm.invoke(messages).content
        return item_name.strip() if item_name and item_name.strip() else fallback_name
    except Exception as e:
        logger.warning(f"主题识别失败，使用文件名作为主题：{fallback_name}，原因：{e}")
        return fallback_name


def _load_prompt(name: str, **kwargs) -> str:
    try:
        from app.core.load_prompt import load_prompt

        return load_prompt(name, **kwargs)
    except Exception:
        if name == "item_name_recognition":
            return f"请根据文件名 {kwargs.get('file_title', '')} 和内容识别主题：{kwargs.get('context', '')}"
        return "你是文档主题识别助手。"


def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    更新chunks中内容
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    state["item_name"] = item_name
    for chunk in chunks:
        chunk['item_name'] = item_name
    state["chunks"] = chunks
    logger.info("完成了state 和 chunks 中的[item_name]的赋值和修改")

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """

    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)


    try:
        # 校验并获取到对应值
        chunks, file_title = step_1_get_chunks(state)
        # 取前top 个chunk 拼接chunks 上下文 得到 content 返回 拼接后的上下文
        context = step_2_build_context(chunks)
        # 将拼接后的上下文发送给大模型 进行总结得到item_name
        item_name = step_3_call_llm(context, file_title)
        # 将state['chunks'] 加入 item_name
        step_4_update_chunks_and_state(state, item_name, chunks)
    except Exception as e:
        # 加上 exc_info=True，把具体的错误堆栈打出来！
        logger.error(f"[{func_name}] 节点出现异常: {str(e)}", exc_info=True)
        raise
    finally:
        # 结束当前节点信息，用于任务监控和日志溯源
        logger.info(f">>> 结束执行核心节点：【文档切分】{func_name}")

    add_done_task(state["task_id"], func_name)
    return state



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
