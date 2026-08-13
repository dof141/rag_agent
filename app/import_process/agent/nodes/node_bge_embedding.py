import os
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"], function_name)
    try:
        #对chunks 进行向量化
        #校验chunks 是否存在
        chunks = state["chunks"]
        if not chunks or not isinstance(chunks, list):
            logger(f"数据校验失败，{chunks}不存在")
            raise ValueError(f"数据校验失败，{chunks}不存在")
        #拼接 content 确保向量化的数据信息完整 将item_name 至于前部
        finally_chunks = [] #存储处理完后的chunk
        batch_size=4 #每次向量化数据
        for i in range(0, len(chunks), batch_size):
            batch_chunk = chunks[i:i+batch_size]
            current_texts =[]
            for chunk in batch_chunk:
                item_name =chunk.get("item_name")
                content = chunk.get("content")
                current_texts.append(f"商品:{item_name}，内容介绍:{content}")

            result = generate_embeddings(current_texts)
            for index, chunk in enumerate(batch_chunk):
             #使用向量模型进行向量化，将向量化后的数据 拼接回chunks 中
                chunk_item = chunk.copy()
                chunk_item['dense_vector'] = result['dense'][index]
                chunk_item['sparse_vector'] = result['sparse'][index]
                finally_chunks.append(chunk_item)
        state["chunks"] = finally_chunks
        logger.info(f"--- BGE-M3 向量化 处理完成，共处理{len(finally_chunks)} 条数据")
    except Exception as e:
        logger.error(f"[{function_name}] 使用 node_bge_embedding 解析出现异常 异常信息：{e}", exc_info=True)
        raise
    finally:
        # . 进入的日志和任务状态的配置
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为：{state}")

    add_done_task(state["task_id"], function_name)
    return state
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"向量维度：{result_chunks}")

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")
