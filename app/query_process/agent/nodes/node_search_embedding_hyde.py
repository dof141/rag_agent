import time
import sys

from langchain_core.messages import HumanMessage

from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.retrieval import get_retrieval
from app.utils.task_utils import  add_done_task,add_running_task
from app.core.logger import logger


def step_1_create_hyde_doc(rewritten_query):
    """
    调用大模型生成假设性文档
    :param rewritten_query:
    :return:
    """
    #提示词
    prompt = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=prompt)
    ]
    #调用大模型
    llm_client = get_llm_client()
    hyde_doc = llm_client.invoke(messages).content
    logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
    logger.info(f"Step 1: 文档预览: {hyde_doc[:50]}...")

    return hyde_doc

def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),  # 调整默认权重以偏向稠密向量 (0.8, 0.2)
    norm_score: bool = True,    # 默认开启归一化
    output_fields=["chunk_id", "content", "item_name"],
):
    """
    阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

    :param rewritten_query: 改写后的查询
    :param hyde_doc: Step 1 生成的假设性文档
    :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
    :param req_limit: Milvus 搜索时的候选召回数量
    :param top_k: 最终返回的 Top K 结果数量
    :param ranker_weights: 混合检索权重 (Dense, Sparse)
    :param norm_score: 是否对分数进行归一化
    :param output_fields: 返回结果中包含的字段
    :return: 检索结果列表
    """
    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")
    if not hyde_doc:
        raise ValueError("hypothetical_doc 不能为空")

    logger.info(f"Step 2: Query + HyDE Doc 总长度: {len(rewritten_query + ' ' + hyde_doc)}")
    chunks = get_retrieval().search_chunks_with_hyde(
        query=rewritten_query,
        hyde_doc=hyde_doc,
        item_names=item_names,
        top_k=top_k,
    )
    logger.info(f"假设性问题检索结果: {chunks}")
    return chunks



def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    logger.info("---HyDE 开始处理---")
    add_running_task(state["request_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    #由大模型生成假设性答案
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    if not rewritten_query:
        logger.info("node_search_embedding_hyde 错误 rewritten_query 没有值")
    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    #将假设文档和用户问题进行拼接 再到向量数据库中进行查找
    hyde_embedding_chunks =  step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )
    # ...
    add_done_task(state["request_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---HyDE 处理结束---")
    return {"hyde_embedding_chunks":hyde_embedding_chunks}



"""
chunks内容：[{'id': 467602001275654313, 'distance': 0.8355008363723755, 
'entity': {'item_name': 'hak180', 'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467602001275654313, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n'}}, 
{'id': 467602001275654334, 'distance': 0.8355008363723755,
 'entity': {'item_name': 'hak180', 'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467602001275654334, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n'}}, {'id': 467643443758521548, 'distance': 0.8355008363723755, 'entity': {'item_name': 'hak180', 'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521548, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n'}}, {'id': 467643443758521568, 'distance': 0.8355008363723755, 'entity': {'item_name': 'hak180', 'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521568, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n'}}, {'id': 467643443758521641, 'distance': 0.8355008363723755, 'entity': {'item_name': 'hak180', 'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521641, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n'}}]

"""
if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["hak180"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")

