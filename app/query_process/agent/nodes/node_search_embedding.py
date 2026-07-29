import time
import sys

from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger


def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    从向量数据库中 对用户的提问进行向量搜索 并返回搜索到的结果
    """
    logger.info("---量内容检索 开始处理---")
    add_running_task(state["request_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    # 获取到重写后用户提问 和item_name
    rewritten_query = state["rewritten_query"]
    # 获取item
    item_names = state["item_names"]
    # 向量化
    query_dict = generate_embeddings([rewritten_query])

    quoted = ", ".join(f'"{v}"' for v in item_names) if item_names else ""
    # 构造最终过滤表达式（如果 item_names 为空，则全局无过滤检索）
    expr = f"item_name in [{quoted}]" if item_names else None
    # 设置request
    reqs = create_hybrid_search_requests(
        dense_vector=query_dict['dense'][0],
        sparse_vector=query_dict['sparse'][0],
        limit=10,
        expr=expr
    )
    # 搜索client
    milvus_client = get_milvus_client()
    if not milvus_client:
        logger.error("无法连接到 Milvus")
        return milvus_client

    # 定义权重重排
    response = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        limit=5,  # 最终返回的TOP5相似度最高结果
        norm_score=True,  # 是否返回分数
        output_fields=["chunk_id", "content", "item_name","file_title","parent_title"]  # 指定返回的业务字段
    )

    # 拿到返回结果，并且返回chunks
    embedding_chunks = response[0] if response else []
    # ...
    add_done_task(state["request_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---量内容检索 处理结束---")
    return {"embedding_chunks": embedding_chunks}

"""
[
{'id': 467602001275654313, 'distance': 0.8289464116096497, 
    'entity': {'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467602001275654313, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n', 'item_name': 'hak180'}}, 

{'id': 467602001275654334, 'distance': 0.8289464116096497,
    'entity': {'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467602001275654334, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n', 'item_name': 'hak180'}}, {'id': 467643443758521548, 'distance': 0.8289464116096497, 'entity': {'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521548, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n', 'item_name': 'hak180'}}, {'id': 467643443758521568, 'distance': 0.8289464116096497, 'entity': {'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521568, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n', 'item_name': 'hak180'}}, {'id': 467643443758521641, 'distance': 0.8289464116096497, 'entity': {'file_title': 'hak180产品安全手册', 'parent_title': '## 设备', 'chunk_id': 467643443758521641, 'content': '## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n', 'item_name': 'hak180'}}]}

"""

if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "hak180烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["hak180"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
