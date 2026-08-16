import time
import sys

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
"""
完成对向量召回结果和 假设回答进行召回结果的 重排序
 重排序 ： 按照分数进行 使用rrf 算法进行排序
"""


def step_3_reciprocal_rank_fusion(score_chunks,top_k=5):
    """
    对两路召回结果进行打分 重培训 使用rrf 算法 并最终取前top5
    先对传入的 集合进行遍历,拿到其中的 chunks 和 权重
    再对chunks 进行遍历，获取到其中的chunks id 和 内容 以及分数， 并且对分数进行rrf 算法计算 得出最终得分
    将计算后的chunks存入到 一个chunks 集合中 并且按照 最终得分进行排序
    对排序后的chunks 进行取 top5  返回最终结果
    :param score_chunks:
    :return:
    """
    #定义一个存储分数的集合
    score_dict = {}
    #定义一个存储chunks 列表的集合
    chunks_dict = {}
    #对传入的集合进行遍历
    for chunks,weight in score_chunks:
        #进行遍历得到两个不同的
        for index,chunk in enumerate(chunks,start=1):
            #获取到 到chunk_id
            chunk_id = chunk['id'] or chunk.get('entity').get('chunk_id')
            #对分数进行计算 并保存 以chunk_id 保存
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + weight / (60 + index)
            #保存chunks
            chunks_dict.setdefault(chunk_id,chunk)
    #对整理好的chunks进行融合重排
    merge_chunks = []
    for chunk_id,score in score_dict.items():
        #获取到chunks
        chunk = chunks_dict.get(chunk_id)
        #将分数和chunk加入到融合集合
        merge_chunks.append((chunk,score))
    #对merge_chunks 进行排序
    merge_chunks.sort(key=lambda x: x[1], reverse=True)
    #获取指定top 的chunk
    merge_chunks = merge_chunks[:top_k]
    rank_chunks = [chunk for chunk,score in merge_chunks]

    #返回
    logger.info(f"重排序后的chunks 为{rank_chunks}")
    return rank_chunks




def node_rrf(state:QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    logger.info("---RRF---")
    add_running_task(state["request_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #进行重排序
    #拿到 向量搜索结果，和假设性回答搜索结果
    embedding_chunks = state["embedding_chunks"]
    hyde_embedding_chunks = state["hyde_embedding_chunks"]
    #放入一个集合中
    score_chunks = [
        (embedding_chunks,1.0),
        (hyde_embedding_chunks,1.0),
    ]
    #使用rrf 算法进行数据排序
    rrf_chunks = step_3_reciprocal_rank_fusion(score_chunks,5)
    #更新state
    state["rrf_chunks"] = rrf_chunks
    add_done_task(state['request_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["hak180"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
