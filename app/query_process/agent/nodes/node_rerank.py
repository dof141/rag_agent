# =====================================================================
# 🚨 必须在所有第三方库 import 之前设置环境变量，防止 C++ 底层多线程死锁崩溃
# =====================================================================

import os

# 必须放在 run_pipeline.py 的最顶行！
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import sys

from app.retrieval import get_retrieval
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

"""
负责对 RRF 算法后的召回结果与网络搜索结果进行非同源打分排序：
1. 整合本地召回与 Web 搜索结果，统一字段并标源，同时限制单条文本最大长度（防 Tokenizer 溢出）
2. 使用 Rerank 模型按 [query, doc] 格式进行 Batch 分批打分
3. 使用防断崖算法动态截取 TopK，提高答案精确性
"""

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
RERANK_MAX_TOPK: int = 10  # 动态 TopK 硬上限：最多取前 N 条
RERANK_MIN_TOPK: int = 1  # 最小 TopK：至少保留前 N 条
RERANK_GAP_RATIO: float = 0.25  # 相对断崖阈值（比率）
RERANK_GAP_ABS: float = 0.5  # 绝对断崖阈值（分值）
MAX_TEXT_CHAR_LEN: int = 800  # 单篇文档参与打分的最大字符数（防止 C++ 内存溢出）
RERANK_BATCH_SIZE: int = 4  # 分批打分 Batch Size（提升稳定性）


def step_1_merge_rrf_mcp(state: QueryGraphState) -> list[dict]:
    """
    对本地召回与 Web 搜索两个集合进行整合，并进行安全文本截断
    """
    rrf_chunks = state.get("rrf_chunks") or []
    web_search_docs = state.get("web_search_docs") or []

    doc_list = []

    # 1. 处理本地 RRF 召回结果
    for chunk in rrf_chunks:
        entity = chunk.get('entity', {}) if isinstance(chunk, dict) else {}
        chunk_id = entity.get('chunk_id', '') or chunk.get('id', '')
        content = entity.get('content', '') or chunk.get('content', '') or ''
        file_title = entity.get('file_title', '') or chunk.get('file_title', '') or entity.get('title', '') or chunk.get('title', '')
        parent_title = entity.get('parent_title', '') or chunk.get('parent_title', '')
        item_name = entity.get('item_name', '') or chunk.get('item_name', '')

        # 安全截断，防止超长网页/表格文本导致 Tokenizer C++ 越界
        truncated_text = content[:MAX_TEXT_CHAR_LEN].strip()
        if truncated_text:
            doc_list.append({
                'chunk_id': chunk_id,
                'text': truncated_text,
                'content': content,
                'file_title': file_title,
                'parent_title': parent_title,
                'item_name': item_name,
                'title': file_title,
                'source': "local",
                'url': ''
            })

    if web_search_docs:
        # 2. 处理 Web 联网搜索结果
        for chunk in web_search_docs:
            snippet = chunk.get('snippet', '') or ''
            url = chunk.get('url', '') or ''
            title = chunk.get('title', '') or ''

            truncated_text = snippet[:MAX_TEXT_CHAR_LEN].strip()
            if truncated_text:
                doc_list.append({
                    'chunk_id': '',
                    'text': truncated_text,
                    'content': snippet,
                    'file_title': title or '网络检索资料',
                    'parent_title': url or '网页来源',
                    'item_name': '网络搜素',
                    'title': title,
                    'source': "web",
                    "url": url
                })

    logger.info(f"多路数据融合完成，共计 {len(doc_list)} 条文档")
    return doc_list


def step_2_rerank_doc_list(doc_list: list[dict], state: QueryGraphState) -> list[dict]:
    """
    将问题与整合后的 chunks 发给 Rerank 模型进行 Batch 分批打分并降序排序
    """
    if not doc_list:
        return []

    query = state.get('rewritten_query') or state.get("original_query") or ""
    if not query:
        logger.warning("Rerank 接收到的查询 Query 为空！")
        return doc_list

    try:
        doc_list_with_score = get_retrieval().rerank_documents(query, doc_list)
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return doc_list

    logger.info(f"排序完成！Top1 得分: {doc_list_with_score[0]['score'] if doc_list_with_score else 0}")
    return doc_list_with_score


def step_3_topk(doc_list_with_score: list[dict]) -> list[dict]:
    """
    进行防断崖处理：当后一条数据相比前一条数据分数急剧下降（产生断崖）时，截断后面的不相关数据。
    """
    if not doc_list_with_score:
        return []

    total_len = len(doc_list_with_score)
    max_topk = RERANK_MAX_TOPK
    min_topk = RERANK_MIN_TOPK
    gap_abs = RERANK_GAP_ABS
    gap_ratio = RERANK_GAP_RATIO

    # 实际最多可考察的元素数量
    topk = min(max_topk, total_len)

    if topk > min_topk:
        # 遍历范围：从 min_topk - 1 到 min(topk - 1, total_len - 1)
        # 注意：必须比较 index 与 index + 1，防止越界 upperBound 为 total_len - 1
        for index in range(min_topk - 1, min(topk - 1, total_len - 1)):
            score_1 = doc_list_with_score[index]['score']
            score_2 = doc_list_with_score[index + 1]['score']  # ✅ 已修复：对比前后相邻的两个分数

            gap = score_1 - score_2
            rel = gap / (abs(score_1) + 1e-6)

            # 判断是否触发断崖
            if gap > gap_abs or rel > gap_ratio:
                logger.info(
                    f"在位置 index={index} (得分:{score_1:.4f}) 和 index={index + 1} (得分:{score_2:.4f}) 之间检测到断崖，触发截断！")
                topk = index + 1
                break

    doc_topk_list = doc_list_with_score[:topk]
    logger.info(f"最终截取 TopK 长度: {len(doc_topk_list)}")
    return doc_topk_list


def node_rerank(state: QueryGraphState) -> dict:
    """
    LangGraph 节点函数：使用 Cross-Encoder 对多路召回文档精确打分并截断
    """
    logger.info("---Rerank处理开始---")
    req_id = state.get("request_id", "")
    is_stream = state.get("is_stream", False)

    if req_id:
        add_running_task(req_id, sys._getframe().f_code.co_name, is_stream)

    try:
        # 1. 多路召回文档整合
        doc_list = step_1_merge_rrf_mcp(state)

        # 2. Rerank 模型计算得分与排序
        doc_list_with_score = step_2_rerank_doc_list(doc_list, state)

        # 3. 防断崖截取 TopK
        topk_docs = step_3_topk(doc_list_with_score)

    except Exception as e:
        logger.exception(f"node_rerank 执行过程抛出异常: {e}")
        topk_docs = []

    if req_id:
        add_done_task(req_id, sys._getframe().f_code.co_name, is_stream)

    # 返回 LangGraph 更新字典（保持状态同步）
    state['reranked_docs'] = topk_docs
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 模拟测试数据
    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"}},
        {"entity": {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"}},
        {"entity": {"chunk_id": "local_3", "content": "无关的测试文档内容，今天天气很好" * 50, "title": "测试文档"}}
        # 测试超长文本
    ]

    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "苹果手机最新的系统更新发布了"}
    ]

    mock_state = {
        "request_id": "test_req_001",
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        res_state = node_rerank(mock_state)
        reranked = res_state.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:30]}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试失败: {e}")
