import sys
import os
import json
import logging

from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())

class GoodsResponse(BaseModel):
    item_names: List[str] = Field(description="课程或学习资料名称列表，若无则返回空列表")
    rewritten_query: str = Field(description="对用户本次提问进行重写后的独立完整问题")
def step_3_llm_item_name_and_rewrite_query(original_query, history_chats)  ->GoodsResponse:
    """
    负责让大模型提取出问题中的课程/资料实体名称并重写 Query
    """
    try:
        contents = ""
        for chat in history_chats:
            item_str = ",".join(chat.get('item_names', [])) if chat.get('item_names') else "无"
            contents += f"角色:{chat.get('role')}, 内容:{chat.get('text')}, 重写问题:{chat.get('rewritten_query', '')}, 关联主体:{item_str};\n"

        prompt = load_prompt("rewritten_query_and_itemnames", history_text=contents, query=original_query)
        messages = [HumanMessage(content=prompt)]

        llm = get_llm_client(json_mode=True)
        parser = PydanticOutputParser(pydantic_object=GoodsResponse)
        chain = llm | parser

        response = chain.invoke(messages)
        logger.info(f"已经完成问题的重写和item_name 的提取，结果为：{response}")
        return response

    except Exception as e:
        logger.error(f"step_3 执行失败: {e}")
        # 降级容错方案：如果 LLM 解析失败，不直接抛异常断开，而是兜底返回原始 Query
        return GoodsResponse(item_names=[], rewritten_query=original_query)


def step_4_vectorize_and_query(item_names:List[str]) ->List[Dict]:
    """
    负责从向量数据库中查询对应匹配的item
    :param item_names:
    :return:
    """
    #1.对大模型的item 进行向量化
    """
     result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
    """
    result=generate_embeddings(item_names)
    #2.拿向量化后的数据到milvus 数据库中进行混合检索
        #拿到数据库连接
    milvus = get_milvus_client()
    if not milvus:
        logger.error("Step 4: 无法连接到 Milvus")
        return milvus
    #遍历
    final_result =[]
    for index,item_name in enumerate(item_names):
        dense = result["dense"][index]
        sparse = result["sparse"][index]
        #混合检索
        #构建 request
        reqs = create_hybrid_search_requests(dense,sparse)
        #定义权重重排
        response = hybrid_search(
            client=milvus,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            norm_score=True #是否返回分数
        )
        logger.info(f"response:{response}")
        #定义返的 结果体
        matches = []
        if response and len(response)>0:
            for hit in response[0]:
                entity = hit["entity"]
                hit_name = entity["item_name"]
                score = hit["distance"]
                if hit_name:
                    matches.append({
                        "item_name": hit_name,
                        "score":score
                    })
        #3.拿到混合检索后的数据 与分数拼接返回
        final_result.append({
            "extracted":item_name,
            "matches":matches
        })
    logger.info(f"向量混合检索完成，最终返回数据与得分：{final_result}")
    return final_result



def step_5_confirmed_and_optional_item_name(item_match):
    high_items = []
    optional_item = []

    for match in item_match:
        extracted_name = match["extracted"]
        matches = match["matches"]

        # 按得分从大到小排序
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        high_matches = [x for x in matches if x.get("score", 0) >= 0.85]
        middle_matches = [x for x in matches if x.get("score", 0) >= 0.6]

        # ----------------- 情况 1：存在明确高分项 -----------------
        if high_matches:
            # 优先精准匹配字面完全相同的商品
            exact_match = next((x for x in high_matches if x.get("item_name") == extracted_name), None)

            if exact_match:
                high_items.append(exact_match.get("item_name"))
            elif len(high_matches) == 1:
                # 只有一个高分项，直接锁定
                high_items.append(high_matches[0].get("item_name"))
            else:
                # 有多个高分项且无法精确完全匹配：此时应视为“模糊/存在多个候选”，交由用户选择！
                for m in high_matches[:3]:
                    optional_item.append(m.get("item_name"))

            # 处理完高分逻辑后，直接跳过当前 item 的后续处理
            continue

        # ----------------- 情况 2：无高分，只有中等分数 -----------------
        if middle_matches:
            for m in middle_matches[:3]:
                optional_item.append(m.get("item_name"))
            continue

        logger.info(f"没有匹配的item_name,忽略：{extracted_name}")

    result = {
        "confirmed_item_names": list(set(high_items)),
        "optional_item_names": list(set(optional_item))
    }
    logger.info(f"最终处理结果为: {result}")
    return result



def step_6_deal_list(state: QueryGraphState, item_results, history_chats, rewritten_query):
    confirm_item = item_results.get("confirmed_item_names", [])
    optional_item = item_results.get("optional_item_names", [])

    # 1. 如果有确定的学习主题 (最高优先级)
    if confirm_item:
        state['item_names'] = confirm_item
        state['history'] = history_chats
        state['rewritten_query'] = rewritten_query
        state['answer'] = None  # 明确置空 answer，继续交给下一个 LangGraph 节点处理
        logger.info(f"有确定的item_name: {confirm_item}")
        return state

    # 2. 没有确定学习主题，但有多个候选提示用户
    if optional_item:
        state['candidate_items'] = [
            {
                "id":name,
                "item_name":name,
            } for name in optional_item
        ]
        state['item_names'] = []
        state['awaiting_confirmation']=True
        state['answer'] = None
        state['rewritten_query'] = rewritten_query
        logger.info(f"有可选的item_name: {optional_item}")
        return state

    # 3. 彻底未搜到相关笔记或学习资料
    state['answer'] = "未在私有笔记库中检索到相关学习资料或笔记主题，请尝试转换关键词或提供更具体的知识点名称。"
    state['item_names'] = []
    logger.info(f"没有检测到匹配的item_name, 当前结果: {item_results}")
    return state

def node_item_name_confirm(state: QueryGraphState,):
    """
    节点功能：确认用户问题中的核心商品名称。
    1.获取历史聊天记录
    2.保存本次当前聊天
    3.调用大模型  重写query 获取到item
    4.对进行向量搜索 对搜索结果进行打分
    5.如果查到并且分数够高 则进行内容召回，如果查到但是分数不高 则进行二次确认，如果没有查到，则直接返回回答没有查看到

    """
    logger.info(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["request_id"], sys._getframe().f_code.co_name,state["is_stream"])
    #获取到session id
    session_id = state["session_id"]
    logger.info(f"当前会话session_id: {session_id}")
    #获取到历史聊天记录
    history_chats = get_recent_messages(session_id,limit=10)
    serializable_history = []
    for item in history_chats:
        item_copy = dict(item)
        if "_id" in item_copy:
            item_copy["_id"] = str(item_copy["_id"])  # 转为 str
        serializable_history.append(item_copy)

    # 更新 LangGraph 状态
    state["history"] = serializable_history#将历史聊天记录保存到节点中，这里其实不太稳定
    #保存本次聊天
    """
     session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
        """

    #3. 使用大模型对问题进行重写
    """
    重写的好处
    1.能够消除指代词歧义，明确需要重新的主体
    2.能够使上下文 更加完整，补全提问的上下文环境
    3.可以去除掉用户问题中的口语化表达，
    4.能够润色问题增加召回率，后续模型查询更加精准 提高整体速度
    """
    extract_res = step_3_llm_item_name_and_rewrite_query(state['original_query'],history_chats)


    item_names = extract_res.item_names
    rewritten_query  = extract_res.rewritten_query
    state['rewritten_query'] = rewritten_query
    item_results={}
    if len(item_names)>0:
        #从milvus 中取出匹配的item 集合 返回格式为 [{大模型item:{match:【{item:score},{item:score}】}},{}]
        item_match = step_4_vectorize_and_query(item_names)
        #从搜索到的向量数据中挑选出低分和和高分返回
        item_results = step_5_confirmed_and_optional_item_name(item_match)
        #将 返回的结果进行区分 确认 执行流程
    state = step_6_deal_list(state,item_results,history_chats,rewritten_query)
    #保存本次提问
    save_chat_message(session_id=session_id,
                      role='user',
                      text=state['original_query'],
                      rewritten_query=state.get('rewritten_query', ""),
                      item_names=state.get('item_names', []),
                      image_urls=state.get('image_urls', [])
                      )
    # 记录任务结束
    add_done_task(state["request_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print(f"---node_item_name_confirm---处理结束")

    return state

if __name__ == '__main__':
    # 测试代码块
    print("\n" + "=" * 50)
    print(">>> 启动 node_item_name_confirm 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "sess-1rwwqp1jv8zmrkku638",
        "original_query": "华为显示屏多少钱？",  # 针对用户提到的具体 case
        "is_stream": False,
        "item_names": [],
        "request_id": "123"
    }

    try:
        # 运行节点
        result = node_item_name_confirm(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Item Names: {result.get('item_names')}")
        print(f"Answer: {result.get('answer')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
