import re
import sys
from typing import List, Set
from urllib.parse import quote, unquote, urlparse

from app.clients.mongo_history_utils_new import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task, set_task_result, get_node_durations, get_total_duration

MAX_CONTEXT_CHARS = 12000


def step_1_check_answer(state: QueryGraphState) -> bool:
    """
    判断是否存在前置节点生成的最终 answer
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    if answer:
        if is_stream:
            push_to_session(state["request_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["request_id"], "answer", answer)
        return True
    return False


def step_2_load_prompt(state: QueryGraphState) -> str:
    """
    装载与格式化 Prompt 模板
    """
    reranked_docs = state.get("reranked_docs", [])
    query = state.get('rewritten_query') or state.get('original_query') or ""
    history = state.get('history', [])
    item_names = state.get('item_names', [])

    docs = []
    used_length = 0

    # 1. 构建 Context，兼容各种文档格式
    for index, doc in enumerate(reranked_docs):
        entity = doc.get("entity", {}) if isinstance(doc, dict) else {}
        text = entity.get("content") or doc.get("content") or doc.get("text") or ""
        source = doc.get("source") or entity.get("source") or "knowledge_base"
        title = doc.get("title") or entity.get("file_title") or ""
        score = doc.get("score") or doc.get("distance") or 0.0

        content = f"[{index}][source={source}][title={title}][score={score}]\n\n{text}"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info("Context 长度达到上限，停止追加文档切片！")
            break
        docs.append(content)
        used_length += len(content)

    final_context = "\n\n".join(docs)

    # 2. 构建对话历史
    history_str = ""
    logger.info(f"本次获取到的历史聊天记录共 {len(history)} 条")
    if history:
        for message in history:
            role = message.get("role")
            text = message.get("text", "")
            if role == "user" and text:
                current_history = f"【用户】: {text}\n"
            elif role == "assistant" and text:
                current_history = f"【助手】: {text}\n"
            else:
                continue

            if used_length + len(current_history) > MAX_CONTEXT_CHARS:
                logger.info("Context 包含历史记录后达到上限！")
                break
            history_str += current_history
            used_length += len(current_history)
    else:
        history_str = "没有历史对话记录！"

    # 3. 处理 item_names
    item_names_str = ",".join(item_names) if isinstance(item_names, list) else str(item_names)

    # 4. 加载并充填 Prompt
    answer_out_prompt = load_prompt(
        "answer_out",
        context=final_context,
        history=history_str,
        item_names=item_names_str,
        question=query
    )
    logger.info(f"已经完成提示词生成，总长度: {len(answer_out_prompt)}")
    return answer_out_prompt


def step_3_create_answer(state: QueryGraphState, prompt: str) -> str:
    """
    将提示词发送给大模型，获取完整回答
    """
    llm = get_llm_client()
    is_stream = state.get("is_stream", False)
    answer = ""

    if is_stream:
        for chunk in llm.stream(prompt):
            delta = chunk.content or ""
            answer += delta
            # 向 SSE 持续推送 Token
            push_to_session(state["request_id"], SSEEvent.DELTA, {"delta": delta})
    else:
        response = llm.invoke(prompt)
        answer = response.content or ""

    state['answer'] = answer
    logger.info(f"LLM 模型原始返回长度：{len(answer)}")
    return answer


def step_4_extract_images_url(state: QueryGraphState) -> List[str]:
    """
    【终极无死角提取】专治带空格、中文、内部小括号的 MinIO 图片 URL
    """
    images: List[str] = []
    set_images: Set[str] = set()

    # 1. 获取所有的切片文档
    reranked_docs = state.get("reranked_docs", []) or state.get("chunks", [])

    for doc in reranked_docs:
        # 收集 doc 里所有的文本字符串
        text_snippets = []

        def extract_strings(d):
            if isinstance(d, dict):
                for v in d.values():
                    extract_strings(v)
            elif isinstance(d, list):
                for item in d:
                    extract_strings(item)
            elif isinstance(d, str):
                text_snippets.append(d)

        extract_strings(doc)
        combined_text = "\n".join(text_snippets)

        if not combined_text.strip():
            continue

        # 2. 第一重提取：直接抓取以 http/https 开头且以 .jpg/.png 结尾的文本串（贪婪匹配到最后一个图片扩展名）
        # 能够无视路径中的空格和中间小括号
        matches = re.findall(r'(https?://[^\r\n<>"\'\]]+?\.(?:png|jpg|jpeg|gif|webp))', combined_text, re.IGNORECASE)

        # 3. 如果第一重没拿全，使用 Markdown 模式匹配
        if not matches:
            # 匹配 ![alt](url) 中的 url，解决内部含括号问题
            raw_markdown_urls = re.findall(r'!\[.*?\]\((http[^\r\n]+)\)', combined_text)
            for m in raw_markdown_urls:
                # 如果结尾多带了 Markdown 的右括号，剥离多余的右括号直到遇到 .jpg 等后缀
                cleaned_m = re.sub(r'(\.(?:png|jpg|jpeg|gif|webp)).*$', r'\1', m, flags=re.IGNORECASE)
                matches.append(cleaned_m)

        # 4. 清理并进行标准 URL 编码（转义空格为 %20，转义中文）
        for raw_url in matches:
            raw_url = raw_url.strip()

            # 清理末尾可能的杂质字符
            raw_url = re.sub(r'[\)\s>"]+$', '', raw_url)

            # 确保是以图片后缀结尾
            if not re.search(r'\.(?:png|jpg|jpeg|gif|webp)$', raw_url, re.IGNORECASE):
                continue

            # 转义 URL 中的空格和中文（解决前端 404/加载失败）
            parsed = urlparse(raw_url)
            safe_path = quote(unquote(parsed.path), safe='/')
            safe_url = f"{parsed.scheme}://{parsed.netloc}{safe_path}"

            if safe_url not in set_images:
                images.append(safe_url)
                set_images.add(safe_url)

    logger.info(f"【终极硬提取成功】从 chunks 中共提取到 {len(images)} 张真实图片：{images}")
    state['image_urls'] = images
    return images


def step_4_5_extract_sources(state: QueryGraphState) -> list:
    """
    提取重排后的高相关性知识库切片引用源数据 (Recall Sources)
    包含：文档名称 file_title、章节标题 parent_title、产品/笔记分类 item_name、匹配得分 score、切片原文 content
    """
    sources = []
    reranked_docs = state.get("reranked_docs", []) or state.get("chunks", [])
    
    for doc in reranked_docs[:5]:
        entity = doc.get("entity", {}) if isinstance(doc, dict) else {}
        
        file_title = doc.get("file_title") or entity.get("file_title") or doc.get("title") or entity.get("title") or "个人学习笔记"
        parent_title = doc.get("parent_title") or entity.get("parent_title") or ""
        item_name = doc.get("item_name") or entity.get("item_name") or "学习资料"
        content = doc.get("content") or doc.get("text") or entity.get("content") or ""
        
        score_val = doc.get("score") if doc.get("score") is not None else doc.get("distance", 0.92)
        try:
            score = round(float(score_val), 4)
        except (ValueError, TypeError):
            score = 0.92

        chunk_id = doc.get("chunk_id") or doc.get("id") or entity.get("chunk_id") or f"c-{len(sources)}"

        sources.append({
            "chunk_id": str(chunk_id),
            "file_title": str(file_title),
            "parent_title": str(parent_title),
            "item_name": str(item_name),
            "score": score,
            "content": str(content)
        })

    state["sources"] = sources
    logger.info(f"【知识库引用源提取成功】共提取到 {len(sources)} 条真实的 Sources 数据: {sources}")
    return sources


STANDARD_COMPLETED_NODES = [
    {"node_id": "node_item_name_confirm", "name": "确认学习主题", "status": "completed"},
    {"node_id": "node_search_embedding", "name": "切片搜索", "status": "completed"},
    {"node_id": "node_search_embedding_hyde", "name": "切片搜索(假设性文档)", "status": "completed"},
    {"node_id": "node_web_search_mcp", "name": "网络搜索", "status": "completed"},
    {"node_id": "node_rrf", "name": "倒排融合", "status": "completed"},
    {"node_id": "node_rerank", "name": "重排序", "status": "completed"},
    {"node_id": "node_answer_output", "name": "生成答案", "status": "completed"}
]


def step_5_write_history(state: QueryGraphState):
    """
    持久化聊天记录到 MongoDB
    """
    answer = state.get("answer", "")
    session_id = state.get("session_id", "")
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    image_urls = state.get("image_urls", [])
    sources = state.get("sources", [])
    node_steps = state.get("node_steps", STANDARD_COMPLETED_NODES)
    total_duration = state.get("total_duration", 0.0)

    if answer and session_id:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=rewritten_query,
            image_urls=image_urls,
            sources=sources,
            node_steps=node_steps,
            total_duration=total_duration
        )
        logger.info(f"完成了本次对话的 MongoDB 存储 (包含 node_steps 节点历史与总耗时 {total_duration}s)！")


def node_answer_output(state: QueryGraphState) -> dict:
    """
    节点入口：由代码 100% 掌控图片提取与拼接
    """
    logger.info("---node_answer_output 节点处理开始---")
    request_id = state["request_id"]
    is_stream = state.get("is_stream", False)

    add_running_task(request_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 检查是否存在前置 answer
    answer_exists = step_1_check_answer(state)

    if not answer_exists:
        # 2. 大模型仅生成纯文本回答
        prompt = step_2_load_prompt(state)
        text_answer = step_3_create_answer(state, prompt)

        # 3. 代码硬核提取原始切片里的图片
        real_images = step_4_extract_images_url(state)

        # 4. 提取知识库引用来源 (Recall Sources)
        sources = step_4_5_extract_sources(state)

        # 4. 智能判断大模型是否已在段落中按语义嵌入图片
        if real_images:
            missing_images = []
            for img_url in real_images:
                parsed_path = urlparse(img_url).path
                unquoted_path = unquote(parsed_path)
                if (img_url not in text_answer and 
                    parsed_path not in text_answer and 
                    unquoted_path not in text_answer):
                    missing_images.append(img_url)
            
            if missing_images:
                img_block = "\n\n" + "\n\n".join([f"![参考图示]({url})" for url in missing_images])
                final_answer = text_answer + img_block
            else:
                final_answer = text_answer
        else:
            final_answer = text_answer

        state["answer"] = final_answer

        # 标记当前节点完成以准确计算包含当前节点的执行耗时
        add_done_task(request_id, sys._getframe().f_code.co_name, is_stream)

        # 提取动态节点耗时与总耗时
        node_durations = get_node_durations(request_id)
        total_duration = get_total_duration(request_id)

        dynamic_completed_nodes = []
        for node in STANDARD_COMPLETED_NODES:
            node_id = node["node_id"]
            node_copy = dict(node)
            if node_id in node_durations:
                node_copy["duration"] = node_durations[node_id]
            dynamic_completed_nodes.append(node_copy)

        state["node_steps"] = dynamic_completed_nodes
        state["total_duration"] = total_duration

        # 5. 给前端推送 SSE FINAL 信号
        if is_stream:
            push_to_session(
                request_id,
                SSEEvent.FINAL,
                {
                    "answer": final_answer,
                    "status": "completed",
                    "image_urls": real_images or [],
                    "sources": sources or [],
                    "node_steps": dynamic_completed_nodes,
                    "total_duration": total_duration
                }
            )
        else:
            set_task_result(request_id, "answer", final_answer)

    # 6. 保存聊天记录
    step_5_write_history(state)

    logger.info("---node_answer_output 节点处理结束---")

    return {
        "answer": state.get("answer", ""),
        "image_urls": state.get("image_urls", []),
        "sources": state.get("sources", []),
        "node_steps": state.get("node_steps", STANDARD_COMPLETED_NODES),
        "total_duration": state.get("total_duration", 0.0)
    }