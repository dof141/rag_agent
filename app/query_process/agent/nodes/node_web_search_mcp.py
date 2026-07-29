import asyncio
import json
import time
import sys

from agents.mcp import MCPServerStreamableHttp

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger

# 本地 open-websearch MCP 服务地址
OPEN_SEARCH_MCP_URL = "http://localhost:3001/mcp"


async def call_search_web_mcp(query: str, count: int):
    """
    调用本地 open-search MCP 外部服务来进行网络搜索
    """
    # 1. 先创建 mcp server 服务连接
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": OPEN_SEARCH_MCP_URL,
            # open-websearch 的 HTTP 端点要求包含 Accept 响应格式
            "headers": {
                "Accept": "application/json, text/event-stream"
            },
            "timeout": 10,
        },
        max_retry_attempts=3
    )
    # 2. 进行网络搜索，关闭连接
    try:
        await search_mcp.connect()
        tools = await search_mcp.list_tools()
        logger.info(f"工具列表：{tools}")

        # open-websearch 的工具名称为 "search"，搜索数量字段为 "limit"
        result = await search_mcp.call_tool(
            tool_name="search",
            arguments={
                "query": query,
                "limit": count,
                # 可选: 指定引擎列表，如 ["sogou", "baidu", "csdn", "juejin"]，默认使用 sogou
                "engines": ["sogou"]
            }
        )
        return result
    except Exception as e:
        logger.error(f"call_search_web_mcp 中调用外部 mcp 出现异常: {e}")
        return None
    finally:
        await search_mcp.cleanup()


async def node_web_search_mcp(state: QueryGraphState):
    """
    节点功能：使用原生 async/await 调用外部搜索引擎补充信息（原生异步节点）
    """
    add_running_task(state["request_id"], sys._getframe().f_code.co_name, state["is_stream"])
    logger.info("---node-web-search-mcp异步处理开始---")

    query = state.get('rewritten_query') or state.get('original_query') or ""
    logger.info(f"调用外部 mcp 引擎，搜索关键词: {query}")

    web_search_docs = []
    try:
        # 直接原生 await 异步函数
        mcp_result = await call_search_web_mcp(query, 5)
        if mcp_result and hasattr(mcp_result, 'content') and mcp_result.content:
            raw_text = mcp_result.content[0].text
            raw_data = json.loads(raw_text)

            # open-websearch 返回的列表字段为 "results" (做向下兼容处理)
            web_search_docs = raw_data.get("results") or raw_data.get("pages") or []
            logger.info(f"外部 mcp 搜索返回结果共 {len(web_search_docs)} 条")
        else:
            logger.warning("外部 mcp 搜索未返回有效数据")
    except Exception as e:
        logger.error(f"node_web_search_mcp 执行异常: {e}")

    add_done_task(state["request_id"], sys._getframe().f_code.co_name, state["is_stream"])
    logger.info("---node-web-search-mcp处理结束---")
    return {"web_search_docs": web_search_docs}


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地异步测试")
    print("=" * 50)

    test_state = {
        "request_id": "test_req_001",
        "session_id": "test_mcp_session",
        "rewritten_query": "如何安装Neo4j",
        "is_stream": False
    }

    try:
        result_state = asyncio.run(node_web_search_mcp(test_state))

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results, indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")