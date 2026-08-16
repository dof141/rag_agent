def main():
    import json

    from app.core.logger import logger
    from app.query_process.agent.main_graph import query_app
    from app.query_process.agent.state import create_query_default_state

    logger.info("===== 开始测试 =====")

    initial_state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
    )
    final_state = None
    config = {"configurable": {"thread_id": "111"}}

    # 只输出最终的状态值（字典形式），不包含节点名称、执行日志或元数据
    for event in query_app.stream(initial_state, config):
        for key, value in event.items():
            logger.info(f"节点: {key} , 输出结果：{value}")
            final_state = value

    logger.info(
        f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}"
    )
    logger.info("图结构:")
    # 需要先安装图结构打印依赖：uv add grandalf
    query_app.get_graph().print_ascii()
    logger.info("===== 测试结束 =====")


if __name__ == "__main__":
    main()
