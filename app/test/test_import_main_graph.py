def main():
    import json

    from app.core.logger import logger
    from app.import_process.agent.main_graph import kb_work_app
    from app.import_process.agent.state import create_default_state

    logger.info("===== 开始测试 =====")

    initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    final_state = None

    # 只输出最终的状态值（字典形式），不包含节点名称、执行日志或元数据
    for event in kb_work_app.stream(initial_state):
        for key, value in event.items():
            logger.info(f"节点: {key}")
            final_state = value

    logger.info(
        f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}"
    )
    logger.info("图结构:")
    # 需要先安装图结构打印依赖：uv add grandalf
    kb_work_app.get_graph().print_ascii()
    logger.info("===== 测试结束 =====")


if __name__ == "__main__":
    main()
