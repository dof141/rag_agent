"""
这个节点负责 当用户意图模糊时，从数据库中进行主体召回时存在多个主体
这个节点将进行图流程暂停，前端用户返回确认主体后 再运行该节点
也就是实现整个流程可打断
"""
from langgraph.types import interrupt

from app.query_process.agent.state import QueryGraphState


def node_wait_item_confirmation(state: QueryGraphState):
    selected_id = interrupt({
        "type": "item_confirmation",
        "message": "请选择您要查询的学习主题/笔记分类",
        "request_id": state["request_id"],
        "candidates": state["candidate_items"],
    })

    selected = next(
        (x for x in state["candidate_items"] if x["id"] == selected_id),
        None,
    )
    if selected is None:
        raise ValueError("选择的主体不在候选列表中")

    return {
        "item_names": [selected["item_name"]],
        "candidate_items": [],
        "awaiting_confirmation": False,
        "answer": None,
    }