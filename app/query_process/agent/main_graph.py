from langgraph.constants import END
from langgraph.graph import StateGraph

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import (
    create_item_name_confirm_node,
)
from app.query_process.agent.nodes.node_rerank import create_rerank_node
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import (
    create_search_embedding_node,
)
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    create_search_embedding_hyde_node,
)
from app.query_process.agent.nodes.node_wait_item_confirmation import (
    node_wait_item_confirmation,
)
from app.query_process.agent.state import QueryGraphState
from app.query_process.runtime import QueryRuntime


SEARCH_NODES = (
    "node_search_embedding",
    "node_search_embedding_hyde",
)


def route_after_recognition(state: QueryGraphState):
    if state["awaiting_confirmation"]:
        return "node_wait_item_confirmation"
    if state.get("answer"):
        return "node_answer_output"
    return SEARCH_NODES


def build_query_graph(runtime: QueryRuntime, checkpointer=None):
    builder = StateGraph(QueryGraphState)
    builder.add_node("node_answer_output", node_answer_output)
    builder.add_node(
        "node_item_name_confirm",
        create_item_name_confirm_node(runtime.retrieval),
    )
    builder.add_node("node_rerank", create_rerank_node(runtime.retrieval))
    builder.add_node("node_rrf", node_rrf)
    builder.add_node(
        "node_search_embedding",
        create_search_embedding_node(runtime.retrieval),
    )
    builder.add_node(
        "node_search_embedding_hyde",
        create_search_embedding_hyde_node(runtime.retrieval),
    )
    builder.add_node("node_wait_item_confirmation", node_wait_item_confirmation)
    builder.set_entry_point("node_item_name_confirm")

    builder.add_conditional_edges(
        "node_item_name_confirm",
        route_after_recognition,
        {
            "node_answer_output": "node_answer_output",
            "node_search_embedding": "node_search_embedding",
            "node_search_embedding_hyde": "node_search_embedding_hyde",
            "node_wait_item_confirmation": "node_wait_item_confirmation",
        },
    )
    for node_name in SEARCH_NODES:
        builder.add_edge("node_wait_item_confirmation", node_name)
    builder.add_edge(list(SEARCH_NODES), "node_rrf")
    builder.add_edge("node_rrf", "node_rerank")
    builder.add_edge("node_rerank", "node_answer_output")
    builder.add_edge("node_answer_output", END)

    return builder.compile(checkpointer=checkpointer)
