#流程图
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.query_process.agent.nodes.node_wait_item_confirmation import node_wait_item_confirmation
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

builder= StateGraph(QueryGraphState)
#注册节点
builder.add_node("node_answer_output",node_answer_output) #结束边
builder.add_node("node_item_name_confirm",node_item_name_confirm) #意图识别
builder.add_node("node_rerank",node_rerank) #精排
builder.add_node("node_rrf",node_rrf) #粗排序
builder.add_node("node_search_embedding",node_search_embedding) #向量搜索结果
builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde) #hyde 搜索
builder.add_node("node_web_search_mcp",node_web_search_mcp) #网络搜索
builder.add_node("node_wait_item_confirmation",node_wait_item_confirmation)
#起始节点
builder.set_entry_point("node_item_name_confirm")
#定义条件路由
# def route_node(state:QueryGraphState):
#     answer = state["answer"]
#     if answer:
#         return "node_answer_output"
#     return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"

SEARCH_NODES = (
    "node_search_embedding",
    "node_search_embedding_hyde",
    "node_web_search_mcp",
)

def route_after_recognition(state:QueryGraphState):
    if state["awaiting_confirmation"]:
        return "node_wait_item_confirmation"
    if state.get("answer"):
        return "node_answer_output"
    return SEARCH_NODES

#定义条件边
builder.add_conditional_edges("node_item_name_confirm",route_after_recognition,{
                                  "node_answer_output":"node_answer_output",
                                  "node_search_embedding":"node_search_embedding",
                                  "node_search_embedding_hyde":"node_search_embedding_hyde",
                                  "node_web_search_mcp":"node_web_search_mcp",
                                  "node_wait_item_confirmation":"node_wait_item_confirmation"
                              })
#定义边
# builder.add_edge("node_search_embedding","node_rrf")
# builder.add_edge("node_search_embedding_hyde","node_rrf")
# builder.add_edge("node_web_search_mcp","node_rrf")
for node_name in SEARCH_NODES:
    builder.add_edge("node_wait_item_confirmation",node_name)
builder.add_edge(list(SEARCH_NODES),"node_rrf")
builder.add_edge("node_rrf","node_rerank")
builder.add_edge("node_rerank","node_answer_output")
builder.add_edge("node_answer_output",END)

#构建流
query_app = builder.compile(checkpointer=InMemorySaver())