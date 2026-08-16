from langgraph.graph import END, StateGraph

from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_generate_embeddings import (
    create_generate_embeddings_node,
)
from app.import_process.agent.nodes.node_import_vector_store import create_vector_import_node
from app.import_process.agent.nodes.node_item_name_recognition import (
    node_item_name_recognition,
)
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState
from app.import_process.runtime import ImportRuntime


def build_import_graph(runtime: ImportRuntime):
    work_flow = StateGraph(ImportGraphState)
    work_flow.add_node("node_entry", node_entry)
    work_flow.add_node("node_pdf_to_md", node_pdf_to_md)
    work_flow.add_node("node_md_img", node_md_img)
    work_flow.add_node("node_document_split", node_document_split)
    work_flow.add_node("node_item_name_recognition", node_item_name_recognition)
    work_flow.add_node("node_generate_embeddings", create_generate_embeddings_node(runtime.embedding))
    work_flow.add_node("node_import_vector_store", create_vector_import_node(runtime.vector_store))

    work_flow.set_entry_point("node_entry")

    def route_node(state: ImportGraphState) -> str:
        if state["is_md_read_enabled"]:
            return "node_md_img"
        if state["is_pdf_read_enabled"]:
            return "node_pdf_to_md"
        return END

    work_flow.add_conditional_edges(
        "node_entry",
        route_node,
        {
            "node_md_img": "node_md_img",
            "node_pdf_to_md": "node_pdf_to_md",
            END: END,
        },
    )
    work_flow.add_edge("node_pdf_to_md", "node_md_img")
    work_flow.add_edge("node_md_img", "node_document_split")
    work_flow.add_edge("node_document_split", "node_item_name_recognition")
    work_flow.add_edge("node_item_name_recognition", "node_generate_embeddings")
    work_flow.add_edge("node_generate_embeddings", "node_import_vector_store")
    work_flow.add_edge("node_import_vector_store", END)
    return work_flow.compile()
