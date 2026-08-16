from functools import wraps

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
from app.import_process.errors import ImportTaskError
from app.import_process.runtime import ImportRuntime
from app.import_process.task_repository import TaskRepositoryError
from app.utils.task_utils import update_task_status


def ensure_task_persistence(state) -> None:
    update_task_status(state["task_id"], "processing")


def wrap_import_node(
    node,
    *,
    stage: str,
    public_message: str,
    before_node=None,
):
    @wraps(node)
    def wrapped(state):
        try:
            if before_node is not None:
                before_node(state)
            return node(state)
        except ImportTaskError:
            raise
        except TaskRepositoryError as exc:
            raise ImportTaskError("task_persistence", "任务状态持久化失败") from exc
        except Exception as exc:
            raise ImportTaskError(stage, public_message) from exc

    return wrapped


def build_import_graph(runtime: ImportRuntime):
    work_flow = StateGraph(ImportGraphState)
    work_flow.add_node(
        "node_entry",
        wrap_import_node(
            node_entry,
            stage="file_validation",
            public_message="文件检查失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_pdf_to_md",
        wrap_import_node(
            node_pdf_to_md,
            stage="document_parse",
            public_message="文档解析失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_md_img",
        wrap_import_node(
            node_md_img,
            stage="image_processing",
            public_message="文档图片处理失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_document_split",
        wrap_import_node(
            node_document_split,
            stage="document_split",
            public_message="文档切分失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_item_name_recognition",
        wrap_import_node(
            node_item_name_recognition,
            stage="item_name",
            public_message="文档主体识别失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_generate_embeddings",
        wrap_import_node(
            create_generate_embeddings_node(runtime.embedding),
            stage="embedding",
            public_message="文档向量生成失败",
            before_node=ensure_task_persistence,
        ),
    )
    work_flow.add_node(
        "node_import_vector_store",
        wrap_import_node(
            create_vector_import_node(runtime.vector_store),
            stage="vector_store",
            public_message="向量库写入失败",
            before_node=ensure_task_persistence,
        ),
    )

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
