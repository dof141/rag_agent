from app.import_process.errors import ImportTaskError
from app.vector_store.interface import VectorChunk, VectorDocument, VectorStoreError


def create_vector_import_node(vector_store):
    def node_import_vector_store(state):
        try:
            document = VectorDocument(
                user_id=state["user_id"],
                document_id=state["document_id"],
                file_title=state["file_title"],
                item_name=state["item_name"],
                item_dense_vector=tuple(state["item_dense_vector"]),
                item_sparse_vector=state.get("item_sparse_vector"),
                chunks=tuple(
                    VectorChunk(
                        index=index,
                        content=chunk["content"],
                        title=chunk.get("title", ""),
                        parent_title=chunk.get("parent_title", state["file_title"]),
                        part=int(chunk.get("part", index + 1)),
                        dense_vector=tuple(chunk["dense_vector"]),
                        sparse_vector=chunk.get("sparse_vector"),
                    )
                    for index, chunk in enumerate(state.get("chunks") or [])
                ),
            )
            result = vector_store.import_document(document)
            state["import_result"] = {
                "item_count": result.item_count,
                "chunk_count": result.chunk_count,
            }
            return state
        except (VectorStoreError, ValueError, KeyError, TypeError) as exc:
            raise ImportTaskError("vector_store", "向量库写入失败") from exc
        except Exception as exc:
            raise ImportTaskError("vector_store", "向量库写入失败") from exc

    return node_import_vector_store
