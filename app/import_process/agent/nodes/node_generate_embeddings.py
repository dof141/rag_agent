from app.embedding.interface import EmbeddingError
from app.import_process.errors import ImportTaskError


def create_generate_embeddings_node(provider):
    def node_generate_embeddings(state):
        chunks = state.get("chunks") or []
        texts = [state.get("item_name") or "", *(chunk.get("content", "") for chunk in chunks)]
        try:
            result = provider.embed_documents(texts)
            dense = result["dense"]
            if len(dense) != len(texts):
                raise ValueError("embedding 结果数量与输入文本数量不一致")
            sparse = result.get("sparse")
            if sparse is not None and len(sparse) != len(texts):
                raise ValueError("sparse embedding 结果数量与输入文本数量不一致")
            state["item_dense_vector"] = tuple(float(value) for value in dense[0])
            state["item_sparse_vector"] = sparse[0] if sparse is not None else None
            updated_chunks = []
            for index, chunk in enumerate(chunks):
                updated = dict(chunk)
                updated["dense_vector"] = tuple(float(value) for value in dense[index + 1])
                updated["sparse_vector"] = sparse[index + 1] if sparse is not None else None
                updated_chunks.append(updated)
            state["chunks"] = updated_chunks
            return state
        except (EmbeddingError, ValueError, KeyError, TypeError) as exc:
            raise ImportTaskError("embedding", "文档向量生成失败") from exc
        except Exception as exc:
            raise ImportTaskError("embedding", "文档向量生成失败") from exc

    return node_generate_embeddings
