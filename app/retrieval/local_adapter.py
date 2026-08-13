from typing import Any, Dict, List

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.reranker_utils import get_reranker_model
from app.retrieval.interface import SearchHit


class LocalRetrievalAdapter:
    def search_chunks(
        self,
        query: str,
        item_names: List[str] | None = None,
        *,
        top_k: int = 5,
    ) -> List[SearchHit]:
        embedding = generate_embeddings([query])
        reqs = create_hybrid_search_requests(
            dense_vector=embedding["dense"][0],
            sparse_vector=embedding["sparse"][0],
            limit=10,
            expr=self._item_filter(item_names),
        )
        response = self._hybrid_search(
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=(0.9, 0.1),
            top_k=top_k,
            output_fields=["chunk_id", "content", "item_name", "file_title", "parent_title"],
        )
        return response[0] if response else []

    def search_chunks_with_hyde(
        self,
        query: str,
        hyde_doc: str,
        item_names: List[str] | None = None,
        *,
        top_k: int = 5,
    ) -> List[SearchHit]:
        if not query:
            raise ValueError("query cannot be empty")
        if not hyde_doc:
            raise ValueError("hyde_doc cannot be empty")
        combined_text = f"{query} {hyde_doc}"
        embedding = generate_embeddings([combined_text])
        reqs = create_hybrid_search_requests(
            dense_vector=embedding["dense"][0],
            sparse_vector=embedding["sparse"][0],
            limit=10,
            expr=self._item_filter(item_names),
        )
        response = self._hybrid_search(
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=(0.9, 0.1),
            top_k=top_k,
            output_fields=["chunk_id", "content", "item_name", "file_title", "parent_title"],
        )
        return response[0] if response else []

    def match_item_names(self, item_names: List[str]) -> List[Dict[str, Any]]:
        if not item_names:
            return []

        embedding = generate_embeddings(item_names)
        final_result = []
        for index, item_name in enumerate(item_names):
            reqs = create_hybrid_search_requests(
                embedding["dense"][index],
                embedding["sparse"][index],
            )
            response = self._hybrid_search(
                collection_name=milvus_config.item_name_collection,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                top_k=5,
                output_fields=["item_name"],
            )
            matches = []
            if response and len(response) > 0:
                for hit in response[0]:
                    entity = hit.get("entity", {})
                    hit_name = entity.get("item_name")
                    if hit_name:
                        matches.append(
                            {
                                "item_name": hit_name,
                                "score": hit.get("distance", 0),
                            }
                        )
            final_result.append({"extracted": item_name, "matches": matches})
        return final_result

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents or not query:
            return documents

        rerank_model = get_reranker_model()
        pairs = [[query, doc["text"]] for doc in documents]
        all_scores = []
        for i in range(0, len(pairs), 4):
            batch_scores = rerank_model.compute_score(pairs[i : i + 4], normalize=True)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(batch_scores)

        scored_docs = []
        for score, item in zip(all_scores, documents):
            item_copy = dict(item)
            item_copy["score"] = float(score)
            scored_docs.append(item_copy)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs

    def _hybrid_search(
        self,
        *,
        collection_name: str,
        reqs,
        ranker_weights,
        top_k: int,
        output_fields: List[str],
    ):
        milvus_client = get_milvus_client()
        if not milvus_client:
            logger.error("Cannot connect to Milvus")
            return None
        return hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            limit=top_k,
            norm_score=True,
            output_fields=output_fields,
        )

    @staticmethod
    def _item_filter(item_names: List[str] | None) -> str | None:
        quoted = ", ".join(f'"{v}"' for v in item_names) if item_names else ""
        return f"item_name in [{quoted}]" if item_names else None
