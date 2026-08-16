from app.retrieval.models import SearchHit, SearchQuery
from app.retrieval.vector_search import VectorSearchError
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.vector_store.config import MilvusVectorStoreConfig


class MilvusVectorSearch:
    def __init__(
        self,
        config: MilvusVectorStoreConfig,
        user_id: str,
        *,
        client=None,
        request_factory=None,
        ranker_factory=None,
    ):
        if client is None or request_factory is None or ranker_factory is None:
            from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

            if client is None:
                client = MilvusClient(uri=config.url, token=config.token)
            if request_factory is None:
                request_factory = AnnSearchRequest
            if ranker_factory is None:
                ranker_factory = WeightedRanker

        self._config = config
        self._user_id = user_id
        self._client = client
        self._request_factory = request_factory
        self._ranker_factory = ranker_factory

    def search_items(self, query: SearchQuery, *, top_k: int = 5) -> list[SearchHit]:
        return self._search(
            collection_name=self._config.item_collection,
            primary_field="item_id",
            query=query,
            top_k=top_k,
            expr=self._user_expr(),
        )

    def search_chunks(
        self,
        query: SearchQuery,
        item_names: list[str],
        *,
        top_k: int = 5,
    ) -> list[SearchHit]:
        return self._search(
            collection_name=self._config.chunks_collection,
            primary_field="chunk_id",
            query=query,
            top_k=top_k,
            expr=self._chunk_expr(item_names),
        )

    def _search(
        self,
        *,
        collection_name: str,
        primary_field: str,
        query: SearchQuery,
        top_k: int,
        expr: str,
    ) -> list[SearchHit]:
        try:
            if not query.dense or not query.sparse:
                raise ValueError("dense and sparse vectors are required")

            reqs = [
                self._request_factory(
                    data=[list(query.dense)],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE"},
                    limit=top_k,
                    expr=expr,
                ),
                self._request_factory(
                    data=[query.sparse],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP"},
                    limit=top_k,
                    expr=expr,
                ),
            ]
            result_groups = self._client.hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=self._ranker_factory(0.9, 0.1, norm_score=True),
                limit=top_k,
                output_fields=[
                    primary_field,
                    "content",
                    "item_name",
                    "file_title",
                    "parent_title",
                ],
            )
            hits = result_groups[0] if result_groups else []
            return [self._to_search_hit(hit, primary_field) for hit in hits]
        except Exception as exc:
            raise VectorSearchError("Milvus 知识库检索失败") from exc

    def _user_expr(self) -> str:
        return f'user_id == "{escape_milvus_string(self._user_id)}"'

    def _chunk_expr(self, item_names: list[str]) -> str:
        expr = self._user_expr()
        if not item_names:
            return expr
        escaped_items = ", ".join(
            f'"{escape_milvus_string(item_name)}"' for item_name in item_names
        )
        return f"{expr} and item_name in [{escaped_items}]"

    @staticmethod
    def _to_search_hit(hit: dict, primary_field: str) -> SearchHit:
        entity = hit.get("entity") or hit.get("payload") or {}
        identifier = hit.get("id")
        if identifier is None:
            identifier = entity.get(primary_field, "")
        score = hit["distance"] if "distance" in hit else hit.get("score", 0.0)
        return SearchHit(
            id=str(identifier),
            score=score,
            content=entity.get("content", ""),
            item_name=entity.get("item_name", ""),
            file_title=entity.get("file_title", ""),
            parent_title=entity.get("parent_title", ""),
        )
