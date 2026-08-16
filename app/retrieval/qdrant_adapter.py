from app.retrieval.models import SearchHit, SearchQuery
from app.retrieval.vector_search import VectorSearchError
from app.vector_store.config import QdrantVectorStoreConfig


class QdrantVectorSearch:
    def __init__(
        self,
        config: QdrantVectorStoreConfig,
        user_id: str,
        *,
        client=None,
        models_module=None,
    ):
        self._config = config
        self._user_id = user_id
        if client is None or models_module is None:
            from qdrant_client import QdrantClient, models

            models_module = models if models_module is None else models_module
            client = client or QdrantClient(
                url=config.url,
                api_key=config.api_key,
                prefer_grpc=False,
                cloud_inference=config.cloud_inference,
            )
        self._client = client
        self._models = models_module

    def search_items(self, query: SearchQuery, *, top_k: int = 5) -> list[SearchHit]:
        return self._search(
            collection_name=self._config.item_collection,
            query=query,
            top_k=top_k,
            filter_conditions=[],
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
            query=query,
            top_k=top_k,
            filter_conditions=(
                [
                    self._models.FieldCondition(
                        key="item_name",
                        match=self._models.MatchAny(any=item_names),
                    )
                ]
                if item_names
                else []
            ),
        )

    def _search(
        self,
        *,
        collection_name: str,
        query: SearchQuery,
        top_k: int,
        filter_conditions: list,
    ) -> list[SearchHit]:
        prefetch_limit = max(top_k, 10)
        try:
            response = self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    self._models.Prefetch(
                        query=list(query.dense),
                        using="dense",
                        limit=prefetch_limit,
                    ),
                    self._models.Prefetch(
                        query=self._models.Document(
                            text=query.text,
                            model=self._config.bm25_model,
                        ),
                        using="bm25",
                        limit=prefetch_limit,
                    ),
                ],
                query=self._models.FusionQuery(fusion=self._models.Fusion.RRF),
                query_filter=self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="user_id",
                            match=self._models.MatchValue(value=self._user_id),
                        ),
                        *filter_conditions,
                    ]
                ),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorSearchError("Qdrant 知识库检索失败") from exc

        return [self._to_search_hit(point) for point in response.points]

    @staticmethod
    def _to_search_hit(point) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
            id=str(point.id),
            score=point.score,
            content=payload.get("content", ""),
            item_name=payload.get("item_name", ""),
            file_title=payload.get("file_title", ""),
            parent_title=payload.get("parent_title", ""),
        )
