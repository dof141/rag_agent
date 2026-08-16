from typing import Protocol

from app.retrieval.models import SearchHit, SearchQuery


class VectorSearchError(RuntimeError):
    pass


class VectorSearch(Protocol):
    def search_items(self, query: SearchQuery, *, top_k: int = 5) -> list[SearchHit]:
        ...

    def search_chunks(
        self,
        query: SearchQuery,
        item_names: list[str],
        *,
        top_k: int = 5,
    ) -> list[SearchHit]:
        ...
