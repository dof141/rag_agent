from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

SearchHit = Dict[str, Any]


@dataclass(frozen=True)
class RerankedDocuments:
    documents: List[Dict[str, Any]]
    degraded: bool = False
    warning_code: str | None = None
    warning_message: str | None = None


class Retrieval(Protocol):
    def search_chunks(
        self,
        query: str,
        item_names: List[str] | None = None,
        *,
        top_k: int = 5,
    ) -> List[SearchHit]:
        ...

    def search_chunks_with_hyde(
        self,
        query: str,
        hyde_doc: str,
        item_names: List[str] | None = None,
        *,
        top_k: int = 5,
    ) -> List[SearchHit]:
        ...

    def match_item_names(self, item_names: List[str]) -> List[Dict[str, Any]]:
        ...

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> RerankedDocuments:
        ...
