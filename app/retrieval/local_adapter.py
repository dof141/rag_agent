from typing import Any

from app.embedding.interface import EmbeddingProvider
from app.retrieval.interface import RerankedDocuments
from app.retrieval.models import SearchHit, SearchQuery
from app.retrieval.vector_search import VectorSearch


class RetrievalModule:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_search: VectorSearch,
        reranker,
        *,
        expected_dimension: int | None = None,
    ):
        self._embedding = embedding
        self._vector_search = vector_search
        self._reranker = reranker
        self._expected_dimension = expected_dimension

    @property
    def vector_search(self) -> VectorSearch:
        return self._vector_search

    def match_item_names(self, item_names: list[str]) -> list[dict[str, Any]]:
        if not item_names:
            return []

        queries = self._embed_queries(item_names)
        results = []
        for item_name, query in zip(item_names, queries):
            hits = self._vector_search.search_items(query, top_k=5)
            results.append(
                {
                    "extracted": item_name,
                    "matches": [
                        {"item_name": hit.item_name, "score": hit.score}
                        for hit in hits
                        if hit.item_name
                    ],
                }
            )
        return results

    def search_chunks(
        self,
        query: str,
        item_names: list[str] | None = None,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        search_query = self._embed_queries([query])[0]
        hits = self._vector_search.search_chunks(
            search_query,
            list(item_names or []),
            top_k=top_k,
        )
        return [self._to_chunk_dict(hit) for hit in hits]

    def search_chunks_with_hyde(
        self,
        query: str,
        hyde_doc: str,
        item_names: list[str] | None = None,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not hyde_doc.strip():
            raise ValueError("hyde_doc cannot be empty")
        return self.search_chunks(
            f"{query} {hyde_doc}",
            item_names,
            top_k=top_k,
        )

    def rerank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> RerankedDocuments:
        if not documents:
            return RerankedDocuments(documents=[])

        outcome = self._reranker(query, [document["text"] for document in documents])
        reranked_documents = []
        for item in outcome.items:
            if type(item.index) is not int or not 0 <= item.index < len(documents):
                raise ValueError("reranker returned invalid document index")
            document = dict(documents[item.index])
            if item.score is not None:
                document["score"] = item.score
            reranked_documents.append(document)

        return RerankedDocuments(
            documents=reranked_documents,
            degraded=outcome.degraded,
            warning_code=outcome.warning_code,
            warning_message=outcome.warning_message,
        )

    def _embed_queries(self, texts: list[str]) -> list[SearchQuery]:
        embedding = self._embedding.embed_documents(texts)
        dense_vectors = embedding.get("dense") or []
        if len(dense_vectors) != len(texts):
            raise ValueError("embedding result count does not match input texts")

        sparse_vectors = embedding.get("sparse")
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError("embedding result count does not match input texts")

        queries = []
        for index, text in enumerate(texts):
            dense = dense_vectors[index]
            if not dense:
                raise ValueError("dense embedding cannot be empty")
            if (
                self._expected_dimension is not None
                and len(dense) != self._expected_dimension
            ):
                raise ValueError(
                    "dense embedding dimension does not match expected dimension"
                )
            sparse = None if sparse_vectors is None else dict(sparse_vectors[index])
            queries.append(SearchQuery(text=text, dense=tuple(dense), sparse=sparse))
        return queries

    @staticmethod
    def _to_chunk_dict(hit: SearchHit) -> dict[str, Any]:
        return {
            "id": hit.id,
            "score": hit.score,
            "entity": {
                "content": hit.content,
                "item_name": hit.item_name,
                "file_title": hit.file_title,
                "parent_title": hit.parent_title,
                "source": hit.source,
            },
        }


# Transitional import compatibility while query nodes are migrated.
LocalRetrievalAdapter = RetrievalModule
