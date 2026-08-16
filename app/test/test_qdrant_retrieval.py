import unittest

from qdrant_client import models
from qdrant_client.http.models import QueryResponse, ScoredPoint

from app.retrieval import SearchHit, SearchQuery, VectorSearchError
from app.retrieval.qdrant_adapter import QdrantVectorSearch
from app.vector_store.config import QdrantVectorStoreConfig


class RecordingQdrantClient:
    def __init__(self, *, fail_query=False):
        self.fail_query = fail_query
        self.query_calls = []

    def query_points(self, **kwargs):
        if self.fail_query:
            raise RuntimeError("provider-detail-must-not-leak")
        self.query_calls.append(kwargs)
        return QueryResponse(
            points=[
                ScoredPoint(
                    id=123,
                    version=1,
                    score=0.91,
                    payload={
                        "content": "matching content",
                        "item_name": "guide",
                        "file_title": "guide.pdf",
                    },
                )
            ]
        )


class QdrantVectorSearchTest(unittest.TestCase):
    def config(self):
        return QdrantVectorStoreConfig(
            url="https://qdrant.example",
            api_key="qd-secret",
            item_collection="items",
            chunks_collection="chunks",
            bm25_model="Qdrant/bm25",
        )

    def query(self):
        return SearchQuery(text="where is the guide", dense=(0.1, 0.2))

    def test_search_items_uses_hybrid_query_and_user_filter(self):
        client = RecordingQdrantClient()
        search = QdrantVectorSearch(self.config(), user_id="user-a", client=client)

        result = search.search_items(self.query(), top_k=5)

        self.assertEqual(
            result,
            [
                SearchHit(
                    id="123",
                    score=0.91,
                    content="matching content",
                    item_name="guide",
                    file_title="guide.pdf",
                    parent_title="",
                )
            ],
        )
        self.assertEqual(len(client.query_calls), 1)
        call = client.query_calls[0]
        self.assertEqual(call["collection_name"], "items")
        self.assertEqual(call["limit"], 5)
        self.assertTrue(call["with_payload"])
        self.assertEqual(len(call["prefetch"]), 2)
        dense_prefetch, bm25_prefetch = call["prefetch"]
        self.assertEqual(dense_prefetch.query, [0.1, 0.2])
        self.assertEqual(dense_prefetch.using, "dense")
        self.assertEqual(dense_prefetch.limit, 10)
        self.assertEqual(bm25_prefetch.query.text, "where is the guide")
        self.assertEqual(bm25_prefetch.query.model, "Qdrant/bm25")
        self.assertEqual(bm25_prefetch.using, "bm25")
        self.assertEqual(bm25_prefetch.limit, 10)
        self.assertEqual(call["query"].fusion, models.Fusion.RRF)
        self.assertEqual(len(call["query_filter"].must), 1)
        user_filter = call["query_filter"].must[0]
        self.assertEqual(user_filter.key, "user_id")
        self.assertEqual(user_filter.match.value, "user-a")

    def test_search_chunks_filters_by_bound_user_and_requested_items(self):
        client = RecordingQdrantClient()
        search = QdrantVectorSearch(self.config(), user_id="user-a", client=client)

        result = search.search_chunks(self.query(), ["guide", "faq"], top_k=12)

        self.assertEqual(result[0].parent_title, "")
        call = client.query_calls[0]
        self.assertEqual(call["collection_name"], "chunks")
        self.assertEqual(call["limit"], 12)
        self.assertEqual([prefetch.limit for prefetch in call["prefetch"]], [12, 12])
        self.assertEqual(len(call["query_filter"].must), 2)
        filters = {condition.key: condition.match for condition in call["query_filter"].must}
        self.assertEqual(filters["user_id"].value, "user-a")
        self.assertEqual(filters["item_name"].any, ["guide", "faq"])

    def test_search_chunks_without_items_searches_all_items_for_bound_user(self):
        client = RecordingQdrantClient()
        search = QdrantVectorSearch(self.config(), user_id="user-a", client=client)

        search.search_chunks(self.query(), [], top_k=5)

        self.assertEqual(len(client.query_calls), 1)
        conditions = client.query_calls[0]["query_filter"].must
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].key, "user_id")
        self.assertEqual(conditions[0].match.value, "user-a")

    def test_provider_errors_do_not_leak_from_search(self):
        search = QdrantVectorSearch(
            self.config(),
            user_id="user-a",
            client=RecordingQdrantClient(fail_query=True),
        )

        with self.assertRaisesRegex(VectorSearchError, "^Qdrant 知识库检索失败$") as raised:
            search.search_items(self.query())

        self.assertNotIn("provider-detail-must-not-leak", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
