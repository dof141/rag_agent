import unittest

from app.retrieval import SearchHit, SearchQuery, VectorSearchError
from app.retrieval.milvus_adapter import MilvusVectorSearch
from app.vector_store.config import MilvusVectorStoreConfig


class RecordingMilvusClient:
    def __init__(self, *, response=None, fail_search=False):
        self.response = response if response is not None else [[]]
        self.fail_search = fail_search
        self.hybrid_search_calls = []

    def hybrid_search(self, **kwargs):
        if self.fail_search:
            raise RuntimeError("provider-detail-must-not-leak")
        self.hybrid_search_calls.append(kwargs)
        return self.response


class RecordingRequestFactory:
    def __init__(self):
        self.requests = []

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        return kwargs


class RecordingRankerFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, *weights, **kwargs):
        self.calls.append((weights, kwargs))
        return {"weights": weights, **kwargs}


class MilvusVectorSearchTest(unittest.TestCase):
    def config(self):
        return MilvusVectorStoreConfig(
            url="http://milvus.example",
            token="token",
            item_collection="items",
            chunks_collection="chunks",
        )

    def query(self):
        return SearchQuery(
            text="where is the guide",
            dense=(0.1, 0.2),
            sparse={7: 0.8},
        )

    def test_search_items_uses_two_hybrid_requests_and_maps_first_result_group(self):
        client = RecordingMilvusClient(
            response=[
                [
                    {
                        "id": 123,
                        "distance": 0.91,
                        "entity": {
                            "content": "matching content",
                            "item_name": "guide",
                            "file_title": "guide.pdf",
                        },
                    }
                ]
            ]
        )
        request_factory = RecordingRequestFactory()
        ranker_factory = RecordingRankerFactory()
        search = MilvusVectorSearch(
            self.config(),
            user_id="user-a",
            client=client,
            request_factory=request_factory,
            ranker_factory=ranker_factory,
        )

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
        self.assertEqual(
            request_factory.requests,
            [
                {
                    "data": [[0.1, 0.2]],
                    "anns_field": "dense_vector",
                    "param": {"metric_type": "COSINE"},
                    "limit": 5,
                    "expr": 'user_id == "user-a"',
                },
                {
                    "data": [{7: 0.8}],
                    "anns_field": "sparse_vector",
                    "param": {"metric_type": "IP"},
                    "limit": 5,
                    "expr": 'user_id == "user-a"',
                },
            ],
        )
        self.assertEqual(len(client.hybrid_search_calls), 1)
        call = client.hybrid_search_calls[0]
        self.assertEqual(call["collection_name"], "items")
        self.assertEqual(call["reqs"], request_factory.requests)
        self.assertEqual(call["ranker"], {"weights": (0.9, 0.1), "norm_score": True})
        self.assertEqual(call["limit"], 5)
        self.assertEqual(
            call["output_fields"],
            ["item_id", "content", "item_name", "file_title", "parent_title"],
        )
        self.assertEqual(ranker_factory.calls, [((0.9, 0.1), {"norm_score": True})])

    def test_search_chunks_escapes_bound_user_and_item_names(self):
        client = RecordingMilvusClient()
        request_factory = RecordingRequestFactory()
        search = MilvusVectorSearch(
            self.config(),
            user_id='user\\name "quoted"\nnext',
            client=client,
            request_factory=request_factory,
            ranker_factory=RecordingRankerFactory(),
        )

        search.search_chunks(self.query(), ['first "item"', "second\\item\nline"], top_k=12)

        expr = request_factory.requests[0]["expr"]
        self.assertEqual(
            expr,
            'user_id == "user\\\\name \\"quoted\\" next" and '
            'item_name in ["first \\"item\\"", "second\\\\item line"]',
        )
        self.assertEqual(request_factory.requests[1]["expr"], expr)
        call = client.hybrid_search_calls[0]
        self.assertEqual(call["collection_name"], "chunks")
        self.assertEqual(call["limit"], 12)
        self.assertEqual(
            call["output_fields"],
            ["chunk_id", "content", "item_name", "file_title", "parent_title"],
        )

    def test_search_chunks_without_item_names_only_filters_by_bound_user(self):
        request_factory = RecordingRequestFactory()
        search = MilvusVectorSearch(
            self.config(),
            user_id="user-a",
            client=RecordingMilvusClient(),
            request_factory=request_factory,
            ranker_factory=RecordingRankerFactory(),
        )

        search.search_chunks(self.query(), [], top_k=3)

        self.assertEqual(request_factory.requests[0]["expr"], 'user_id == "user-a"')

    def test_search_hit_uses_entity_primary_key_and_score_when_top_level_values_absent(self):
        search = MilvusVectorSearch(
            self.config(),
            user_id="user-a",
            client=RecordingMilvusClient(
                response=[
                    [
                        {
                            "score": 0.42,
                            "entity": {
                                "chunk_id": 456,
                                "content": "chunk content",
                                "item_name": "guide",
                                "file_title": "guide.pdf",
                                "parent_title": "chapter 1",
                            },
                        }
                    ]
                ]
            ),
            request_factory=RecordingRequestFactory(),
            ranker_factory=RecordingRankerFactory(),
        )

        result = search.search_chunks(self.query(), [])

        self.assertEqual(
            result,
            [
                SearchHit(
                    id="456",
                    score=0.42,
                    content="chunk content",
                    item_name="guide",
                    file_title="guide.pdf",
                    parent_title="chapter 1",
                )
            ],
        )

    def test_missing_vectors_and_provider_errors_are_public_milvus_errors(self):
        search = MilvusVectorSearch(
            self.config(),
            user_id="user-a",
            client=RecordingMilvusClient(fail_search=True),
            request_factory=RecordingRequestFactory(),
            ranker_factory=RecordingRankerFactory(),
        )

        for query in (
            SearchQuery(text="missing dense", dense=(), sparse={1: 0.1}),
            SearchQuery(text="missing sparse", dense=(0.1,), sparse=None),
            self.query(),
        ):
            with self.assertRaisesRegex(VectorSearchError, "^Milvus 知识库检索失败$") as raised:
                search.search_items(query)
            self.assertNotIn("provider-detail-must-not-leak", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
