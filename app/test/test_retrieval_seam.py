import unittest
from unittest.mock import patch

from app.retrieval.interface import RerankedDocuments
from app.retrieval.models import SearchHit, SearchQuery


class RecordingEmbedding:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return self.result


class RecordingVectorSearch:
    def __init__(self, *, item_hits=None, chunk_hits=None):
        self.item_hits = item_hits or {}
        self.chunk_hits = chunk_hits or []
        self.item_calls = []
        self.chunk_calls = []

    def search_items(self, query, *, top_k=5):
        self.item_calls.append((query, top_k))
        return self.item_hits.get(query.text, [])

    def search_chunks(self, query, item_names, *, top_k=5):
        self.chunk_calls.append((query, list(item_names), top_k))
        return list(self.chunk_hits)


class FakeRetrieval:
    def __init__(self):
        self.calls = []

    def search_chunks(self, query, item_names=None, *, top_k=5):
        self.calls.append((query, item_names, top_k))
        return [{"id": 1, "entity": {"content": "answer", "item_name": "demo"}}]

    def search_chunks_with_hyde(self, query, hyde_doc, item_names=None, *, top_k=5):
        self.calls.append((query, hyde_doc, item_names, top_k))
        return [{"id": 2, "entity": {"content": "hyde answer", "item_name": "demo"}}]

    def match_item_names(self, item_names):
        self.calls.append(tuple(item_names))
        return [{"extracted": item_names[0], "matches": [{"item_name": "demo", "score": 0.91}]}]

    def rerank_documents(self, query, documents):
        self.calls.append((query, documents))
        return RerankedDocuments(documents=[dict(documents[0], score=0.99)])


class RetrievalSeamTest(unittest.TestCase):
    def test_retrieval_module_exposes_injected_vector_search_as_read_only(self):
        from app.retrieval.local_adapter import RetrievalModule

        vector_search = RecordingVectorSearch()
        retrieval = RetrievalModule(RecordingEmbedding({"dense": []}), vector_search, object())

        self.assertIs(retrieval.vector_search, vector_search)
        with self.assertRaises(AttributeError):
            retrieval.vector_search = object()

    def test_match_item_names_short_circuits_empty_input(self):
        from app.retrieval.local_adapter import RetrievalModule

        embedding = RecordingEmbedding({"dense": []})
        vector_search = RecordingVectorSearch()
        retrieval = RetrievalModule(embedding, vector_search, object())

        self.assertEqual(retrieval.match_item_names([]), [])
        self.assertEqual(embedding.calls, [])
        self.assertEqual(vector_search.item_calls, [])

    def test_match_item_names_batches_embeddings_and_preserves_input_order(self):
        from app.retrieval.local_adapter import RetrievalModule

        embedding = RecordingEmbedding(
            {
                "dense": [[0.1, 0.2], [0.3, 0.4]],
                "sparse": [{1: 0.5}, {2: 0.6}],
            }
        )
        vector_search = RecordingVectorSearch(
            item_hits={
                "alpha": [
                    SearchHit(
                        id="item-1",
                        score=0.91,
                        content="",
                        item_name="Alpha Guide",
                        file_title="",
                        parent_title="",
                    )
                ],
                "beta": [
                    SearchHit(
                        id="item-2",
                        score=0.82,
                        content="",
                        item_name="Beta FAQ",
                        file_title="",
                        parent_title="",
                    )
                ],
            }
        )
        retrieval = RetrievalModule(embedding, vector_search, object())

        result = retrieval.match_item_names(["alpha", "beta"])

        self.assertEqual(embedding.calls, [["alpha", "beta"]])
        self.assertEqual(
            vector_search.item_calls,
            [
                (SearchQuery("alpha", (0.1, 0.2), {1: 0.5}), 5),
                (SearchQuery("beta", (0.3, 0.4), {2: 0.6}), 5),
            ],
        )
        self.assertEqual(
            result,
            [
                {
                    "extracted": "alpha",
                    "matches": [{"item_name": "Alpha Guide", "score": 0.91}],
                },
                {
                    "extracted": "beta",
                    "matches": [{"item_name": "Beta FAQ", "score": 0.82}],
                },
            ],
        )

    def test_search_chunks_embeds_and_returns_provider_neutral_dicts(self):
        from app.retrieval.local_adapter import RetrievalModule

        embedding = RecordingEmbedding(
            {"dense": [[0.1, 0.2]], "sparse": [{4: 0.7}]}
        )
        hit = SearchHit(
            id="chunk-1",
            score=0.88,
            content="answer",
            item_name="guide",
            file_title="manual.md",
            parent_title="Install",
            source="knowledge_base",
        )
        vector_search = RecordingVectorSearch(chunk_hits=[hit])
        retrieval = RetrievalModule(embedding, vector_search, object())

        result = retrieval.search_chunks("where", ["guide"], top_k=7)

        self.assertEqual(embedding.calls, [["where"]])
        self.assertEqual(
            vector_search.chunk_calls,
            [(SearchQuery("where", (0.1, 0.2), {4: 0.7}), ["guide"], 7)],
        )
        self.assertEqual(
            result,
            [
                {
                    "id": "chunk-1",
                    "score": 0.88,
                    "entity": {
                        "content": "answer",
                        "item_name": "guide",
                        "file_title": "manual.md",
                        "parent_title": "Install",
                        "source": "knowledge_base",
                    },
                }
            ],
        )

    def test_search_chunks_with_hyde_embeds_combined_text(self):
        from app.retrieval.local_adapter import RetrievalModule

        embedding = RecordingEmbedding({"dense": [[0.3]], "sparse": [{8: 0.4}]})
        vector_search = RecordingVectorSearch()
        retrieval = RetrievalModule(embedding, vector_search, object())

        result = retrieval.search_chunks_with_hyde(
            "question",
            "hypothetical answer",
            ["guide"],
            top_k=3,
        )

        self.assertEqual(result, [])
        self.assertEqual(embedding.calls, [["question hypothetical answer"]])
        self.assertEqual(
            vector_search.chunk_calls,
            [
                (
                    SearchQuery(
                        "question hypothetical answer",
                        (0.3,),
                        {8: 0.4},
                    ),
                    ["guide"],
                    3,
                )
            ],
        )

    def test_search_chunks_rejects_blank_query_before_dependencies(self):
        from app.retrieval.local_adapter import RetrievalModule

        for query in ("", "  \t"):
            with self.subTest(query=query):
                embedding = RecordingEmbedding({"dense": [[0.1]]})
                vector_search = RecordingVectorSearch()
                retrieval = RetrievalModule(embedding, vector_search, object())

                with self.assertRaisesRegex(ValueError, "query cannot be empty"):
                    retrieval.search_chunks(query)

                self.assertEqual(embedding.calls, [])
                self.assertEqual(vector_search.chunk_calls, [])

    def test_hyde_search_rejects_blank_inputs_before_dependencies(self):
        from app.retrieval.local_adapter import RetrievalModule

        cases = [
            ("", "document", "query cannot be empty"),
            ("  \t", "document", "query cannot be empty"),
            ("question", "", "hyde_doc cannot be empty"),
            ("question", " \n", "hyde_doc cannot be empty"),
        ]
        for query, hyde_doc, message in cases:
            with self.subTest(query=query, hyde_doc=hyde_doc):
                embedding = RecordingEmbedding({"dense": [[0.1]]})
                vector_search = RecordingVectorSearch()
                retrieval = RetrievalModule(embedding, vector_search, object())

                with self.assertRaisesRegex(ValueError, message):
                    retrieval.search_chunks_with_hyde(query, hyde_doc)

                self.assertEqual(embedding.calls, [])
                self.assertEqual(vector_search.chunk_calls, [])

    def test_embedding_count_mismatch_raises_stable_value_error(self):
        from app.retrieval.local_adapter import RetrievalModule

        vector_search = RecordingVectorSearch()
        retrieval = RetrievalModule(
            RecordingEmbedding({"dense": [[0.1]], "sparse": [{1: 0.2}]}),
            vector_search,
            object(),
        )

        with self.assertRaisesRegex(ValueError, "embedding result count"):
            retrieval.match_item_names(["alpha", "beta"])

        self.assertEqual(vector_search.item_calls, [])

    def test_empty_dense_vector_raises_stable_value_error(self):
        from app.retrieval.local_adapter import RetrievalModule

        retrieval = RetrievalModule(
            RecordingEmbedding({"dense": [[]]}),
            RecordingVectorSearch(),
            object(),
        )

        with self.assertRaisesRegex(ValueError, "dense embedding"):
            retrieval.search_chunks("question")

    def test_dense_dimension_mismatch_never_reaches_vector_search(self):
        from app.retrieval.local_adapter import RetrievalModule

        vector_search = RecordingVectorSearch()
        retrieval = RetrievalModule(
            RecordingEmbedding({"dense": [[0.1, 0.2]]}),
            vector_search,
            object(),
            expected_dimension=3,
        )

        with self.assertRaisesRegex(ValueError, "dense embedding dimension"):
            retrieval.search_chunks("question")

        self.assertEqual(vector_search.chunk_calls, [])

    def test_node_search_embedding_uses_retrieval_interface(self):
        from app.query_process.agent.nodes import node_search_embedding

        fake = FakeRetrieval()
        with (
            patch.object(node_search_embedding, "get_retrieval", return_value=fake),
            patch.object(node_search_embedding, "add_running_task", lambda *args, **kwargs: None),
            patch.object(node_search_embedding, "add_done_task", lambda *args, **kwargs: None),
        ):
            result = node_search_embedding.node_search_embedding(
                {
                    "request_id": "req-1",
                    "rewritten_query": "how to use it",
                    "item_names": ["demo"],
                    "is_stream": False,
                }
            )

        self.assertEqual(fake.calls, [("how to use it", ["demo"], 5)])
        self.assertEqual(
            result,
            {"embedding_chunks": [{"id": 1, "entity": {"content": "answer", "item_name": "demo"}}]},
        )

    def test_hyde_search_uses_retrieval_interface(self):
        from app.query_process.agent.nodes import node_search_embedding_hyde

        fake = FakeRetrieval()
        with patch.object(node_search_embedding_hyde, "get_retrieval", return_value=fake):
            result = node_search_embedding_hyde.step_2_search_embedding_hyde(
                rewritten_query="how to use it",
                hyde_doc="a hypothetical answer",
                item_names=["demo"],
                top_k=3,
            )

        self.assertEqual(fake.calls, [("how to use it", "a hypothetical answer", ["demo"], 3)])
        self.assertEqual(result, [{"id": 2, "entity": {"content": "hyde answer", "item_name": "demo"}}])

    def test_item_name_matching_uses_retrieval_interface(self):
        from app.query_process.agent.nodes import node_item_name_confirm

        fake = FakeRetrieval()
        with patch.object(node_item_name_confirm, "get_retrieval", return_value=fake):
            result = node_item_name_confirm.step_4_vectorize_and_query(["demo"])

        self.assertEqual(fake.calls, [("demo",)])
        self.assertEqual(result, [{"extracted": "demo", "matches": [{"item_name": "demo", "score": 0.91}]}])

    def test_rerank_uses_retrieval_interface(self):
        from app.query_process.agent.nodes import node_rerank

        fake = FakeRetrieval()
        docs = [{"text": "candidate", "content": "candidate"}]
        with patch.object(node_rerank, "get_retrieval", return_value=fake):
            result = node_rerank.step_2_rerank_doc_list(
                docs,
                {"rewritten_query": "question"},
            )

        self.assertEqual(fake.calls, [("question", docs)])
        self.assertEqual(
            result,
            RerankedDocuments(
                documents=[
                    {"text": "candidate", "content": "candidate", "score": 0.99}
                ]
            ),
        )

    def test_retrieval_maps_rerank_indexes_without_mutating_documents(self):
        from app.reranker.interface import RerankItem, RerankOutcome
        from app.retrieval.local_adapter import LocalRetrievalAdapter

        docs = [
            {"text": "first", "meta": 1},
            {"text": "second", "meta": 2},
        ]

        def fake_reranker(query, texts):
            self.assertEqual(query, "question")
            self.assertEqual(texts, ["first", "second"])
            return RerankOutcome(
                items=[
                    RerankItem(index=1, score=0.9),
                    RerankItem(index=0, score=0.2),
                ]
            )

        result = LocalRetrievalAdapter(
            RecordingEmbedding({"dense": []}),
            RecordingVectorSearch(),
            fake_reranker,
        ).rerank_documents(
            "question",
            docs,
        )

        self.assertEqual(
            [document["text"] for document in result.documents],
            ["second", "first"],
        )
        self.assertEqual(result.documents[0]["meta"], 2)
        self.assertEqual(result.documents[0]["score"], 0.9)
        self.assertNotIn("score", docs[0])
        self.assertNotIn("score", docs[1])

    def test_rerank_rejects_blank_query_before_calling_reranker(self):
        from app.reranker.interface import RerankOutcome
        from app.retrieval.local_adapter import RetrievalModule

        docs = [{"text": "candidate", "meta": 1}]
        for query in ("", "  \t"):
            with self.subTest(query=query):
                calls = []

                def reranker(rerank_query, texts):
                    calls.append((rerank_query, texts))
                    return RerankOutcome(items=[])

                retrieval = RetrievalModule(
                    RecordingEmbedding({"dense": []}),
                    RecordingVectorSearch(),
                    reranker,
                )

                with self.assertRaisesRegex(ValueError, "query cannot be empty"):
                    retrieval.rerank_documents(query, docs)

                self.assertEqual(calls, [])
                self.assertEqual(docs, [{"text": "candidate", "meta": 1}])

    def test_retrieval_preserves_reranker_degradation_metadata(self):
        from app.reranker.interface import RerankItem, RerankOutcome
        from app.retrieval.local_adapter import RetrievalModule

        docs = [{"text": "first"}, {"text": "second"}]

        def degraded_reranker(query, texts):
            self.assertEqual((query, texts), ("question", ["first", "second"]))
            return RerankOutcome(
                items=[RerankItem(index=1, score=None)],
                degraded=True,
                warning_code="reranker_unavailable",
                warning_message="fallback order used",
            )

        result = RetrievalModule(
            RecordingEmbedding({"dense": []}),
            RecordingVectorSearch(),
            degraded_reranker,
        ).rerank_documents("question", docs)

        self.assertEqual(result.documents, [{"text": "second"}])
        self.assertTrue(result.degraded)
        self.assertEqual(result.warning_code, "reranker_unavailable")
        self.assertEqual(result.warning_message, "fallback order used")
        self.assertEqual(docs, [{"text": "first"}, {"text": "second"}])

    def test_retrieval_rejects_invalid_reranker_document_indexes(self):
        from app.reranker.interface import RerankItem, RerankOutcome
        from app.retrieval.local_adapter import RetrievalModule

        docs = [{"text": "first"}, {"text": "second"}]
        for invalid_index in (-1, 2, "1"):
            with self.subTest(index=invalid_index):
                retrieval = RetrievalModule(
                    RecordingEmbedding({"dense": []}),
                    RecordingVectorSearch(),
                    lambda _query, _texts: RerankOutcome(
                        items=[RerankItem(index=invalid_index, score=0.5)]
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "reranker returned invalid document index",
                ):
                    retrieval.rerank_documents("question", docs)

        self.assertEqual(docs, [{"text": "first"}, {"text": "second"}])


if __name__ == "__main__":
    unittest.main()
