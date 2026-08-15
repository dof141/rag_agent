import unittest
from unittest.mock import patch

from app.retrieval.interface import RerankedDocuments


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

        result = LocalRetrievalAdapter(reranker=fake_reranker).rerank_documents(
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


if __name__ == "__main__":
    unittest.main()
