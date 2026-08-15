import unittest
from unittest.mock import patch

from app.retrieval.interface import RerankedDocuments


WARNING = {
    "code": "reranker_degraded",
    "message": "重排序服务暂时不可用，本次回答已使用原始检索顺序生成",
}


class DegradedRetrieval:
    def rerank_documents(self, query, documents):
        return RerankedDocuments(
            documents=[dict(document) for document in documents[:10]],
            degraded=True,
            warning_code=WARNING["code"],
            warning_message=WARNING["message"],
        )


class RerankerDegradationTest(unittest.TestCase):
    def test_degraded_topk_keeps_first_ten_without_scores(self):
        from app.query_process.agent.nodes import node_rerank

        documents = [{"text": str(index)} for index in range(15)]

        result = node_rerank.step_3_topk(documents, degraded=True)

        self.assertEqual([document["text"] for document in result], [str(i) for i in range(10)])
        self.assertTrue(all("score" not in document for document in result))

    def test_successful_topk_still_uses_score_cliff(self):
        from app.query_process.agent.nodes import node_rerank

        documents = [
            {"text": "first", "score": 0.95},
            {"text": "second", "score": 0.2},
            {"text": "third", "score": 0.19},
        ]

        result = node_rerank.step_3_topk(documents, degraded=False)

        self.assertEqual(result, [documents[0]])

    def test_node_records_and_streams_one_degradation_warning(self):
        from app.query_process.agent.nodes import node_rerank
        from app.utils.sse_utils import SSEEvent

        state = {
            "request_id": "req-1",
            "is_stream": True,
            "original_query": "question",
            "rrf_chunks": [
                {
                    "id": index,
                    "entity": {
                        "chunk_id": str(index),
                        "content": f"document {index}",
                    },
                }
                for index in range(15)
            ],
            "web_search_docs": [],
            "warnings": [dict(WARNING)],
        }

        with (
            patch.object(node_rerank, "get_retrieval", return_value=DegradedRetrieval()),
            patch.object(node_rerank, "add_running_task", lambda *args, **kwargs: None),
            patch.object(node_rerank, "add_done_task", lambda *args, **kwargs: None),
            patch.object(node_rerank, "push_to_session") as push,
        ):
            result = node_rerank.node_rerank(state)

        self.assertEqual(len(result["reranked_docs"]), 10)
        self.assertTrue(all("score" not in document for document in result["reranked_docs"]))
        self.assertEqual(result["warnings"], [WARNING])
        push.assert_called_once_with("req-1", SSEEvent.WARNING, WARNING)


if __name__ == "__main__":
    unittest.main()
