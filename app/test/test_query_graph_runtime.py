import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.query_process.agent.nodes import (
    node_answer_output,
    node_item_name_confirm,
    node_rerank,
    node_search_embedding,
    node_search_embedding_hyde,
)
from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion
from app.query_process.agent.state import create_query_default_state
from app.query_process.runtime import QueryRuntime
from app.retrieval.interface import RerankedDocuments
from app.utils.sse_utils import SSEEvent


NO_EVIDENCE_ANSWER = "知识库中没有足够依据回答该问题。"


class RecordingRetrieval:
    def __init__(self):
        self.calls = []

    def match_item_names(self, item_names):
        self.calls.append(("match_item_names", item_names))
        return [
            {
                "extracted": "demo",
                "matches": [{"item_name": "demo", "score": 0.91}],
            }
        ]

    def search_chunks(self, query, item_names=None, *, top_k=5):
        self.calls.append(("search_chunks", query, item_names, top_k))
        return [{"id": "plain", "entity": {"content": "plain answer"}}]

    def search_chunks_with_hyde(
        self,
        query,
        hyde_doc,
        item_names=None,
        *,
        top_k=5,
    ):
        self.calls.append(
            ("search_chunks_with_hyde", query, hyde_doc, item_names, top_k)
        )
        return [{"id": "hyde", "entity": {"content": "hyde answer"}}]

    def rerank_documents(self, query, documents):
        self.calls.append(("rerank_documents", query, documents))
        return RerankedDocuments(
            documents=[dict(document, score=0.99) for document in documents]
        )


class QueryGraphRuntimeTest(unittest.TestCase):
    def test_graph_retrieval_nodes_use_injected_runtime(self):
        from app.query_process.agent import main_graph

        retrieval = RecordingRetrieval()
        runtime = QueryRuntime(
            user_id="user-a",
            settings_version=3,
            retrieval=retrieval,
        )
        graph = main_graph.build_query_graph(runtime, InMemorySaver())
        graph_nodes = graph.get_graph().nodes

        base_state = create_query_default_state(
            request_id="req-1",
            user_id="user-a",
            session_id="session-1",
            original_query="question",
            rewritten_query="rewritten question",
            item_names=["demo"],
            is_stream=False,
        )

        with (
            patch.object(
                node_item_name_confirm,
                "step_3_llm_item_name_and_rewrite_query",
                return_value=node_item_name_confirm.GoodsResponse(
                    item_names=["demo"],
                    rewritten_query="rewritten question",
                ),
            ),
            patch.object(node_item_name_confirm, "get_recent_messages", return_value=[]),
            patch.object(node_item_name_confirm, "save_chat_message"),
            patch.object(node_item_name_confirm, "add_running_task"),
            patch.object(node_item_name_confirm, "add_done_task"),
        ):
            item_state = graph_nodes["node_item_name_confirm"].data.invoke(
                dict(base_state)
            )

        with (
            patch.object(node_search_embedding, "add_running_task"),
            patch.object(node_search_embedding, "add_done_task"),
        ):
            embedding_state = graph_nodes["node_search_embedding"].data.invoke(
                dict(base_state)
            )

        with (
            patch.object(
                node_search_embedding_hyde,
                "step_1_create_hyde_doc",
                return_value="hypothetical answer",
            ),
            patch.object(node_search_embedding_hyde, "add_running_task"),
            patch.object(node_search_embedding_hyde, "add_done_task"),
        ):
            hyde_state = graph_nodes["node_search_embedding_hyde"].data.invoke(
                dict(base_state)
            )

        rerank_state = dict(
            base_state,
            rrf_chunks=[
                {
                    "id": "plain",
                    "entity": {"chunk_id": "plain", "content": "plain answer"},
                }
            ],
            web_search_docs=[],
        )
        with (
            patch.object(node_rerank, "add_running_task"),
            patch.object(node_rerank, "add_done_task"),
        ):
            reranked_state = graph_nodes["node_rerank"].data.invoke(rerank_state)

        self.assertEqual(item_state["item_names"], ["demo"])
        self.assertEqual(embedding_state["embedding_chunks"][0]["id"], "plain")
        self.assertEqual(hyde_state["hyde_embedding_chunks"][0]["id"], "hyde")
        self.assertEqual(reranked_state["reranked_docs"][0]["score"], 0.99)
        self.assertEqual(
            [call[0] for call in retrieval.calls],
            [
                "match_item_names",
                "search_chunks",
                "search_chunks_with_hyde",
                "rerank_documents",
            ],
        )
        self.assertFalse({"runtime", "retrieval", "api_key", "client"}.intersection(item_state))

    def test_default_graph_does_not_schedule_web_search(self):
        from app.query_process.agent import main_graph

        runtime = QueryRuntime(
            user_id="user-a",
            settings_version=3,
            retrieval=RecordingRetrieval(),
        )

        graph = main_graph.build_query_graph(runtime, InMemorySaver()).get_graph()

        self.assertNotIn("node_web_search_mcp", graph.nodes)
        self.assertFalse(
            any(edge.target == "node_web_search_mcp" for edge in graph.edges)
        )

    def test_rrf_accumulates_scores_for_hits_from_multiple_routes(self):
        repeated = {"id": "A", "entity": {"content": "repeated"}}
        single = {"id": "B", "entity": {"content": "single"}}

        ranked = step_3_reciprocal_rank_fusion(
            [
                ([single, repeated], 1.0),
                ([repeated], 1.0),
            ],
            top_k=2,
        )

        self.assertEqual([chunk["id"] for chunk in ranked], ["A", "B"])

    def test_empty_evidence_skips_llm_images_and_sources_for_all_modes(self):
        for is_stream in (False, True):
            with self.subTest(is_stream=is_stream):
                state = create_query_default_state(
                    request_id="req-1",
                    user_id="user-a",
                    session_id="session-1",
                    original_query="question",
                    rewritten_query="rewritten question",
                    reranked_docs=[],
                    is_stream=is_stream,
                )

                with (
                    patch.object(node_answer_output, "get_llm_client") as get_llm,
                    patch.object(node_answer_output, "step_4_extract_images_url") as images,
                    patch.object(node_answer_output, "step_4_5_extract_sources") as sources,
                    patch.object(node_answer_output, "add_running_task"),
                    patch.object(node_answer_output, "add_done_task"),
                    patch.object(node_answer_output, "get_node_durations", return_value={}),
                    patch.object(node_answer_output, "get_total_duration", return_value=0.5),
                    patch.object(node_answer_output, "save_chat_message"),
                    patch.object(node_answer_output, "set_task_result") as set_result,
                    patch.object(node_answer_output, "push_to_session") as push,
                ):
                    result = node_answer_output.node_answer_output(state)

                get_llm.assert_not_called()
                images.assert_not_called()
                sources.assert_not_called()
                self.assertEqual(result["answer"], NO_EVIDENCE_ANSWER)
                self.assertEqual(result["image_urls"], [])
                self.assertEqual(result["sources"], [])
                self.assertNotIn(
                    "node_web_search_mcp",
                    {step["node_id"] for step in result["node_steps"]},
                )

                final_calls = [
                    call
                    for call in push.call_args_list
                    if call.args[1] == SSEEvent.FINAL
                ]
                if is_stream:
                    self.assertEqual(len(final_calls), 1)
                    self.assertEqual(final_calls[0].args[2]["answer"], NO_EVIDENCE_ANSWER)
                    set_result.assert_not_called()
                else:
                    self.assertEqual(final_calls, [])
                    set_result.assert_called_once_with(
                        "req-1",
                        "answer",
                        NO_EVIDENCE_ANSWER,
                    )


if __name__ == "__main__":
    unittest.main()
