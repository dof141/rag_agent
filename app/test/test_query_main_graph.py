import unittest

from langgraph.checkpoint.memory import InMemorySaver

from app.query_process.agent.main_graph import build_query_graph
from app.query_process.runtime import QueryRuntime


class FailOnUseRetrieval:
    def __getattr__(self, name):
        raise AssertionError(f"graph construction must not use retrieval.{name}")


class QueryMainGraphTest(unittest.TestCase):
    def test_builds_expected_graph_from_injected_runtime(self):
        runtime = QueryRuntime(
            user_id="user-a",
            settings_version=7,
            retrieval=FailOnUseRetrieval(),
        )

        graph = build_query_graph(runtime, InMemorySaver()).get_graph()

        expected_nodes = {
            "node_answer_output",
            "node_item_name_confirm",
            "node_rerank",
            "node_rrf",
            "node_search_embedding",
            "node_search_embedding_hyde",
            "node_wait_item_confirmation",
        }
        self.assertTrue(expected_nodes.issubset(graph.nodes))
        self.assertNotIn("node_web_search_mcp", graph.nodes)

        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("node_rrf", "node_rerank"), edges)
        self.assertIn(("node_rerank", "node_answer_output"), edges)


if __name__ == "__main__":
    unittest.main()
