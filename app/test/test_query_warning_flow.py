import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


WARNING = {
    "code": "reranker_degraded",
    "message": "重排序服务暂时不可用，本次回答已使用原始检索顺序生成",
}

FAST_FAIL_MONGO_URL = (
    "mongodb://127.0.0.1:27017/"
    "?serverSelectionTimeoutMS=10&connectTimeoutMS=10"
)


class QueryWarningFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(
            os.environ,
            {"MONGO_URL": FAST_FAIL_MONGO_URL},
        )
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def test_answer_output_includes_warnings_in_final_and_return_value(self):
        from app.query_process.agent.nodes import node_answer_output
        from app.utils.sse_utils import SSEEvent

        state = {
            "request_id": "req-1",
            "session_id": "session-1",
            "is_stream": True,
            "original_query": "question",
            "reranked_docs": [{"text": "supporting evidence"}],
            "warnings": [dict(WARNING)],
        }

        def create_answer(current_state, prompt):
            current_state["answer"] = "answer"
            return "answer"

        with (
            patch.object(node_answer_output, "add_running_task"),
            patch.object(node_answer_output, "add_done_task"),
            patch.object(node_answer_output, "get_node_durations", return_value={}),
            patch.object(node_answer_output, "get_total_duration", return_value=1.25),
            patch.object(node_answer_output, "step_1_check_answer", return_value=False),
            patch.object(node_answer_output, "step_2_load_prompt", return_value="prompt"),
            patch.object(node_answer_output, "step_3_create_answer", side_effect=create_answer),
            patch.object(node_answer_output, "step_4_extract_images_url", return_value=[]),
            patch.object(node_answer_output, "step_4_5_extract_sources", return_value=[]),
            patch.object(node_answer_output, "save_chat_message"),
            patch.object(node_answer_output, "push_to_session") as push,
        ):
            result = node_answer_output.node_answer_output(state)

        final_calls = [
            call
            for call in push.call_args_list
            if call.args[1] == SSEEvent.FINAL
        ]
        self.assertEqual(len(final_calls), 1)
        self.assertEqual(final_calls[0].args[2]["warnings"], [WARNING])
        self.assertEqual(result["warnings"], [WARNING])

    def test_history_write_passes_warnings_to_mongo_facade(self):
        from app.query_process.agent.nodes import node_answer_output

        state = {
            "session_id": "session-1",
            "answer": "answer",
            "warnings": [dict(WARNING)],
        }

        with patch.object(node_answer_output, "save_chat_message") as save:
            node_answer_output.step_5_write_history(state)

        self.assertEqual(save.call_args.kwargs["warnings"], [WARNING])

    def test_mongo_document_stores_warnings(self):
        from app.clients import mongo_history_utils_new

        inserted = []

        class Collection:
            def insert_one(self, document):
                inserted.append(document)
                return SimpleNamespace(inserted_id="message-1")

        mongo_tool = SimpleNamespace(chat_message=Collection())
        with patch.object(
            mongo_history_utils_new,
            "get_history_mongo_tool",
            return_value=mongo_tool,
        ):
            mongo_history_utils_new.save_chat_message(
                session_id="session-1",
                role="assistant",
                text="answer",
                warnings=[dict(WARNING)],
            )

        self.assertEqual(inserted[0]["warnings"], [WARNING])

    def test_mongo_modules_do_not_connect_at_import_time(self):
        module_names = [
            "app.clients.mongo_history_utils",
            "app.clients.mongo_history_utils_new",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                previous = sys.modules.pop(module_name, None)
                try:
                    with patch("pymongo.MongoClient") as mongo_client:
                        importlib.import_module(module_name)
                    mongo_client.assert_not_called()
                finally:
                    sys.modules.pop(module_name, None)
                    if previous is not None:
                        sys.modules[module_name] = previous

    def test_history_serializer_preserves_warnings_and_defaults_old_records(self):
        from app.query_process.agent import main_graph

        module_name = "app.query_process.api.query_server"
        previous = sys.modules.pop(module_name, None)
        try:
            with patch.object(main_graph, "query_app", object(), create=True):
                query_server = importlib.import_module(module_name)

            current = query_server._serialize_history_record(
                {
                    "_id": "message-1",
                    "session_id": "session-1",
                    "warnings": [dict(WARNING)],
                }
            )
            legacy = query_server._serialize_history_record(
                {
                    "_id": "message-2",
                    "session_id": "session-1",
                }
            )
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

        self.assertEqual(current["warnings"], [WARNING])
        self.assertEqual(legacy["warnings"], [])


if __name__ == "__main__":
    unittest.main()
