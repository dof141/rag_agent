import sys
import unittest
from unittest.mock import patch

from app.import_process.agent.nodes import node_item_name_recognition
from app.import_process.agent import main_graph
from app.import_process.errors import ImportTaskError
from app.import_process.runtime import ImportRuntime
from app.import_process.task_repository import TaskRepositoryError
from app.import_process.agent.nodes.node_generate_embeddings import (
    create_generate_embeddings_node,
)


class ImportFailureSemanticsTest(unittest.TestCase):
    def test_persistence_preflight_runs_before_node_side_effect(self):
        side_effects = []

        def write_vector(_state):
            side_effects.append("vector-written")

        def fail_persistence(_state):
            raise TaskRepositoryError("database unavailable")

        wrapped = main_graph.wrap_import_node(
            write_vector,
            stage="vector_store",
            public_message="向量库写入失败",
            before_node=fail_persistence,
        )

        with self.assertRaises(ImportTaskError) as raised:
            wrapped({"task_id": "task-a"})

        self.assertEqual(side_effects, [])
        self.assertEqual(raised.exception.stage, "task_persistence")

    def test_graph_node_failure_has_stable_public_stage(self):
        def fail(_state):
            raise RuntimeError("private parser details")

        wrapped = main_graph.wrap_import_node(
            fail,
            stage="document_parse",
            public_message="文档解析失败",
        )

        with self.assertRaises(ImportTaskError) as raised:
            wrapped({})

        self.assertEqual(raised.exception.stage, "document_parse")
        self.assertEqual(raised.exception.public_message, "文档解析失败")
        self.assertNotIn("private parser details", str(raised.exception))

    def test_graph_node_preserves_existing_import_task_error(self):
        expected = ImportTaskError("embedding", "文档向量生成失败")

        def fail(_state):
            raise expected

        wrapped = main_graph.wrap_import_node(
            fail,
            stage="document_parse",
            public_message="文档解析失败",
        )

        with self.assertRaises(ImportTaskError) as raised:
            wrapped({})

        self.assertIs(raised.exception, expected)

    def test_item_name_timeout_falls_back_to_file_stem(self):
        class FailingLlm:
            def invoke(self, messages):
                raise TimeoutError("provider timed out")

        with patch.object(
            node_item_name_recognition,
            "get_llm_client",
            return_value=FailingLlm(),
        ) as get_client, patch.object(node_item_name_recognition, "logger"):
            item_name = node_item_name_recognition.step_3_call_llm(
                context="document context",
                file_title="course-notes.docx",
            )

        self.assertEqual(item_name, "course-notes")
        get_client.assert_called_once_with(
            json_mode=False,
            timeout=node_item_name_recognition.ITEM_NAME_TIMEOUT_SECONDS,
        )

    def test_item_name_node_has_no_embedding_or_storage_side_effect(self):
        self.assertFalse(hasattr(node_item_name_recognition, "step_5_generate_embeddings"))
        self.assertFalse(hasattr(node_item_name_recognition, "step_6_save_to_vector_db"))
        self.assertFalse(hasattr(node_item_name_recognition, "generate_embeddings"))
        self.assertFalse(hasattr(node_item_name_recognition, "get_milvus_client"))

        before_modules = set(sys.modules)
        state = {
            "task_id": "task-item-name",
            "chunks": [{"title": "Title", "content": "Content"}],
            "file_title": "notes.docx",
        }

        with (
            patch.object(node_item_name_recognition, "add_running_task"),
            patch.object(node_item_name_recognition, "add_done_task"),
            patch.object(node_item_name_recognition, "logger"),
            patch.object(
                node_item_name_recognition,
                "step_3_call_llm",
                return_value="notes",
            ),
        ):
            result = node_item_name_recognition.node_item_name_recognition(state)

        self.assertEqual(result["item_name"], "notes")
        self.assertEqual(result["chunks"][0]["item_name"], "notes")
        after_modules = set(sys.modules)
        self.assertFalse(
            [
                name
                for name in after_modules - before_modules
                if name.startswith("pymilvus") or name.startswith("pymilvus.model")
            ]
        )

    def test_embedding_failure_becomes_public_import_task_error(self):
        class FailingEmbedding:
            def embed_documents(self, texts):
                raise RuntimeError("provider secret body")

        runtime = ImportRuntime(embedding=FailingEmbedding(), vector_store=object())
        node = create_generate_embeddings_node(runtime.embedding)

        with self.assertRaises(ImportTaskError) as raised:
            node({"item_name": "notes", "chunks": [{"content": "Content"}]})

        self.assertEqual(raised.exception.stage, "embedding")
        self.assertEqual(raised.exception.public_message, "文档向量生成失败")


if __name__ == "__main__":
    unittest.main()
