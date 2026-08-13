import unittest
from unittest.mock import Mock, patch

from app.import_process.agent.nodes import (
    node_bge_embedding,
    node_import_milvus,
    node_item_name_recognition,
)


class ImportFailureSemanticsTest(unittest.TestCase):
    def test_item_name_timeout_falls_back_to_file_stem(self):
        llm = Mock()
        llm.invoke.side_effect = TimeoutError("provider timed out")

        with patch.object(
            node_item_name_recognition,
            "get_llm_client",
            return_value=llm,
        ) as get_client:
            item_name = node_item_name_recognition.step_3_call_llm(
                context="document context",
                file_title="course-notes.docx",
            )

        self.assertEqual(item_name, "course-notes")
        get_client.assert_called_once_with(
            json_mode=False,
            timeout=node_item_name_recognition.ITEM_NAME_TIMEOUT_SECONDS,
        )

    def test_item_name_embedding_failure_escapes_node(self):
        state = {
            "task_id": "task-item-name",
            "chunks": [{"title": "Title", "content": "Content"}],
            "file_title": "notes.docx",
        }

        with (
            patch.object(node_item_name_recognition, "add_running_task"),
            patch.object(node_item_name_recognition, "add_done_task") as done,
            patch.object(
                node_item_name_recognition,
                "step_3_call_llm",
                return_value="notes",
            ),
            patch.object(
                node_item_name_recognition,
                "step_5_generate_embeddings",
                side_effect=RuntimeError("embedding unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                node_item_name_recognition.node_item_name_recognition(state)

        done.assert_not_called()

    def test_chunk_embedding_failure_escapes_node(self):
        state = {
            "task_id": "task-embedding",
            "chunks": [
                {"item_name": "notes", "content": "Content", "title": "Title"}
            ],
        }

        with (
            patch.object(node_bge_embedding, "add_running_task"),
            patch.object(node_bge_embedding, "add_done_task") as done,
            patch.object(
                node_bge_embedding,
                "generate_embeddings",
                side_effect=RuntimeError("embedding unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                node_bge_embedding.node_bge_embedding(state)

        done.assert_not_called()

    def test_milvus_failure_escapes_node(self):
        state = {
            "task_id": "task-milvus",
            "chunks": [
                {
                    "item_name": "notes",
                    "content": "Content",
                    "dense_vector": [0.1],
                    "sparse_vector": {1: 0.2},
                }
            ],
        }

        with (
            patch.object(node_import_milvus, "add_running_task"),
            patch.object(node_import_milvus, "add_done_task") as done,
            patch.object(
                node_import_milvus,
                "get_milvus_client",
                side_effect=RuntimeError("milvus unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "milvus unavailable"):
                node_import_milvus.node_import_milvus(state)

        done.assert_not_called()


if __name__ == "__main__":
    unittest.main()
