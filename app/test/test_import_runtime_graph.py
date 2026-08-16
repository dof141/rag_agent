import unittest

from app.import_process.agent.nodes.node_generate_embeddings import (
    create_generate_embeddings_node,
)
from app.import_process.agent.nodes.node_import_vector_store import (
    create_vector_import_node,
)
from app.import_process.agent.state import create_default_state
from app.import_process.runtime import ImportRuntime


class RecordingEmbedding:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(texts)
        return self.result


class RecordingVectorStore:
    def __init__(self):
        self.documents = []

    def import_document(self, document):
        self.documents.append(document)
        return type("Result", (), {"item_count": 1, "chunk_count": len(document.chunks)})()


class ImportRuntimeGraphTest(unittest.TestCase):
    def test_embedding_and_import_use_frozen_runtime_outside_state(self):
        embedding = RecordingEmbedding(
            {
                "dense": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            }
        )
        vector_store = RecordingVectorStore()
        runtime = ImportRuntime(embedding=embedding, vector_store=vector_store)
        state = create_default_state(
            task_id="task-a",
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name="manual",
            chunks=[
                {
                    "content": "first",
                    "title": "A",
                    "parent_title": "manual.pdf",
                    "part": 1,
                },
                {
                    "content": "second",
                    "title": "B",
                    "parent_title": "manual.pdf",
                    "part": 2,
                },
            ],
        )

        state = create_generate_embeddings_node(runtime.embedding)(state)
        state = create_vector_import_node(runtime.vector_store)(state)

        self.assertEqual(len(embedding.calls), 1)
        self.assertEqual(embedding.calls[0][0], "manual")
        self.assertEqual(len(vector_store.documents), 1)
        self.assertEqual(vector_store.documents[0].user_id, "user-a")
        self.assertFalse({"runtime", "api_key", "client"}.intersection(state))
        self.assertEqual(state["import_result"], {"item_count": 1, "chunk_count": 2})

    def test_sparse_vectors_are_mapped_when_embedding_provider_returns_them(self):
        embedding = RecordingEmbedding(
            {
                "dense": [[1.0], [2.0]],
                "sparse": [{1: 0.1}, {2: 0.2}],
            }
        )
        vector_store = RecordingVectorStore()
        state = create_default_state(
            task_id="task-a",
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name="manual",
            chunks=[
                {
                    "content": "first",
                    "title": "A",
                    "parent_title": "manual.pdf",
                    "part": 1,
                },
            ],
        )

        state = create_generate_embeddings_node(embedding)(state)
        create_vector_import_node(vector_store)(state)

        document = vector_store.documents[0]
        self.assertEqual(document.item_sparse_vector, {1: 0.1})
        self.assertEqual(document.chunks[0].sparse_vector, {2: 0.2})


if __name__ == "__main__":
    unittest.main()
