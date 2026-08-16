import unittest
from dataclasses import dataclass

from app.vector_store.config import QdrantVectorStoreConfig
from app.vector_store.interface import (
    VectorChunk,
    VectorDocument,
    VectorImportResult,
    VectorStoreError,
)
from app.vector_store.qdrant_adapter import QdrantVectorStore


@dataclass(frozen=True)
class FakeVectorParams:
    size: int
    distance: str


@dataclass(frozen=True)
class FakeSparseVectorParams:
    modifier: str


@dataclass(frozen=True)
class FakeDocument:
    text: str
    model: str


@dataclass(frozen=True)
class FakePointStruct:
    id: str
    vector: dict
    payload: dict


@dataclass(frozen=True)
class FakeMatchValue:
    value: str


@dataclass(frozen=True)
class FakeFieldCondition:
    key: str
    match: FakeMatchValue


class FakeFilter:
    def __init__(self, must):
        self.must = must
        self.matches = {condition.key: condition.match.value for condition in must}


class FakeModels:
    class Distance:
        COSINE = "COSINE"

    class Modifier:
        IDF = "IDF"

    VectorParams = FakeVectorParams
    SparseVectorParams = FakeSparseVectorParams
    Document = FakeDocument
    PointStruct = FakePointStruct
    MatchValue = FakeMatchValue
    FieldCondition = FakeFieldCondition
    Filter = FakeFilter


class RecordingQdrantClient:
    def __init__(self, *, fail_upsert=False):
        self.fail_upsert = fail_upsert
        self.created = []
        self.deleted = []
        self.upserts = []
        self.existing_collections = set()

    def collection_exists(self, collection_name):
        return collection_name in self.existing_collections

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.existing_collections.add(kwargs["collection_name"])

    def delete(self, **kwargs):
        self.deleted.append(kwargs)

    def upsert(self, **kwargs):
        if self.fail_upsert:
            raise RuntimeError("must-not-leak-provider-body")
        self.upserts.append(kwargs)


class QdrantVectorStoreTest(unittest.TestCase):
    def config(self):
        return QdrantVectorStoreConfig(
            url="https://qdrant.example",
            api_key="qd-secret",
            item_collection="items",
            chunks_collection="chunks",
            dimension=2,
            cloud_inference=True,
            batch_size=1,
        )

    def document(self):
        return VectorDocument(
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name="manual",
            item_dense_vector=(0.1, 0.2),
            item_sparse_vector=None,
            chunks=(
                VectorChunk(
                    index=0,
                    content="first chunk",
                    title="A",
                    parent_title="manual.pdf",
                    part=1,
                    dense_vector=(1.0, 2.0),
                    sparse_vector=None,
                ),
                VectorChunk(
                    index=1,
                    content="second chunk",
                    title="B",
                    parent_title="manual.pdf",
                    part=2,
                    dense_vector=(3.0, 4.0),
                    sparse_vector=None,
                ),
            ),
        )

    def test_import_replaces_one_user_document_when_collections_exist(self):
        client = RecordingQdrantClient()
        client.existing_collections = {"items", "chunks"}
        store = QdrantVectorStore(self.config(), client=client, models_module=FakeModels)

        result = store.import_document(self.document())

        self.assertEqual(result, VectorImportResult(item_count=1, chunk_count=2))
        self.assertEqual(client.created, [])
        self.assertEqual(len(client.deleted), 2)
        self.assertTrue(
            all(
                delete["points_selector"].matches
                == {"user_id": "user-a", "document_id": "doc-a"}
                for delete in client.deleted
            )
        )
        item_point = client.upserts[0]["points"][0]
        self.assertEqual(item_point.vector["bm25"].text, "manual")
        self.assertEqual(item_point.vector["bm25"].model, "Qdrant/bm25")
        self.assertEqual(item_point.payload["user_id"], "user-a")
        self.assertEqual(item_point.payload["document_id"], "doc-a")
        chunk_points = [point for call in client.upserts[1:] for point in call["points"]]
        self.assertEqual([point.payload["chunk_index"] for point in chunk_points], [0, 1])
        self.assertEqual(chunk_points[0].vector["bm25"].text, "first chunk")

    def test_import_creates_hybrid_schema_without_deleting_empty_new_collections(self):
        client = RecordingQdrantClient()
        store = QdrantVectorStore(self.config(), client=client, models_module=FakeModels)

        result = store.import_document(self.document())

        self.assertEqual(result, VectorImportResult(item_count=1, chunk_count=2))
        self.assertEqual({call["collection_name"] for call in client.created}, {"items", "chunks"})
        for call in client.created:
            self.assertEqual(call["vectors_config"]["dense"].size, 2)
            self.assertEqual(call["vectors_config"]["dense"].distance, "COSINE")
            self.assertEqual(call["sparse_vectors_config"]["bm25"].modifier, "IDF")
        self.assertEqual(client.deleted, [])

    def test_failure_is_reported_without_provider_details(self):
        client = RecordingQdrantClient(fail_upsert=True)
        store = QdrantVectorStore(self.config(), client=client, models_module=FakeModels)
        with self.assertRaisesRegex(VectorStoreError, "Qdrant") as raised:
            store.import_document(self.document())
        self.assertNotIn("must-not-leak-provider-body", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
