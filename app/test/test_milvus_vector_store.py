import unittest

from app.vector_store.config import MilvusVectorStoreConfig
from app.vector_store.interface import VectorChunk, VectorDocument, VectorImportResult
from app.vector_store.milvus_adapter import MilvusVectorStore


class FakeDataType:
    INT64 = "INT64"
    INT8 = "INT8"
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"


class RecordingSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class RecordingIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class RecordingMilvusClient:
    def __init__(self):
        self.schemas = {}
        self.deletes = []
        self.inserts = []
        self.deleted_collections = []

    def has_collection(self, collection_name):
        return collection_name in self.schemas

    def create_schema(self, **kwargs):
        return RecordingSchema()

    def prepare_index_params(self):
        return RecordingIndexParams()

    def create_collection(self, **kwargs):
        self.schemas[kwargs["collection_name"]] = kwargs["schema"]

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def insert(self, **kwargs):
        self.inserts.append(kwargs)
        return {"inserted_count": len(kwargs["data"])}

    def flush(self, **kwargs):
        pass

    def load_collection(self, **kwargs):
        pass

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)
        self.schemas.pop(collection_name, None)


class MilvusVectorStoreTest(unittest.TestCase):
    def config(self):
        return MilvusVectorStoreConfig(
            url="http://milvus.example",
            token="token",
            item_collection="items",
            chunks_collection="chunks",
            dimension=2,
            batch_size=1,
        )

    def document(self):
        return VectorDocument(
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name='manual "quoted"',
            item_dense_vector=(0.1, 0.2),
            item_sparse_vector={1: 0.5},
            chunks=(
                VectorChunk(
                    index=0,
                    content="first",
                    title="A",
                    parent_title="manual.pdf",
                    part=1,
                    dense_vector=(1.0, 2.0),
                    sparse_vector={2: 0.4},
                ),
            ),
        )

    def test_milvus_schema_and_filters_include_user_and_document(self):
        client = RecordingMilvusClient()
        store = MilvusVectorStore(self.config(), client=client, data_type=FakeDataType)

        result = store.import_document(self.document())

        self.assertEqual(result, VectorImportResult(item_count=1, chunk_count=1))
        fields = {
            collection: {field["field_name"] for field in schema.fields}
            for collection, schema in client.schemas.items()
        }
        self.assertTrue({"user_id", "document_id"}.issubset(fields["items"]))
        self.assertTrue({"user_id", "document_id"}.issubset(fields["chunks"]))
        self.assertIn('user_id == "user-a"', client.deletes[0]["filter"])
        self.assertIn('item_name == "manual \\"quoted\\""', client.deletes[0]["filter"])
        self.assertIn('user_id == "user-a"', client.deletes[1]["filter"])
        self.assertIn('document_id == "doc-a"', client.deletes[1]["filter"])
        self.assertEqual(client.inserts[0]["data"][0]["sparse_vector"], {1: 0.5})
        self.assertEqual(client.inserts[1]["data"][0]["sparse_vector"], {2: 0.4})

    def test_rebuild_collections_requires_explicit_call(self):
        client = RecordingMilvusClient()
        store = MilvusVectorStore(self.config(), client=client, data_type=FakeDataType)

        store.rebuild_collections()

        self.assertEqual(client.deleted_collections, ["items", "chunks"])
        self.assertEqual(set(client.schemas), {"items", "chunks"})


if __name__ == "__main__":
    unittest.main()
