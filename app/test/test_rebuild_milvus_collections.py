import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from app.tools import rebuild_milvus_collections


class RecordingStore:
    calls = []

    def __init__(self, config):
        self.config = config

    def rebuild_collections(self):
        self.calls.append(self.config)


class FakeSettings:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_snapshot(self, user_id):
        return self.snapshot


class FakeServices:
    def __init__(self, snapshot):
        self.settings = FakeSettings(snapshot)
        self.initialized = False

    def initialize_database_only(self):
        self.initialized = True


class RebuildMilvusCollectionsTest(unittest.TestCase):
    def setUp(self):
        RecordingStore.calls = []

    def test_requires_confirmation_before_deleting(self):
        services = FakeServices(
            SimpleNamespace(vector_store_type="milvus", milvus="milvus-config")
        )
        with patch.object(
            rebuild_milvus_collections,
            "load_application_services",
            return_value=services,
        ), patch.object(rebuild_milvus_collections, "MilvusVectorStore", RecordingStore):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                missing = rebuild_milvus_collections.main(["--user-id", "user-a"])
                wrong = rebuild_milvus_collections.main(
                    ["--user-id", "user-a", "--confirm", "wrong"]
                )

        self.assertNotEqual(missing, 0)
        self.assertEqual(wrong, 2)
        self.assertEqual(RecordingStore.calls, [])

    def test_confirmed_rebuild_uses_runtime_snapshot(self):
        services = FakeServices(
            SimpleNamespace(vector_store_type="milvus", milvus="milvus-config")
        )
        with patch.object(
            rebuild_milvus_collections,
            "load_application_services",
            return_value=services,
        ), patch.object(rebuild_milvus_collections, "MilvusVectorStore", RecordingStore):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = rebuild_milvus_collections.main(
                    [
                        "--user-id",
                        "user-a",
                        "--confirm",
                        rebuild_milvus_collections.CONFIRMATION,
                    ]
                )

        self.assertEqual(code, 0)
        self.assertTrue(services.initialized)
        self.assertEqual(RecordingStore.calls, ["milvus-config"])


if __name__ == "__main__":
    unittest.main()
