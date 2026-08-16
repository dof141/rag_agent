import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.vector_store.factory import create_vector_store


class VectorStoreFactoryTest(unittest.TestCase):
    def snapshot(self, store):
        return SimpleNamespace(
            vector_store_type=store,
            qdrant="qdrant-config" if store == "qdrant" else None,
            milvus="milvus-config" if store == "milvus" else None,
        )

    def test_factory_selects_exactly_one_adapter_without_fallback(self):
        qdrant = Mock(return_value="qdrant")
        milvus = Mock(return_value="milvus")

        selected = create_vector_store(
            self.snapshot("qdrant"),
            qdrant_factory=qdrant,
            milvus_factory=milvus,
        )

        self.assertEqual(selected, "qdrant")
        qdrant.assert_called_once_with("qdrant-config")
        milvus.assert_not_called()

        qdrant.side_effect = RuntimeError("target unavailable")
        with self.assertRaisesRegex(RuntimeError, "target unavailable"):
            create_vector_store(
                self.snapshot("qdrant"),
                qdrant_factory=qdrant,
                milvus_factory=milvus,
            )
        milvus.assert_not_called()


if __name__ == "__main__":
    unittest.main()
