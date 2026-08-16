import unittest

from app.vector_store.document_id import build_document_id, build_point_id
from app.vector_store.interface import VectorChunk, VectorDocument


class VectorStoreInterfaceTest(unittest.TestCase):
    def test_document_id_normalizes_filename_per_user(self):
        first = build_document_id("user-a", "  Manual.PDF ")
        second = build_document_id("user-a", "manual.pdf")
        other_user = build_document_id("user-b", "manual.pdf")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_user)
        self.assertEqual(len(first), 64)

    def test_point_ids_are_stable_and_role_specific(self):
        item = build_point_id("user-a", "doc-a", "item", 0)
        chunk = build_point_id("user-a", "doc-a", "chunk", 0)
        self.assertEqual(item, build_point_id("user-a", "doc-a", "item", 0))
        self.assertNotEqual(item, chunk)

    def test_document_rejects_dimension_or_sparse_mismatch(self):
        chunk = VectorChunk(
            index=0,
            content="content",
            title="title",
            parent_title="manual.pdf",
            part=1,
            dense_vector=(0.1, 0.2),
            sparse_vector=None,
        )
        document = VectorDocument(
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name="manual",
            item_dense_vector=(0.1,),
            item_sparse_vector=None,
            chunks=(chunk,),
        )
        with self.assertRaisesRegex(ValueError, "维度"):
            document.validate(expected_dimension=2, require_sparse=False)

        sparse_document = VectorDocument(
            user_id="user-a",
            document_id="doc-a",
            file_title="manual.pdf",
            item_name="manual",
            item_dense_vector=(0.1, 0.2),
            item_sparse_vector=None,
            chunks=(chunk,),
        )
        with self.assertRaisesRegex(ValueError, "sparse"):
            sparse_document.validate(expected_dimension=2, require_sparse=True)


if __name__ == "__main__":
    unittest.main()
