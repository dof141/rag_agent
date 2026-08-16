import inspect
import unittest
from dataclasses import MISSING, FrozenInstanceError, fields
from typing import get_type_hints

from app.retrieval import SearchHit, SearchQuery, VectorSearch, VectorSearchError


class VectorSearchContractTest(unittest.TestCase):
    def test_search_query_is_frozen_value_object(self):
        query = SearchQuery(
            text="question",
            dense=(0.1, 0.2),
            sparse={1: 0.5},
        )

        self.assertEqual(
            [(field.name, field.type, field.default) for field in fields(SearchQuery)],
            [
                ("text", str, MISSING),
                ("dense", tuple[float, ...], MISSING),
                ("sparse", dict[int, float] | None, None),
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            query.text = "changed"

    def test_search_hit_is_frozen_value_object(self):
        hit = SearchHit(
            id="hit-1",
            score=0.9,
            content="content",
            item_name="item",
            file_title="file",
            parent_title="parent",
        )

        self.assertEqual(
            [(field.name, field.type, field.default) for field in fields(SearchHit)],
            [
                ("id", str, MISSING),
                ("score", float, MISSING),
                ("content", str, MISSING),
                ("item_name", str, MISSING),
                ("file_title", str, MISSING),
                ("parent_title", str, MISSING),
                ("source", str, "knowledge_base"),
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            hit.score = 0.1

    def test_vector_search_protocol_exposes_search_contract(self):
        search_items_signature = inspect.signature(VectorSearch.search_items)
        search_chunks_signature = inspect.signature(VectorSearch.search_chunks)
        search_items_hints = get_type_hints(VectorSearch.search_items)
        search_chunks_hints = get_type_hints(VectorSearch.search_chunks)

        self.assertTrue(getattr(VectorSearch, "_is_protocol", False))
        self.assertEqual(
            list(search_items_signature.parameters),
            ["self", "query", "top_k"],
        )
        self.assertEqual(
            list(search_chunks_signature.parameters),
            ["self", "query", "item_names", "top_k"],
        )
        self.assertEqual(
            search_items_signature.parameters["top_k"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            search_chunks_signature.parameters["top_k"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(search_items_signature.parameters["top_k"].default, 5)
        self.assertEqual(search_chunks_signature.parameters["top_k"].default, 5)
        self.assertIs(search_items_hints["query"], SearchQuery)
        self.assertIs(search_chunks_hints["query"], SearchQuery)
        self.assertEqual(search_chunks_hints["item_names"], list[str])
        self.assertEqual(
            search_items_hints["return"],
            list[SearchHit],
        )
        self.assertEqual(
            search_chunks_hints["return"],
            list[SearchHit],
        )

    def test_vector_search_error_is_runtime_error(self):
        self.assertTrue(issubclass(VectorSearchError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
