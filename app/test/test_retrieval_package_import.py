import importlib
import sys
import unittest


class RetrievalPackageImportTest(unittest.TestCase):
    def test_package_imports_without_removed_global_factory(self):
        sys.modules.pop("app.retrieval", None)
        try:
            retrieval = importlib.import_module("app.retrieval")
        except Exception as exc:
            self.fail(f"app.retrieval should import cleanly: {exc}")

        self.assertTrue(hasattr(retrieval, "Retrieval"))
        self.assertTrue(hasattr(retrieval, "VectorSearch"))
        self.assertFalse(hasattr(retrieval, "get_retrieval"))


if __name__ == "__main__":
    unittest.main()
