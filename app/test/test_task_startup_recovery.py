import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.clients.mongo_history_utils import mongo_clean_interrupted_tasks


class TaskStartupRecoveryTest(unittest.TestCase):
    def test_recovery_failure_is_not_reported_as_zero_changes(self):
        with patch(
            "app.clients.mongo_history_utils.get_history_mongo_tool",
            side_effect=OSError("database unavailable"),
        ):
            with self.assertRaises(OSError):
                mongo_clean_interrupted_tasks()

    def test_interrupted_tasks_receive_stable_failure_stage(self):
        calls = []

        class ImportTasks:
            def update_many(self, query, update):
                calls.append((query, update))
                return SimpleNamespace(modified_count=1)

        tool = SimpleNamespace(import_tasks=ImportTasks())
        with patch(
            "app.clients.mongo_history_utils.get_history_mongo_tool",
            return_value=tool,
        ):
            self.assertEqual(mongo_clean_interrupted_tasks(), 1)

        query, update = calls[0]
        self.assertIn("pending", query["status"]["$in"])
        self.assertEqual(update["$set"]["failed_stage"], "startup_recovery")


if __name__ == "__main__":
    unittest.main()
