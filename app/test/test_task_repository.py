import unittest
from unittest.mock import patch

from app.import_process.task_repository import MongoTaskRepository, TaskRepositoryError


class MongoTaskRepositoryTest(unittest.TestCase):
    def test_upsert_raises_when_mongo_does_not_persist_task(self):
        with patch(
            "app.clients.mongo_history_utils.mongo_upsert_task",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "任务状态持久化失败"):
                MongoTaskRepository().upsert("task-a", {"status": "processing"})

    def test_get_distinguishes_mongo_failure_from_missing_task(self):
        with patch(
            "app.clients.mongo_history_utils.mongo_get_task",
            side_effect=OSError("database unavailable"),
        ):
            with self.assertRaises(TaskRepositoryError):
                MongoTaskRepository().get("task-a")


if __name__ == "__main__":
    unittest.main()
