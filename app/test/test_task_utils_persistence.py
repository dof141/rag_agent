import unittest
from unittest.mock import patch

from app.utils import task_utils
from app.import_process.task_repository import TaskRepositoryError


class TaskUtilsPersistenceTest(unittest.TestCase):
    def tearDown(self):
        task_utils.clear_task("task-reload")
        task_utils.clear_task("task-sync")

    def test_reload_restores_failed_stage(self):
        with patch.object(
            task_utils,
            "mongo_get_task",
            return_value={
                "status": "failed",
                "error": "文档解析失败",
                "failed_stage": "document_parse",
            },
        ):
            self.assertEqual(
                task_utils.get_task_result("task-reload", "failed_stage"),
                "document_parse",
            )

    def test_sync_persists_failed_stage(self):
        with (
            patch.object(task_utils, "mongo_get_task", return_value=None),
            patch.object(task_utils, "mongo_upsert_task") as mongo_upsert_task,
        ):
            task_utils.set_task_result("task-sync", "failed_stage", "document_split")

        metadata = mongo_upsert_task.call_args.args[1]
        self.assertEqual(metadata["failed_stage"], "document_split")

    def test_sync_raises_when_mongo_write_fails(self):
        with (
            patch.object(task_utils, "mongo_get_task", return_value=None),
            patch.object(task_utils, "mongo_upsert_task", return_value=False),
        ):
            with self.assertRaises(TaskRepositoryError):
                task_utils.update_task_status("task-sync", "processing")


if __name__ == "__main__":
    unittest.main()
