class TaskRepositoryError(RuntimeError):
    pass


class MongoTaskRepository:
    def upsert(self, task_id: str, metadata: dict) -> None:
        from app.clients.mongo_history_utils import mongo_upsert_task

        if not mongo_upsert_task(task_id, metadata):
            raise TaskRepositoryError("任务状态持久化失败")

    def get(self, task_id: str) -> dict | None:
        from app.clients.mongo_history_utils import mongo_get_task

        try:
            return mongo_get_task(task_id, raise_on_error=True)
        except Exception as exc:
            raise TaskRepositoryError("任务状态查询失败") from exc
