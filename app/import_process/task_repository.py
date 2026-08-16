class MongoTaskRepository:
    def upsert(self, task_id: str, metadata: dict) -> None:
        from app.clients.mongo_history_utils import mongo_upsert_task

        mongo_upsert_task(task_id, metadata)

    def get(self, task_id: str) -> dict | None:
        from app.clients.mongo_history_utils import mongo_get_task

        return mongo_get_task(task_id)
