from typing import Optional

from pydantic import BaseModel, Field

from app.clients.mongo_history_utils import clear_history, get_recent_messages


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


class ConfirmRequest(BaseModel):
    session_id: str = Field(min_length=1)
    pending_request_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)


def _serialize_history_record(record: dict) -> dict:
    return {
        "_id": str(record.get("_id")) if record.get("_id") is not None else "",
        "session_id": record.get("session_id", ""),
        "role": record.get("role", ""),
        "text": record.get("text", ""),
        "rewritten_query": record.get("rewritten_query", ""),
        "item_names": record.get("item_names", []),
        "image_urls": record.get("image_urls", []),
        "sources": record.get("sources", []),
        "node_steps": record.get("node_steps", []),
        "total_duration": record.get("total_duration", 0.0),
        "warnings": record.get("warnings", []),
        "ts": record.get("ts"),
    }


def get_task_history(session_id: str, limit: int = 10):
    records = get_recent_messages(session_id, limit=limit)
    return {
        "session_id": session_id,
        "items": [_serialize_history_record(record) for record in records],
    }


def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}
