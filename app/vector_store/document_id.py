from hashlib import sha256
from pathlib import Path
import unicodedata
from uuid import UUID, uuid5


_POINT_NAMESPACE = UUID("651584e0-a681-4b95-aa04-461de39522e8")


def normalize_filename(filename: str) -> str:
    return unicodedata.normalize("NFKC", Path(filename).name).strip().casefold()


def build_document_id(user_id: str, filename: str) -> str:
    normalized = normalize_filename(filename)
    if not user_id or not normalized:
        raise ValueError("user_id 和文件名不能为空")
    return sha256(f"{user_id}\0{normalized}".encode("utf-8")).hexdigest()


def build_point_id(user_id: str, document_id: str, role: str, index: int) -> str:
    if role not in {"item", "chunk"} or index < 0:
        raise ValueError("point role 或 index 无效")
    return str(uuid5(_POINT_NAMESPACE, f"{user_id}\0{document_id}\0{role}\0{index}"))
