from functools import lru_cache
import os

from app.retrieval.interface import Retrieval


@lru_cache(maxsize=1)
def get_retrieval() -> Retrieval:
    adapter = (os.getenv("RETRIEVAL_ADAPTER") or "local").lower()
    if adapter != "local":
        raise ValueError(f"Unsupported RETRIEVAL_ADAPTER: {adapter}")

    from app.retrieval.local_adapter import LocalRetrievalAdapter

    return LocalRetrievalAdapter()
