from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class RerankerConfig:
    adapter: str
    model: str
    base_url: str
    api_key: str | None
    request_timeout: float
    max_documents: int
    bge_reranker_large: str | None
    bge_reranker_device: str
    bge_reranker_fp16: bool


reranker_config = RerankerConfig(
    adapter=os.getenv("RERANKER_ADAPTER") or "http",
    model=os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3",
    base_url=(
        os.getenv("RERANKER_BASE_URL")
        or "https://api.siliconflow.cn/v1"
    ).rstrip("/"),
    api_key=os.getenv("RERANKER_API_KEY") or os.getenv("SILICONFLOW_API_KEY"),
    request_timeout=float(os.getenv("RERANKER_REQUEST_TIMEOUT") or 8),
    max_documents=max(1, int(os.getenv("RERANKER_MAX_DOCUMENTS") or 20)),
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE") or "cpu",
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true"),
)
