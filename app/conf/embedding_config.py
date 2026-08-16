from dataclasses import dataclass, field
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class EmbeddingConfig:
    adapter: str
    model: str
    dimension: int
    output_type: str
    batch_size: int
    request_timeout: float
    base_url: str
    api_key: str | None = field(repr=False)
    bge_m3_path: str | None
    bge_m3: str | None
    bge_device: str
    bge_fp16: bool


embedding_config = EmbeddingConfig(
    adapter=os.getenv("EMBEDDING_ADAPTER") or "local",
    model=os.getenv("EMBEDDING_MODEL") or "qwen3.7-text-embedding",
    dimension=int(os.getenv("EMBEDDING_DIMENSION") or 1024),
    output_type=os.getenv("EMBEDDING_OUTPUT_TYPE") or "dense&sparse",
    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE") or 10),
    request_timeout=float(os.getenv("EMBEDDING_REQUEST_TIMEOUT") or 20),
    base_url=(
        os.getenv("EMBEDDING_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    ),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3") or "BAAI/bge-m3",
    bge_device=os.getenv("BGE_DEVICE") or "cpu",
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true"),
)
