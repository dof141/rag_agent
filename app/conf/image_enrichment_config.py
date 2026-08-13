import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

from app.core.logger import logger


load_dotenv()


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int, name: str) -> int:
    try:
        return max(1, int(value)) if value is not None else default
    except (TypeError, ValueError):
        logger.warning(f"{name} 配置无效，使用默认值 {default}")
        return default


def _as_float(value: str | None, default: float, name: str) -> float:
    try:
        parsed = float(value) if value is not None else default
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        logger.warning(f"{name} 配置无效，使用默认值 {default}")
        return default


@dataclass(frozen=True)
class ImageEnrichmentConfig:
    requests_per_minute: int
    batch_size: int
    max_concurrency: int
    cache_enabled: bool
    cache_path: str
    prompt_version: str
    request_timeout: float
    model: str


def load_image_enrichment_config(
    environ: Mapping[str, str] = os.environ,
) -> ImageEnrichmentConfig:
    return ImageEnrichmentConfig(
        requests_per_minute=_as_int(
            environ.get("VLM_REQUESTS_PER_MINUTE"), 9, "VLM_REQUESTS_PER_MINUTE"
        ),
        batch_size=_as_int(environ.get("VLM_BATCH_SIZE"), 6, "VLM_BATCH_SIZE"),
        max_concurrency=_as_int(
            environ.get("VLM_MAX_CONCURRENCY"), 3, "VLM_MAX_CONCURRENCY"
        ),
        cache_enabled=_as_bool(environ.get("VLM_CACHE_ENABLED"), True),
        cache_path=environ.get("VLM_CACHE_PATH")
        or "output/cache/image_summaries.sqlite3",
        prompt_version=environ.get("VLM_PROMPT_VERSION") or "v1",
        request_timeout=_as_float(environ.get("VLM_TIMEOUT"), 60.0, "VLM_TIMEOUT"),
        model=environ.get("VL_MODEL") or "qwen-vl-plus",
    )


image_enrichment_config = load_image_enrichment_config()
