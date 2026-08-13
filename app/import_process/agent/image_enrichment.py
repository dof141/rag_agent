import base64
import hashlib
import json
import mimetypes
import sqlite3
import threading
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from app.conf.image_enrichment_config import (
    ImageEnrichmentConfig,
    image_enrichment_config,
)
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_vlm_client
from app.utils.rate_limit_utils import apply_api_rate_limit


ImageTarget = tuple[str, str, tuple[str, str]]


def build_cache_key(
    image_path: str | Path,
    context: tuple[str, str],
    model: str,
    prompt_version: str,
) -> str:
    image_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    context_hash = hashlib.sha256(
        json.dumps(context, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = f"{image_hash}:{context_hash}:{model}:{prompt_version}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def should_enrich_image(image_path: str | Path) -> bool:
    """Only reject images that are certainly unable to add visual information."""
    try:
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            rgba.thumbnail((128, 128))
            pixels = list(rgba.get_flattened_data())
    except Exception as e:
        logger.warning(f"图片无法解码，跳过 VLM 描述：{image_path}，原因：{e}")
        return False

    visible = [(r, g, b) for r, g, b, alpha in pixels if alpha > 0]
    if not visible:
        logger.info(f"图片全透明，跳过 VLM 描述：{image_path}")
        return False

    alpha_values = [alpha for _, _, _, alpha in pixels]
    if min(alpha_values) != max(alpha_values):
        return True

    extrema = [
        (min(pixel[channel] for pixel in visible), max(pixel[channel] for pixel in visible))
        for channel in range(3)
    ]
    if all(high - low <= 2 for low, high in extrema):
        logger.info(f"图片有效像素近乎纯色，跳过 VLM 描述：{image_path}")
        return False
    return True


class ImageSummaryCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_summary_cache (
                cache_key TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return connection

    def get(self, cache_key: str) -> str | None:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                row = connection.execute(
                    "SELECT summary FROM image_summary_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        return row[0] if row else None

    def put(self, cache_key: str, summary: str) -> None:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO image_summary_cache(cache_key, summary)
                    VALUES (?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET summary = excluded.summary
                    """,
                    (cache_key, summary),
                )


class _ProcessVlmLimiter:
    def __init__(self):
        self._request_times = deque()
        self._rate_lock = threading.Lock()
        self._semaphores = {}
        self._semaphore_lock = threading.Lock()

    def invoke(self, client, messages, config: ImageEnrichmentConfig):
        with self._semaphore_lock:
            semaphore = self._semaphores.setdefault(
                config.max_concurrency,
                threading.BoundedSemaphore(config.max_concurrency),
            )
        with semaphore:
            with self._rate_lock:
                apply_api_rate_limit(
                    request_times=self._request_times,
                    max_requests=config.requests_per_minute,
                )
            return client.invoke(messages)


_PROCESS_VLM_LIMITER = _ProcessVlmLimiter()


class ImageEnricher:
    def __init__(
        self,
        config: ImageEnrichmentConfig = image_enrichment_config,
        vlm_factory: Callable[..., object] = get_vlm_client,
    ):
        self.config = config
        self._vlm_factory = vlm_factory
        self._cache = ImageSummaryCache(config.cache_path)

    def summarize_images(
        self,
        targets: Iterable[ImageTarget],
        document_name: str,
    ) -> dict[str, str]:
        summaries = {}
        pending = []
        for image_file, image_path, context in targets:
            if not should_enrich_image(image_path):
                summaries[image_file] = ""
                continue
            cache_key = build_cache_key(
                image_path,
                context,
                self.config.model,
                self.config.prompt_version,
            )
            cached = None
            if self.config.cache_enabled:
                try:
                    cached = self._cache.get(cache_key)
                except Exception as e:
                    logger.warning(f"图片摘要缓存读取失败，按未命中继续：{e}")
            if cached is not None:
                summaries[image_file] = cached
            else:
                pending.append((image_file, image_path, context, cache_key))

        if not pending:
            return summaries

        try:
            client = self._vlm_factory(
                model=self.config.model,
                timeout=self.config.request_timeout,
            )
        except Exception as e:
            logger.warning(f"VLM 客户端不可用，所有待处理图片保留空摘要：{e}")
            summaries.update({item[0]: "" for item in pending})
            return summaries
        for offset in range(0, len(pending), self.config.batch_size):
            batch = pending[offset : offset + self.config.batch_size]
            try:
                batch_summaries = self._summarize_batch(client, batch, document_name)
            except Exception as e:
                logger.warning(f"批量 VLM 图片描述失败，降级为逐图调用：{e}")
                batch_summaries = {
                    item[0]: self._summarize_single(client, item, document_name)
                    for item in batch
                }

            for image_file, _, _, cache_key in batch:
                summary = batch_summaries.get(image_file, "")
                summaries[image_file] = summary
                if self.config.cache_enabled and summary:
                    try:
                        self._cache.put(cache_key, summary)
                    except Exception as e:
                        logger.warning(f"图片摘要缓存写入失败，忽略缓存继续：{e}")

        return summaries

    def _summarize_batch(self, client, batch, document_name: str) -> dict[str, str]:
        image_ids = [item[0] for item in batch]
        content = [
            {
                "type": "text",
                "text": (
                    "请分别总结后续图片，并只返回严格 JSON："
                    '{"summaries":[{"image_id":"文件名","summary":"50字内中文摘要"}]}。'
                    "image_id 必须使用给出的原始文件名，不得遗漏或增加图片。"
                ),
            }
        ]
        for image_file, image_path, context, _ in batch:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"image_id: {image_file}\n"
                            + load_prompt(
                                "image_summary",
                                root_folder=document_name,
                                image_content=context,
                            )
                        ),
                    },
                    self._image_message_part(image_path),
                ]
            )

        response = self._invoke(client, [{"role": "user", "content": content}])
        payload = self._parse_json(response.content)
        entries = payload.get("summaries")
        if not isinstance(entries, list):
            raise ValueError("VLM 批量响应缺少 summaries 数组")
        result = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("VLM 批量响应包含无效摘要项")
            image_id = entry.get("image_id")
            summary = entry.get("summary")
            if image_id not in image_ids or not isinstance(summary, str):
                raise ValueError(f"VLM 批量响应包含未知或无效 image_id：{image_id}")
            if image_id in result:
                raise ValueError(f"VLM 批量响应重复 image_id：{image_id}")
            result[image_id] = self._normalize_summary(summary)
        if set(result) != set(image_ids):
            raise ValueError("VLM 批量响应未覆盖全部图片")
        return result

    def _summarize_single(self, client, item, document_name: str) -> str:
        image_file, image_path, context, _ = item
        try:
            content = [
                self._image_message_part(image_path),
                {
                    "type": "text",
                    "text": load_prompt(
                        "image_summary",
                        root_folder=document_name,
                        image_content=context,
                    ),
                },
            ]
            response = self._invoke(client, [{"role": "user", "content": content}])
            return self._normalize_summary(response.content)
        except Exception as e:
            logger.warning(f"图片 VLM 描述失败，保留空摘要：{image_file}，原因：{e}")
            return ""

    def _invoke(self, client, messages):
        return _PROCESS_VLM_LIMITER.invoke(client, messages, self.config)

    @staticmethod
    def _image_message_part(image_path: str) -> dict:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("VLM 批量响应必须是 JSON 对象")
        return payload

    @staticmethod
    def _normalize_summary(summary: str) -> str:
        return " ".join(summary.strip().splitlines())


_default_enricher = None
_default_enricher_lock = threading.Lock()


def get_image_enricher() -> ImageEnricher:
    global _default_enricher
    with _default_enricher_lock:
        if _default_enricher is None:
            _default_enricher = ImageEnricher()
        return _default_enricher


def summarize_images(targets: Iterable[ImageTarget], document_name: str) -> dict[str, str]:
    return get_image_enricher().summarize_images(targets, document_name)
