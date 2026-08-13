import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from app.conf.image_enrichment_config import (
    ImageEnrichmentConfig,
    load_image_enrichment_config,
)
from app.import_process.agent.image_enrichment import (
    ImageEnricher,
    build_cache_key,
    should_enrich_image,
)


class FakeVlm:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


class ImageEnrichmentTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _patterned_image(self, name, colors=None):
        colors = colors or [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 0, 255),
        ]
        path = self.root / name
        image = Image.new("RGB", (2, 2))
        image.putdata(colors)
        image.save(path)
        return path

    def _config(self, **overrides):
        values = {
            "requests_per_minute": 1000,
            "batch_size": 6,
            "max_concurrency": 3,
            "cache_enabled": True,
            "cache_path": str(self.root / "image-cache.sqlite3"),
            "prompt_version": "v1",
            "request_timeout": 5.0,
            "model": "test-vlm",
        }
        values.update(overrides)
        return ImageEnrichmentConfig(**values)

    def test_cache_key_changes_with_context_model_and_prompt_version(self):
        path = self._patterned_image("notes.png")

        base = build_cache_key(path, ("before", "after"), "model-a", "v1")

        self.assertNotEqual(
            base,
            build_cache_key(path, ("different", "after"), "model-a", "v1"),
        )
        self.assertNotEqual(
            base,
            build_cache_key(path, ("before", "after"), "model-b", "v1"),
        )
        self.assertNotEqual(
            base,
            build_cache_key(path, ("before", "after"), "model-a", "v2"),
        )

    def test_useful_tiny_image_is_not_filtered_by_size(self):
        path = self._patterned_image("tiny-note.png")

        self.assertTrue(should_enrich_image(path))

    def test_transparent_uniform_and_undecodable_images_skip_vlm(self):
        transparent = self.root / "transparent.png"
        Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(transparent)
        uniform = self.root / "uniform.png"
        Image.new("RGB", (4, 4), (250, 250, 250)).save(uniform)
        broken = self.root / "broken.png"
        broken.write_bytes(b"not an image")

        self.assertFalse(should_enrich_image(transparent))
        self.assertFalse(should_enrich_image(uniform))
        self.assertFalse(should_enrich_image(broken))

    def test_transparent_background_with_single_color_strokes_is_retained(self):
        path = self.root / "handwriting.png"
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        image.putpixel((1, 1), (0, 0, 0, 255))
        image.putpixel((2, 2), (0, 0, 0, 255))
        image.save(path)

        self.assertTrue(should_enrich_image(path))

    def test_batch_response_maps_summaries_by_image_id_and_uses_cache(self):
        first = self._patterned_image("first.png")
        second = self._patterned_image(
            "second.png",
            [(0, 255, 0), (0, 0, 0), (255, 255, 255), (255, 0, 255)],
        )
        vlm = FakeVlm(
            [
                '{"summaries": ['
                '{"image_id": "second.png", "summary": "课程截图"},'
                '{"image_id": "first.png", "summary": "手写笔记"}'
                "]}"
            ]
        )
        enricher = ImageEnricher(
            config=self._config(),
            vlm_factory=lambda **_: vlm,
        )
        targets = [
            ("first.png", str(first), ("上文一", "下文一")),
            ("second.png", str(second), ("上文二", "下文二")),
        ]

        first_result = enricher.summarize_images(targets, "course.md")
        second_result = enricher.summarize_images(targets, "course.md")

        self.assertEqual(
            first_result,
            {"first.png": "手写笔记", "second.png": "课程截图"},
        )
        self.assertEqual(second_result, first_result)
        self.assertEqual(len(vlm.messages), 1)

    def test_invalid_batch_response_falls_back_to_single_image_calls(self):
        first = self._patterned_image("first.png")
        second = self._patterned_image("second.png")
        vlm = FakeVlm(["not-json", "第一张摘要", "第二张摘要"])
        enricher = ImageEnricher(
            config=self._config(cache_enabled=False),
            vlm_factory=lambda **_: vlm,
        )

        result = enricher.summarize_images(
            [
                ("first.png", str(first), ("上文一", "下文一")),
                ("second.png", str(second), ("上文二", "下文二")),
            ],
            "course.md",
        )

        self.assertEqual(
            result,
            {"first.png": "第一张摘要", "second.png": "第二张摘要"},
        )
        self.assertEqual(len(vlm.messages), 3)

    def test_vlm_client_initialization_failure_keeps_empty_summaries(self):
        image = self._patterned_image("course.png")

        def unavailable_vlm(**kwargs):
            raise RuntimeError("VLM is not configured")

        enricher = ImageEnricher(
            config=self._config(cache_enabled=False),
            vlm_factory=unavailable_vlm,
        )

        result = enricher.summarize_images(
            [("course.png", str(image), ("上文", "下文"))],
            "course.md",
        )

        self.assertEqual(result, {"course.png": ""})

    def test_prompt_loading_failure_keeps_empty_summaries(self):
        image = self._patterned_image("course.png")
        vlm = FakeVlm([])
        enricher = ImageEnricher(
            config=self._config(cache_enabled=False),
            vlm_factory=lambda **_: vlm,
        )

        with patch(
            "app.import_process.agent.image_enrichment.load_prompt",
            side_effect=FileNotFoundError("prompt missing"),
        ):
            result = enricher.summarize_images(
                [("course.png", str(image), ("上文", "下文"))],
                "course.md",
            )

        self.assertEqual(result, {"course.png": ""})

    def test_invalid_vlm_numeric_configuration_falls_back_to_defaults(self):
        config = load_image_enrichment_config(
            {
                "VLM_REQUESTS_PER_MINUTE": "invalid",
                "VLM_BATCH_SIZE": "invalid",
                "VLM_MAX_CONCURRENCY": "invalid",
                "VLM_TIMEOUT": "invalid",
            }
        )

        self.assertEqual(config.requests_per_minute, 9)
        self.assertEqual(config.batch_size, 6)
        self.assertEqual(config.max_concurrency, 3)
        self.assertEqual(config.request_timeout, 60.0)

    def test_cache_read_and_write_failures_do_not_block_enrichment(self):
        image = self._patterned_image("course.png")
        vlm = FakeVlm(
            [
                '{"summaries": ['
                '{"image_id": "course.png", "summary": "课程截图"}'
                "]}"
            ]
        )
        enricher = ImageEnricher(
            config=self._config(),
            vlm_factory=lambda **_: vlm,
        )

        class BrokenCache:
            def get(self, cache_key):
                raise OSError("cache is locked")

            def put(self, cache_key, summary):
                raise OSError("cache is read only")

        enricher._cache = BrokenCache()

        result = enricher.summarize_images(
            [("course.png", str(image), ("上文", "下文"))],
            "course.md",
        )

        self.assertEqual(result, {"course.png": "课程截图"})


if __name__ == "__main__":
    unittest.main()
