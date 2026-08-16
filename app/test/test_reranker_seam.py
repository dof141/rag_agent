import importlib
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from app.reranker import (
    RerankItem,
    RerankOutcome,
    RerankerConfigurationError,
    RerankerError,
    RerankerProvider,
    RerankerRequestError,
    RerankerResponseError,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RecordingHttpClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if callable(self.responder):
            return self.responder(json)
        if isinstance(self.responder, Exception):
            raise self.responder
        return self.responder


def make_config(**overrides):
    from app.conf.reranker_config import RerankerConfig

    values = {
        "adapter": "http",
        "model": "BAAI/bge-reranker-v2-m3",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "test-key",
        "request_timeout": 8.0,
        "max_documents": 20,
        "bge_reranker_large": "BAAI/bge-reranker-v2-m3",
        "bge_reranker_device": "cpu",
        "bge_reranker_fp16": False,
    }
    values.update(overrides)
    return RerankerConfig(**values)


def load_config_with_environment(environment):
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("dotenv.load_dotenv") as load_dotenv,
    ):
        sys.modules.pop("app.conf.reranker_config", None)
        module = importlib.import_module("app.conf.reranker_config")
    return module, load_dotenv


class RerankerConfigTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("app.conf.reranker_config", None)

    def test_reads_remote_and_local_settings(self):
        module, load_dotenv = load_config_with_environment(
            {
                "RERANKER_ADAPTER": "custom-http",
                "RERANKER_MODEL": "example/reranker",
                "RERANKER_BASE_URL": "https://reranker.example/v1///",
                "SILICONFLOW_API_KEY": "fallback-key",
                "RERANKER_REQUEST_TIMEOUT": "3.5",
                "RERANKER_MAX_DOCUMENTS": "7",
                "BGE_RERANKER_LARGE": "local/reranker",
                "BGE_RERANKER_DEVICE": "cuda",
                "BGE_RERANKER_FP16": "true",
            }
        )

        self.assertEqual(
            module.reranker_config,
            module.RerankerConfig(
                adapter="custom-http",
                model="example/reranker",
                base_url="https://reranker.example/v1",
                api_key="fallback-key",
                request_timeout=3.5,
                max_documents=7,
                bge_reranker_large="local/reranker",
                bge_reranker_device="cuda",
                bge_reranker_fp16=True,
            ),
        )
        load_dotenv.assert_called_once_with()

    def test_uses_defaults_and_enforces_document_minimum(self):
        module, _ = load_config_with_environment(
            {
                "RERANKER_API_KEY": "primary-key",
                "SILICONFLOW_API_KEY": "fallback-key",
                "RERANKER_MAX_DOCUMENTS": "0",
            }
        )

        config = module.reranker_config
        self.assertEqual(config.adapter, "http")
        self.assertEqual(config.model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(config.base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(config.api_key, "primary-key")
        self.assertEqual(config.request_timeout, 8.0)
        self.assertEqual(config.max_documents, 1)
        self.assertIsNone(config.bge_reranker_large)
        self.assertEqual(config.bge_reranker_device, "cpu")
        self.assertFalse(config.bge_reranker_fp16)

        with self.assertRaises(FrozenInstanceError):
            config.adapter = "local"


class RerankerSeamTest(unittest.TestCase):
    def test_env_example_contains_remote_reranker_defaults(self):
        from dotenv import dotenv_values

        env_path = Path(__file__).resolve().parents[2] / ".env.example"
        values = dotenv_values(env_path)
        expected = {
            "RERANKER_ADAPTER": "http",
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANKER_BASE_URL": "https://api.siliconflow.cn/v1",
            "RERANKER_REQUEST_TIMEOUT": "8",
            "RERANKER_MAX_DOCUMENTS": "20",
        }

        self.assertEqual(
            {key: values.get(key) for key in expected},
            expected,
        )
        self.assertEqual(
            values.get("RERANKER_API_KEY"),
            "your_siliconflow_api_key_here",
        )


class RerankerInterfaceTest(unittest.TestCase):
    def test_result_types_are_immutable_and_have_stable_defaults(self):
        item = RerankItem(index=2, score=0.75)
        outcome = RerankOutcome(items=[item])

        self.assertEqual(item.index, 2)
        self.assertEqual(item.score, 0.75)
        self.assertEqual(outcome.items, [item])
        self.assertFalse(outcome.degraded)
        self.assertIsNone(outcome.warning_code)
        self.assertIsNone(outcome.warning_message)

        with self.assertRaises(FrozenInstanceError):
            item.score = None
        with self.assertRaises(FrozenInstanceError):
            outcome.degraded = True

    def test_stable_errors_share_the_reranker_base_error(self):
        self.assertTrue(issubclass(RerankerConfigurationError, RerankerError))
        self.assertTrue(issubclass(RerankerRequestError, RerankerError))
        self.assertTrue(issubclass(RerankerResponseError, RerankerError))

    def test_provider_protocol_exposes_rerank(self):
        self.assertTrue(callable(RerankerProvider.rerank))


class HttpRerankerProviderTest(unittest.TestCase):
    def test_maps_response_indexes_and_sends_generic_request(self):
        from app.reranker.http_adapter import HttpRerankerProvider

        client = RecordingHttpClient(
            FakeResponse(
                {
                    "id": "rerank-1",
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ],
                }
            )
        )
        provider = HttpRerankerProvider(make_config(), http_client=client)

        items = provider.rerank("问题", ["文档一", "文档二"])

        self.assertEqual(
            [(item.index, item.score) for item in items],
            [(1, 0.9), (0, 0.2)],
        )
        self.assertEqual(
            client.calls,
            [
                {
                    "url": "https://api.siliconflow.cn/v1/rerank",
                    "headers": {
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                    },
                    "json": {
                        "model": "BAAI/bge-reranker-v2-m3",
                        "query": "问题",
                        "documents": ["文档一", "文档二"],
                        "return_documents": False,
                        "top_n": 2,
                    },
                    "timeout": 8.0,
                }
            ],
        )

    def test_requires_remote_api_key(self):
        from app.reranker.http_adapter import HttpRerankerProvider

        with self.assertRaisesRegex(RerankerConfigurationError, "RERANKER_API_KEY"):
            HttpRerankerProvider(make_config(api_key=None))

    def test_rejects_non_success_status_and_network_errors(self):
        from app.reranker.http_adapter import HttpRerankerProvider

        cases = [
            (
                RecordingHttpClient(
                    FakeResponse({"message": "limited"}, status_code=429, text="limited")
                ),
                "429",
            ),
            (RecordingHttpClient(TimeoutError("timed out")), "timed out"),
        ]
        for client, message in cases:
            with self.subTest(message=message):
                provider = HttpRerankerProvider(make_config(), http_client=client)
                with self.assertRaisesRegex(RerankerRequestError, message):
                    provider.rerank("问题", ["文档"])

    def test_rejects_invalid_json_and_malformed_results(self):
        from app.reranker.http_adapter import HttpRerankerProvider

        malformed = [
            (ValueError("bad json"), "JSON"),
            ({}, "results"),
            ({"results": []}, "数量"),
            (
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.8},
                    ]
                },
                "重复",
            ),
            (
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.8},
                    ]
                },
                "越界",
            ),
            (
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1},
                    ]
                },
                "relevance_score",
            ),
            (
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": float("nan")},
                    ]
                },
                "有限",
            ),
        ]

        for payload, message in malformed:
            with self.subTest(message=message):
                provider = HttpRerankerProvider(
                    make_config(),
                    http_client=RecordingHttpClient(FakeResponse(payload)),
                )
                with self.assertRaisesRegex(RerankerResponseError, message):
                    provider.rerank("问题", ["文档一", "文档二"])


class RerankerFactoryAndServiceTest(unittest.TestCase):
    def tearDown(self):
        from app.reranker import factory

        factory.clear_reranker_provider_cache()

    def test_http_factory_does_not_import_local_heavy_dependencies(self):
        from app.reranker import factory

        sys.modules.pop("app.reranker.local_adapter", None)
        before = set(sys.modules)
        provider = factory.get_reranker_provider(make_config())
        loaded = set(sys.modules) - before

        self.assertEqual(provider.__class__.__name__, "HttpRerankerProvider")
        self.assertNotIn("app.reranker.local_adapter", sys.modules)
        self.assertFalse(
            [
                name
                for name in loaded
                if name == "FlagEmbedding" or name.startswith("FlagEmbedding.")
            ]
        )

    def test_factory_caches_provider_and_rejects_unknown_adapter(self):
        from app.reranker import factory

        config = make_config()
        first = factory.get_reranker_provider(config)
        second = factory.get_reranker_provider(config)
        self.assertIs(first, second)

        with self.assertRaisesRegex(RerankerConfigurationError, "RERANKER_ADAPTER"):
            factory.get_reranker_provider(make_config(adapter="unknown"))

    def test_local_adapter_loads_once_and_sorts_scores(self):
        from app.reranker.local_adapter import LocalBgeRerankerProvider

        class FakeModel:
            def compute_score(self, pairs, normalize=True):
                self.pairs = pairs
                self.normalize = normalize
                return [0.2, 0.9]

        model = FakeModel()
        calls = []

        def model_factory(**kwargs):
            calls.append(kwargs)
            return model

        provider = LocalBgeRerankerProvider(
            make_config(adapter="local"),
            model_factory=model_factory,
        )
        self.assertEqual(calls, [])

        items = provider.rerank("问题", ["一", "二"])

        self.assertEqual([(item.index, item.score) for item in items], [(1, 0.9), (0, 0.2)])
        self.assertEqual(model.pairs, [["问题", "一"], ["问题", "二"]])
        self.assertTrue(model.normalize)
        self.assertEqual(len(calls), 1)

    def test_local_adapter_initializes_model_once_under_concurrency(self):
        from app.reranker.local_adapter import LocalBgeRerankerProvider

        model = object()
        calls = []

        def model_factory(**kwargs):
            time.sleep(0.02)
            calls.append(kwargs)
            return model

        provider = LocalBgeRerankerProvider(
            make_config(adapter="local"),
            model_factory=model_factory,
        )
        models = []
        threads = [
            threading.Thread(target=lambda: models.append(provider.get_model()))
            for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(result is model for result in models))

    def test_service_caps_candidates_and_returns_provider_result(self):
        from app.reranker.service import rerank_texts

        class RecordingProvider:
            def __init__(self):
                self.documents = None

            def rerank(self, query, documents):
                self.documents = documents
                return [RerankItem(index=1, score=0.8), RerankItem(index=0, score=0.4)]

        provider = RecordingProvider()
        outcome = rerank_texts(
            "问题",
            ["一", "二", "三"],
            config=make_config(max_documents=2),
            provider=provider,
        )

        self.assertEqual(provider.documents, ["一", "二"])
        self.assertFalse(outcome.degraded)
        self.assertEqual([item.index for item in outcome.items], [1, 0])

    def test_service_degrades_to_original_first_ten_on_provider_error(self):
        from app.reranker.service import rerank_texts

        class FailingProvider:
            def rerank(self, query, documents):
                raise TimeoutError("timed out")

        outcome = rerank_texts(
            "问题",
            [str(index) for index in range(15)],
            config=make_config(max_documents=20),
            provider=FailingProvider(),
        )

        self.assertTrue(outcome.degraded)
        self.assertEqual([item.index for item in outcome.items], list(range(10)))
        self.assertTrue(all(item.score is None for item in outcome.items))
        self.assertEqual(outcome.warning_code, "reranker_degraded")
        self.assertEqual(
            outcome.warning_message,
            "重排序服务暂时不可用，本次回答已使用原始检索顺序生成",
        )

    def test_service_handles_empty_documents_without_provider_call(self):
        from app.reranker.service import rerank_texts

        class UnexpectedProvider:
            def rerank(self, query, documents):
                raise AssertionError("provider should not be called")

        outcome = rerank_texts(
            "问题",
            [],
            config=make_config(),
            provider=UnexpectedProvider(),
        )

        self.assertEqual(outcome.items, [])
        self.assertFalse(outcome.degraded)


if __name__ == "__main__":
    unittest.main()
