import importlib
import os
import sys
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


def load_config_with_environment(environment):
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("dotenv.load_dotenv") as load_dotenv,
    ):
        sys.modules.pop("app.conf.reranker_config", None)
        module = importlib.import_module("app.conf.reranker_config")
    return module, load_dotenv


class RerankerConfigTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
