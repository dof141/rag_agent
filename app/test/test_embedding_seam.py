import importlib
import sys
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import patch


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
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
        return self.responder


class RecordingSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class RecordingMilvusClient:
    def __init__(self):
        self.schema = RecordingSchema()

    def has_collection(self, **kwargs):
        return False

    def create_schema(self, **kwargs):
        return self.schema

    def prepare_index_params(self):
        class IndexParams:
            def add_index(self, **kwargs):
                pass

        return IndexParams()

    def create_collection(self, **kwargs):
        pass

    def load_collection(self, **kwargs):
        pass

    def delete(self, **kwargs):
        pass

    def insert(self, **kwargs):
        pass

    def flush(self, **kwargs):
        pass


def make_config(**overrides):
    from app.conf.embedding_config import EmbeddingConfig

    values = {
        "adapter": "dashscope",
        "model": "qwen3.7-text-embedding",
        "dimension": 2,
        "output_type": "dense&sparse",
        "batch_size": 10,
        "request_timeout": 20.0,
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "api_key": "test-key",
        "bge_m3_path": None,
        "bge_m3": "BAAI/bge-m3",
        "bge_device": "cpu",
        "bge_fp16": False,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


def import_node_without_task_storage(module_name):
    task_utils = ModuleType("app.utils.task_utils")
    task_utils.add_running_task = lambda *args, **kwargs: None
    task_utils.add_done_task = lambda *args, **kwargs: None
    pymilvus = ModuleType("pymilvus")
    pymilvus.DataType = type(
        "DataType",
        (),
        {
            "INT64": "INT64",
            "INT8": "INT8",
            "VARCHAR": "VARCHAR",
            "FLOAT_VECTOR": "FLOAT_VECTOR",
            "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
        },
    )
    milvus_utils = ModuleType("app.clients.milvus_utils")
    milvus_utils.get_milvus_client = lambda: None
    with patch.dict(
        sys.modules,
        {
            "app.utils.task_utils": task_utils,
            "app.clients.milvus_utils": milvus_utils,
            "pymilvus": pymilvus,
        },
    ):
        return importlib.import_module(module_name)


class EmbeddingSeamTest(unittest.TestCase):
    def test_01_factory_dashscope_does_not_import_local_heavy_dependencies(self):
        sys.modules.pop("app.embedding.local_adapter", None)
        heavy_prefixes = ("pymilvus.model", "torch")
        before = set(sys.modules)

        from app.embedding import factory

        factory.clear_embedding_provider_cache()
        provider = factory.get_embedding_provider(make_config())
        after = set(sys.modules)

        self.assertEqual(provider.__class__.__name__, "DashScopeEmbeddingProvider")
        self.assertNotIn("app.embedding.local_adapter", after)
        self.assertFalse(
            [
                name
                for name in after - before
                if name == "torch"
                or name.startswith("torch.")
                or name.startswith(heavy_prefixes[0])
            ]
        )

    def test_02_factory_caches_provider_and_can_clear_cache(self):
        from app.embedding import factory

        config = make_config()
        factory.clear_embedding_provider_cache()
        first = factory.get_embedding_provider(config)
        second = factory.get_embedding_provider(config)
        self.assertIs(first, second)

        factory.clear_embedding_provider_cache()
        third = factory.get_embedding_provider(config)
        self.assertIsNot(first, third)

    def test_factory_initializes_provider_once_under_concurrency(self):
        from app.embedding import factory

        config = make_config()
        created = []

        class SlowProvider:
            def __init__(self, selected_config):
                time.sleep(0.02)
                created.append(selected_config)

            def embed_documents(self, texts):
                return {"dense": [], "sparse": []}

        factory.clear_embedding_provider_cache()
        with patch(
            "app.embedding.dashscope_adapter.DashScopeEmbeddingProvider",
            SlowProvider,
        ):
            providers = []
            threads = [
                threading.Thread(
                    target=lambda: providers.append(factory.get_embedding_provider(config))
                )
                for _ in range(5)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(created), 1)
        self.assertTrue(all(provider is providers[0] for provider in providers))

    def test_03_importing_facade_does_not_load_local_heavy_dependencies(self):
        sys.modules.pop("app.lm.embedding_utils", None)
        sys.modules.pop("app.embedding.local_adapter", None)
        before = set(sys.modules)

        module = importlib.import_module("app.lm.embedding_utils")
        after = set(sys.modules)

        self.assertTrue(callable(module.generate_embeddings))
        self.assertTrue(callable(module.get_bge_m3_ef))
        self.assertNotIn("app.embedding.local_adapter", after)
        self.assertFalse(
            [
                name
                for name in after - before
                if name == "torch"
                or name.startswith("torch.")
                or name.startswith("pymilvus.model")
            ]
        )

    def test_dashscope_request_body_and_response_conversion(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider

        response = FakeResponse(
            {
                "output": {
                    "embeddings": [
                        {
                            "text_index": 1,
                            "embedding": [3.0, 4.0],
                            "sparse_embedding": [
                                {"index": 9, "value": 0.9},
                            ],
                        },
                        {
                            "text_index": 0,
                            "embedding": [1.0, 2.0],
                            "sparse_embedding": [
                                {"index": "7", "value": "0.7"},
                            ],
                        },
                    ]
                }
            }
        )
        client = RecordingHttpClient(response)
        provider = DashScopeEmbeddingProvider(config=make_config(), http_client=client)

        result = provider.embed_documents(["first", "second"])

        self.assertEqual(
            result,
            {
                "dense": [[1.0, 2.0], [3.0, 4.0]],
                "sparse": [{7: 0.7}, {9: 0.9}],
            },
        )
        self.assertEqual(
            client.calls,
            [
                {
                    "url": (
                        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
                        "text-embedding/text-embedding"
                    ),
                    "headers": {
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                    },
                    "json": {
                        "model": "qwen3.7-text-embedding",
                        "input": {"texts": ["first", "second"]},
                        "parameters": {
                            "dimension": 2,
                            "output_type": "dense&sparse",
                        },
                    },
                    "timeout": 20.0,
                }
            ],
        )

    def test_local_adapter_loads_model_lazily_and_converts_sparse_vectors(self):
        from app.embedding.local_adapter import LocalBgeM3EmbeddingProvider

        class FakeArray:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, item):
                return FakeArray(self.values[item])

            def tolist(self):
                return list(self.values)

        class FakeSparseMatrix:
            indptr = [0, 2]
            indices = FakeArray([4, 8])
            data = FakeArray([0.25, 0.75])

        class FakeDenseVector:
            def tolist(self):
                return [1, 2]

        class FakeModel:
            def encode_documents(self, texts):
                self.texts = texts
                return {
                    "dense": [FakeDenseVector()],
                    "sparse": FakeSparseMatrix(),
                }

        model = FakeModel()
        model_factory_calls = []

        def model_factory(**kwargs):
            model_factory_calls.append(kwargs)
            return model

        provider = LocalBgeM3EmbeddingProvider(
            make_config(adapter="local"),
            model_factory=model_factory,
        )
        self.assertEqual(model_factory_calls, [])

        result = provider.embed_documents(["local text"])

        self.assertEqual(model.texts, ["local text"])
        self.assertEqual(result, {"dense": [[1.0, 2.0]], "sparse": [{4: 0.25, 8: 0.75}]})
        self.assertEqual(len(model_factory_calls), 1)

    def test_local_adapter_initializes_model_once_under_concurrency(self):
        from app.embedding.local_adapter import LocalBgeM3EmbeddingProvider

        created = []
        model = object()

        def model_factory(**kwargs):
            time.sleep(0.02)
            created.append(kwargs)
            return model

        provider = LocalBgeM3EmbeddingProvider(
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

        self.assertEqual(len(created), 1)
        self.assertTrue(all(result is model for result in models))

    def test_dashscope_batches_requests_and_merges_results(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider

        def respond(request_json):
            embeddings = []
            for text_index, text in enumerate(request_json["input"]["texts"]):
                value = int(text)
                embeddings.append(
                    {
                        "text_index": text_index,
                        "embedding": [float(value), float(value) + 0.5],
                        "sparse_embedding": [
                            {"index": value, "value": float(value) / 10},
                        ],
                    }
                )
            return FakeResponse({"output": {"embeddings": embeddings}})

        client = RecordingHttpClient(respond)
        provider = DashScopeEmbeddingProvider(
            config=make_config(batch_size=2),
            http_client=client,
        )

        result = provider.embed_documents(["0", "1", "2", "3", "4"])

        self.assertEqual(
            [call["json"]["input"]["texts"] for call in client.calls],
            [["0", "1"], ["2", "3"], ["4"]],
        )
        self.assertEqual(
            result["dense"],
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
        )
        self.assertEqual(
            result["sparse"],
            [{0: 0.0}, {1: 0.1}, {2: 0.2}, {3: 0.3}, {4: 0.4}],
        )

    def test_dashscope_rejects_missing_fields_and_count_mismatch(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider
        from app.embedding.interface import EmbeddingResponseError

        malformed_responses = [
            ({"output": {}}, "embeddings"),
            (
                {
                    "output": {
                        "embeddings": [
                            {"embedding": [1.0, 2.0]},
                        ]
                    }
                },
                "sparse_embedding",
            ),
            ({"output": {"embeddings": []}}, "数量"),
        ]

        for payload, expected_message in malformed_responses:
            with self.subTest(payload=payload):
                provider = DashScopeEmbeddingProvider(
                    config=make_config(),
                    http_client=RecordingHttpClient(FakeResponse(payload)),
                )
                with self.assertRaisesRegex(EmbeddingResponseError, expected_message):
                    provider.embed_documents(["text"])

    def test_dashscope_rejects_non_success_status(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider
        from app.embedding.interface import EmbeddingRequestError

        provider = DashScopeEmbeddingProvider(
            config=make_config(),
            http_client=RecordingHttpClient(
                FakeResponse(
                    {"message": "quota exceeded"},
                    status_code=429,
                    text="quota exceeded",
                )
            ),
        )

        with self.assertRaisesRegex(EmbeddingRequestError, "429"):
            provider.embed_documents(["text"])

    def test_dashscope_requires_api_key(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider
        from app.embedding.interface import EmbeddingConfigurationError

        with self.assertRaisesRegex(EmbeddingConfigurationError, "DASHSCOPE_API_KEY"):
            DashScopeEmbeddingProvider(config=make_config(api_key=None))

    def test_dashscope_requires_hybrid_output(self):
        from app.embedding.dashscope_adapter import DashScopeEmbeddingProvider
        from app.embedding.interface import EmbeddingConfigurationError

        with self.assertRaisesRegex(EmbeddingConfigurationError, "dense&sparse"):
            DashScopeEmbeddingProvider(config=make_config(output_type="dense"))

    def test_generate_embeddings_facade_uses_factory_provider(self):
        from app.lm import embedding_utils

        class FakeProvider:
            def __init__(self):
                self.calls = []

            def embed_documents(self, texts):
                self.calls.append(texts)
                return {"dense": [[1.0]], "sparse": [{2: 0.5}]}

        provider = FakeProvider()
        with patch.object(
            embedding_utils,
            "get_embedding_provider",
            return_value=provider,
        ):
            result = embedding_utils.generate_embeddings(["hello"])

        self.assertEqual(provider.calls, [["hello"]])
        self.assertEqual(result, {"dense": [[1.0]], "sparse": [{2: 0.5}]})

    def test_chunk_collection_schema_uses_configured_dimension(self):
        node_import_milvus = import_node_without_task_storage(
            "app.import_process.agent.nodes.node_import_milvus"
        )

        client = RecordingMilvusClient()
        with patch.object(
            node_import_milvus,
            "embedding_config",
            make_config(dimension=768),
            create=True,
        ):
            node_import_milvus.step_2_prepare_collections(client, {})

        dense_field = next(
            field
            for field in client.schema.fields
            if field["field_name"] == "dense_vector"
        )
        self.assertEqual(dense_field["dim"], 768)

    def test_item_name_collection_schema_uses_configured_dimension(self):
        node_item_name_recognition = import_node_without_task_storage(
            "app.import_process.agent.nodes.node_item_name_recognition"
        )

        client = RecordingMilvusClient()
        with (
            patch.object(
                node_item_name_recognition,
                "get_milvus_client",
                return_value=client,
            ),
            patch.object(
                node_item_name_recognition,
                "embedding_config",
                make_config(dimension=768),
                create=True,
            ),
        ):
            node_item_name_recognition.step_6_save_to_vector_db(
                "file.md",
                "item",
                [0.0] * 768,
                {1: 0.5},
            )

        dense_field = next(
            field
            for field in client.schema.fields
            if field["field_name"] == "dense_vector"
        )
        self.assertEqual(dense_field["dim"], 768)


if __name__ == "__main__":
    unittest.main()
