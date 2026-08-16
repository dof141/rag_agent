import unittest

from app.conf.embedding_config import EmbeddingConfig
from app.embedding.interface import EmbeddingRateLimitError, EmbeddingResponseError
from app.embedding.siliconflow_adapter import SiliconFlowEmbeddingAdapter


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class RecordingHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def config(**overrides):
    values = {
        "adapter": "siliconflow",
        "model": "BAAI/bge-m3",
        "dimension": 2,
        "output_type": "dense",
        "batch_size": 2,
        "request_timeout": 20.0,
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "sf-secret",
        "bge_m3_path": None,
        "bge_m3": "BAAI/bge-m3",
        "bge_device": "cpu",
        "bge_fp16": False,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


class SiliconFlowEmbeddingTest(unittest.TestCase):
    def test_batches_and_restores_input_order(self):
        client = RecordingHttpClient(
            [
                FakeResponse(
                    {
                        "data": [
                            {"index": 1, "embedding": [3, 4]},
                            {"index": 0, "embedding": [1, 2]},
                        ]
                    }
                ),
                FakeResponse({"data": [{"index": 0, "embedding": [5, 6]}]}),
            ]
        )
        adapter = SiliconFlowEmbeddingAdapter(config(batch_size=2), http_client=client)

        result = adapter.embed_documents(["first", "second", "third"])

        self.assertEqual(result, {"dense": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})
        self.assertEqual(
            [call["json"]["input"] for call in client.calls],
            [["first", "second"], ["third"]],
        )
        self.assertEqual(client.calls[0]["url"], "https://api.siliconflow.cn/v1/embeddings")
        self.assertNotIn("sparse", result)

    def test_rejects_count_and_dimension_mismatch(self):
        for payload in (
            {"data": []},
            {"data": [{"index": 0, "embedding": [1.0]}]},
        ):
            adapter = SiliconFlowEmbeddingAdapter(
                config(dimension=2),
                http_client=RecordingHttpClient([FakeResponse(payload)]),
            )
            with self.assertRaises(EmbeddingResponseError):
                adapter.embed_documents(["text"])

    def test_sanitizes_provider_errors(self):
        adapter = SiliconFlowEmbeddingAdapter(
            config(api_key="must-not-appear"),
            http_client=RecordingHttpClient(
                [
                    FakeResponse(
                        {"error": "must-not-appear"},
                        status_code=429,
                        text="must-not-appear",
                    )
                ]
            ),
        )
        with self.assertRaises(EmbeddingRateLimitError) as raised:
            adapter.embed_documents(["text"])
        self.assertNotIn("must-not-appear", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
