import unittest

from app.reranker import RerankItem, rerank_texts
from app.test.test_reranker_seam import make_config


class RerankerFacadeSmokeTest(unittest.TestCase):
    def test_facade_uses_injected_provider_without_loading_real_model(self):
        class FakeProvider:
            def rerank(self, query, documents):
                return [RerankItem(index=0, score=0.8)]

        outcome = rerank_texts(
            "什么是 RRF？",
            ["RRF 是一种排序融合算法"],
            config=make_config(),
            provider=FakeProvider(),
        )

        self.assertFalse(outcome.degraded)
        self.assertEqual(outcome.items, [RerankItem(index=0, score=0.8)])


if __name__ == "__main__":
    unittest.main()
