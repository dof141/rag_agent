import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.auth.models import User
from app.eval.run_pipeline import build_dataset, run_one, summarize_samples
from app.eval.run_ragas import to_ragas_rows
from app.query_process.engine import QueryEvent


class RecordingEvaluationEngine:
    def __init__(self):
        self.confirm_requests = []
        self.event_batches = {
            "request-1": [
                QueryEvent(
                    "confirmation_required",
                    {"candidates": [{"id": 42, "item_name": "代数"}]},
                )
            ],
            "request-2": [
                QueryEvent("delta", {"delta": "答案"}),
                QueryEvent(
                    "final",
                    {
                        "answer": "答案",
                        "sources": [{"content": "依据"}],
                        "warnings": [],
                    },
                ),
            ],
        }

    async def ask(self, user, request):
        return SimpleNamespace(request_id="request-1", session_id=request.session_id)

    async def confirm(self, user, request):
        self.confirm_requests.append(request)
        return SimpleNamespace(request_id="request-2", session_id=request.session_id)

    async def events(self, user, request_id):
        async def iterator():
            for event in self.event_batches[request_id]:
                yield event

        return iterator()


class QueryEvaluationTest(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_resumes_with_candidate_id_string(self):
        engine = RecordingEvaluationEngine()
        user = User(id="user-a", username="a", role="admin")

        sample = await run_one(
            "判别式是什么？",
            engine=engine,
            user=user,
            session_id="eval:algebra",
        )

        self.assertEqual(engine.confirm_requests[0].candidate_id, "42")
        self.assertIsInstance(engine.confirm_requests[0].candidate_id, str)
        self.assertEqual(sample["meta"]["status"], "success")
        self.assertEqual(sample["retrieved_contexts"], ["依据"])

    async def test_runner_crash_is_written_as_failure_instead_of_skipped(self):
        async def crashing_runner(*args, **kwargs):
            raise RuntimeError("private detail")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden = root / "golden.jsonl"
            output = root / "output.jsonl"
            golden.write_text(
                json.dumps(
                    {
                        "question": "问题",
                        "ground_truth": "答案",
                        "item_names": ["主题"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows, summary = await build_dataset(
                golden,
                output,
                engine=object(),
                user=User(id="user-a", username="a", role="admin"),
                runner=crashing_runner,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["meta"]["status"], "failure")
            self.assertEqual(summary["failure"], 1)
            self.assertEqual(summary["failure_rate"], 1.0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

    def test_ragas_rows_keep_failure_and_empty_evidence_by_default(self):
        samples = [
            {
                "user_input": "成功",
                "retrieved_contexts": ["依据"],
                "response": "回答",
                "reference": "标准",
                "meta": {"status": "success"},
            },
            {
                "user_input": "失败",
                "retrieved_contexts": [],
                "response": "",
                "reference": "标准",
                "meta": {"status": "failure"},
            },
            {
                "user_input": "空依据",
                "retrieved_contexts": [],
                "response": "知识库中没有足够依据回答该问题。",
                "reference": "",
                "meta": {"status": "empty_evidence"},
            },
        ]

        rows = to_ragas_rows(samples)
        summary = summarize_samples(samples)

        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["failure"], 1)
        self.assertEqual(summary["empty_evidence"], 1)
        self.assertAlmostEqual(summary["failure_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
