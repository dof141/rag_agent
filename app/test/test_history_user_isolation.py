import asyncio
import unittest
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.models import User
from app.auth.repository import UserRepository
from app.auth.security import JwtTokenService, PasswordHasher
from app.persistence.sqlite_database import SQLiteDatabase
from app.query_process.agent.state import create_query_default_state


warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)


def _matches(document, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(_matches(document, branch) for branch in expected):
                return False
            continue
        actual = document.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$exists" in expected and (key in document) != expected["$exists"]:
                return False
            continue
        if actual != expected:
            return False
    return True


class MemoryCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, key, direction):
        self.documents.sort(key=lambda item: item.get(key, 0), reverse=direction < 0)
        return self

    def limit(self, limit):
        self.documents = self.documents[:limit]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class MemoryCollection:
    def __init__(self, documents=()):
        self.documents = [deepcopy(document) for document in documents]
        self.indexes = []
        self.last_delete_filter = None
        self.last_update_filter = None
        self.last_pipeline = None

    def create_index(self, fields, **kwargs):
        self.indexes.append((fields, kwargs))

    def insert_one(self, document):
        stored = deepcopy(document)
        stored.setdefault("_id", ObjectId())
        self.documents.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    def find(self, query):
        return MemoryCursor(
            [document for document in self.documents if _matches(document, query)]
        )

    def delete_many(self, query):
        self.last_delete_filter = deepcopy(query)
        retained = [document for document in self.documents if not _matches(document, query)]
        deleted_count = len(self.documents) - len(retained)
        self.documents = retained
        return SimpleNamespace(deleted_count=deleted_count)

    def update_one(self, query, update):
        self.last_update_filter = deepcopy(query)
        modified_count = 0
        for document in self.documents:
            if _matches(document, query):
                document.update(deepcopy(update["$set"]))
                modified_count = 1
                break
        return SimpleNamespace(modified_count=modified_count)

    def update_many(self, query, update):
        self.last_update_filter = deepcopy(query)
        modified_count = 0
        for document in self.documents:
            if _matches(document, query):
                document.update(deepcopy(update["$set"]))
                modified_count += 1
        return SimpleNamespace(modified_count=modified_count)

    def aggregate(self, pipeline):
        self.last_pipeline = deepcopy(pipeline)
        return []


class HistoryFacadeIsolationTest(unittest.TestCase):
    def modules(self):
        from app.clients import mongo_history_utils, mongo_history_utils_new

        return mongo_history_utils, mongo_history_utils_new

    def test_save_requires_user_id_and_stores_owner(self):
        for module in self.modules():
            with self.subTest(module=module.__name__):
                collection = MemoryCollection()
                tool = SimpleNamespace(chat_message=collection)
                with patch.object(module, "get_history_mongo_tool", return_value=tool):
                    with self.assertRaises(TypeError):
                        module.save_chat_message(
                            session_id="shared",
                            role="user",
                            text="missing owner",
                        )
                    module.save_chat_message(
                        user_id="user-a",
                        session_id="shared",
                        role="user",
                        text="owned",
                    )

                self.assertEqual(collection.documents[0]["user_id"], "user-a")

    def test_recent_messages_hide_other_users_and_legacy_rows(self):
        documents = [
            {"_id": ObjectId(), "user_id": "user-a", "session_id": "shared", "text": "a", "ts": 1},
            {"_id": ObjectId(), "user_id": "user-b", "session_id": "shared", "text": "b", "ts": 2},
            {"_id": ObjectId(), "session_id": "shared", "text": "legacy", "ts": 3},
        ]
        for module in self.modules():
            with self.subTest(module=module.__name__):
                collection = MemoryCollection(documents)
                tool = SimpleNamespace(chat_message=collection)
                with patch.object(module, "get_history_mongo_tool", return_value=tool):
                    messages = module.get_recent_messages("user-a", "shared", limit=10)

                self.assertEqual([message["text"] for message in messages], ["a"])

    def test_delete_and_message_updates_include_owner(self):
        message_id = ObjectId()
        for module in self.modules():
            with self.subTest(module=module.__name__):
                collection = MemoryCollection(
                    [
                        {"_id": message_id, "user_id": "user-a", "session_id": "shared"},
                        {"_id": ObjectId(), "user_id": "user-b", "session_id": "shared"},
                    ]
                )
                tool = SimpleNamespace(chat_message=collection)
                with patch.object(module, "get_history_mongo_tool", return_value=tool):
                    module.update_message_item_names(
                        "user-a", [str(message_id)], ["course-a"]
                    )
                    self.assertEqual(collection.last_update_filter["user_id"], "user-a")

                    module.save_chat_message(
                        user_id="user-a",
                        session_id="shared",
                        role="user",
                        text="updated",
                        message_id=str(message_id),
                    )
                    self.assertEqual(collection.last_update_filter["user_id"], "user-a")

                    deleted = module.clear_history("user-a", "shared")

                self.assertEqual(deleted, 1)
                self.assertEqual(
                    collection.last_delete_filter,
                    {"user_id": "user-a", "session_id": "shared"},
                )
                self.assertEqual(collection.documents[0]["user_id"], "user-b")

    def test_clearing_all_history_only_deletes_current_user(self):
        from app.clients import mongo_history_utils_new

        collection = MemoryCollection(
            [
                {"user_id": "user-a", "session_id": "one"},
                {"user_id": "user-a", "session_id": "two"},
                {"user_id": "user-b", "session_id": "one"},
                {"session_id": "legacy"},
            ]
        )
        with patch.object(
            mongo_history_utils_new,
            "get_history_mongo_tool",
            return_value=SimpleNamespace(chat_message=collection),
        ):
            count = mongo_history_utils_new.clear_history("user-a")
            with self.assertRaises(ValueError):
                mongo_history_utils_new.clear_history("user-a", "")

        self.assertEqual(count, 2)
        self.assertEqual(collection.last_delete_filter, {"user_id": "user-a"})
        self.assertEqual(
            [document.get("user_id") for document in collection.documents],
            ["user-b", None],
        )

    def test_session_summary_starts_with_owner_match(self):
        from app.clients import mongo_history_utils_new

        collection = MemoryCollection()
        with patch.object(
            mongo_history_utils_new,
            "get_history_mongo_tool",
            return_value=SimpleNamespace(chat_message=collection),
        ):
            mongo_history_utils_new.get_all_sessions_summary("user-a")

        self.assertEqual(collection.last_pipeline[0], {"$match": {"user_id": "user-a"}})

    def test_history_indexes_start_with_owner_and_session(self):
        for module in self.modules():
            with self.subTest(module=module.__name__):
                chat_messages = MemoryCollection()
                import_tasks = MemoryCollection()

                class Database:
                    def __getitem__(self, name):
                        return chat_messages if name == "chat_message" else import_tasks

                class Client:
                    def __getitem__(self, name):
                        return Database()

                with patch.object(module, "MongoClient", return_value=Client()):
                    module.HistoryMongoTool()

                self.assertEqual(
                    chat_messages.indexes[0][0],
                    [("user_id", 1), ("session_id", 1), ("ts", -1)],
                )

    def test_migration_claims_only_unowned_rows_after_exact_confirmation(self):
        from app.tools.assign_legacy_history import assign_legacy_history

        collection = MemoryCollection(
            [
                {"session_id": "legacy-a"},
                {"session_id": "legacy-b"},
                {"user_id": None, "session_id": "explicit-null"},
                {"user_id": "user-b", "session_id": "owned"},
            ]
        )

        rejected = assign_legacy_history(
            collection,
            user_id="user-a",
            confirm="wrong",
        )
        self.assertEqual(rejected, 0)
        self.assertIsNone(collection.last_update_filter)

        assigned = assign_legacy_history(
            collection,
            user_id="user-a",
            confirm="ASSIGN_LEGACY_HISTORY",
        )

        self.assertEqual(assigned, 2)
        self.assertEqual(collection.last_update_filter, {"user_id": {"$exists": False}})
        self.assertEqual(
            [document.get("user_id") for document in collection.documents],
            ["user-a", "user-a", None, "user-b"],
        )


class QueryOwnershipPropagationTest(unittest.IsolatedAsyncioTestCase):
    async def test_query_engine_places_current_user_in_graph_state(self):
        from app.query_process.engine import QueryEngine, QueryRequest

        class Graph:
            def __init__(self):
                self.input = None

            async def ainvoke(self, graph_input, config):
                self.input = graph_input
                return {"answer": "answer", "warnings": []}

        graph = Graph()
        services = SimpleNamespace(
            settings=SimpleNamespace(get_snapshot=lambda user_id: object()),
            query_runtime_factory=lambda snapshot: object(),
        )
        engine = QueryEngine(services, graph_builder=lambda runtime, checkpointer: graph)
        with patch("app.query_process.engine.get_task_result", return_value="answer"):
            await engine.ask(
                User(id="user-a", username="a", role="admin"),
                QueryRequest(query="question", session_id="shared"),
            )

        self.assertEqual(graph.input["user_id"], "user-a")

    def test_nodes_require_and_forward_owner(self):
        from app.query_process.agent.nodes import node_answer_output, node_item_name_confirm

        self.assertIn("user_id", create_query_default_state())
        state = create_query_default_state(
            user_id="user-a",
            request_id="request-1",
            session_id="shared",
            original_query="question",
            is_stream=False,
        )
        with (
            patch.object(node_item_name_confirm, "add_running_task"),
            patch.object(node_item_name_confirm, "add_done_task"),
            patch.object(node_item_name_confirm, "get_recent_messages", return_value=[]) as recent,
            patch.object(
                node_item_name_confirm,
                "step_3_llm_item_name_and_rewrite_query",
                return_value=node_item_name_confirm.GoodsResponse(
                    item_names=[], rewritten_query="question"
                ),
            ),
            patch.object(node_item_name_confirm, "save_chat_message") as save,
        ):
            node_item_name_confirm._node_item_name_confirm(state, SimpleNamespace())

        recent.assert_called_once_with("user-a", "shared", limit=10)
        self.assertEqual(save.call_args.kwargs["user_id"], "user-a")

        answer_state = create_query_default_state(
            user_id="user-a",
            session_id="shared",
            answer="answer",
        )
        with patch.object(node_answer_output, "save_chat_message") as answer_save:
            node_answer_output.step_5_write_history(answer_state)
        self.assertEqual(answer_save.call_args.kwargs["user_id"], "user-a")

        with self.assertRaises(ValueError):
            node_answer_output.step_5_write_history(
                {"session_id": "shared", "answer": "answer"}
            )


class HistoryHttpIsolationTest(unittest.TestCase):
    def setUp(self):
        from app.query_process.api.router import create_query_router

        self.temp_dir = TemporaryDirectory()
        database = SQLiteDatabase(Path(self.temp_dir.name) / "rag.db")
        database.initialize()
        self.users = UserRepository(database, PasswordHasher())
        self.user = self.users.ensure_admin("admin", "password")
        self.tokens = JwtTokenService("unit-test-signing-secret", 60)
        services = SimpleNamespace(users=self.users, tokens=self.tokens)
        app = FastAPI()
        app.include_router(create_query_router(services, engine=SimpleNamespace()))
        self.client = TestClient(app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def auth(self):
        return {"Authorization": f"Bearer {self.tokens.issue(self.user.id)}"}

    def test_all_history_routes_require_authentication(self):
        responses = (
            self.client.get("/history/shared"),
            self.client.delete("/history/shared"),
            self.client.get("/api/history/sessions"),
            self.client.delete("/api/history/sessions"),
        )
        self.assertEqual([response.status_code for response in responses], [401] * 4)

    def test_history_routes_forward_current_user(self):
        from app.query_process.api import router

        with (
            patch.object(router, "get_task_history", return_value={"items": []}) as history,
            patch.object(router, "delete_session", return_value=1) as delete,
            patch.object(router, "get_all_sessions_summary", return_value=[]) as summary,
            patch.object(router, "clear_history", return_value=2) as clear,
        ):
            self.assertEqual(
                self.client.get("/history/shared", headers=self.auth()).status_code,
                200,
            )
            self.client.delete("/history/shared", headers=self.auth())
            self.client.get("/api/history/sessions", headers=self.auth())
            self.client.delete("/api/history/sessions", headers=self.auth())

        history.assert_called_once_with(self.user.id, "shared", 20)
        delete.assert_called_once_with(self.user.id, "shared")
        summary.assert_called_once_with(self.user.id)
        clear.assert_called_once_with(self.user.id)


if __name__ == "__main__":
    unittest.main()
