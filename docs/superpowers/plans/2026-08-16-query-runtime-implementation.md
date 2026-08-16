# User-Scoped Query Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make authenticated questions retrieve the current user's imported knowledge from the configured SiliconFlow + Qdrant runtime, while retaining the configured local BGE-M3 + Milvus path.

**Architecture:** A deep `QueryRuntime` module owns the user-bound `Retrieval` interface. `RetrievalModule` hides embedding and reranking, while Qdrant and Milvus adapters sit at an internal `VectorSearch` seam and enforce `user_id` as a construction-time invariant. `QueryEngine` owns request lifecycle, graph checkpoints, authenticated history and exactly one terminal event.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, qdrant-client, pymilvus, MongoDB, unittest, Vue 3, TypeScript, Vitest.

---

## File Structure

- Create `app/retrieval/models.py`: normalized query and hit value objects.
- Create `app/retrieval/vector_search.py`: storage-neutral `VectorSearch` interface and stable errors.
- Create `app/retrieval/qdrant_adapter.py`: user-bound Qdrant dense + BM25 adapter.
- Create `app/retrieval/milvus_adapter.py`: user-bound Milvus dense + sparse adapter.
- Replace `app/retrieval/local_adapter.py`: deep `RetrievalModule` implementation, without global configuration.
- Replace `app/retrieval/factory.py`: snapshot-driven `QueryRuntime` factory.
- Create `app/query_process/runtime.py`: immutable user-scoped runtime.
- Modify query nodes and `app/query_process/agent/main_graph.py`: runtime closure injection.
- Create `app/query_process/engine.py`: request lifecycle and terminal event owner.
- Create `app/query_process/api/router.py`: authenticated transport adapter.
- Modify Mongo history modules: mandatory user ownership.
- Create `frontend/src/services/sse.ts`: authenticated fetch-stream parser.
- Modify `frontend/src/services/api.ts` and `frontend/src/views/ChatView.vue`: remove mock answers and EventSource.
- Expand `app/eval/*`: evaluate failures instead of filtering them.

### Task 1: Define the Vector Search seam

**Files:**
- Create: `app/retrieval/models.py`
- Create: `app/retrieval/vector_search.py`
- Create: `app/test/test_vector_search_contract.py`
- Modify: `app/retrieval/__init__.py`

- [ ] **Step 1: Write the failing value-object and protocol tests**

```python
from dataclasses import FrozenInstanceError
import unittest

from app.retrieval.models import SearchHit, SearchQuery
from app.retrieval.vector_search import VectorSearch, VectorSearchError


class VectorSearchContractTest(unittest.TestCase):
    def test_query_and_hit_are_immutable_and_storage_neutral(self):
        query = SearchQuery(text="判别式", dense=(0.1, 0.2))
        hit = SearchHit(
            id="chunk-1", score=0.9, content="答案", item_name="代数",
            file_title="课程", parent_title="根", source="knowledge_base",
        )
        self.assertEqual(query.dense, (0.1, 0.2))
        self.assertEqual(hit.content, "答案")
        with self.assertRaises(FrozenInstanceError):
            hit.score = 0.1

    def test_vector_search_protocol_exposes_item_and_chunk_search(self):
        self.assertIn("search_items", VectorSearch.__dict__)
        self.assertIn("search_chunks", VectorSearch.__dict__)
        self.assertTrue(issubclass(VectorSearchError, RuntimeError))
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_vector_search_contract -v`

Expected: `ModuleNotFoundError: No module named 'app.retrieval.models'`.

- [ ] **Step 3: Implement the immutable interface**

```python
@dataclass(frozen=True)
class SearchQuery:
    text: str
    dense: tuple[float, ...]
    sparse: dict[int, float] | None = None

@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    content: str
    item_name: str
    file_title: str
    parent_title: str
    source: str = "knowledge_base"

class VectorSearch(Protocol):
    def search_items(self, query: SearchQuery, *, top_k: int = 5) -> list[SearchHit]: ...
    def search_chunks(self, query: SearchQuery, item_names: list[str], *, top_k: int = 5) -> list[SearchHit]: ...
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_vector_search_contract -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the seam**

```powershell
git add app/retrieval/models.py app/retrieval/vector_search.py app/retrieval/__init__.py app/test/test_vector_search_contract.py
git commit -m "重构：定义用户级向量查询接口"
```

### Task 2: Implement Qdrant dense + BM25 search

**Files:**
- Create: `app/retrieval/qdrant_adapter.py`
- Create: `app/test/test_qdrant_retrieval.py`

- [ ] **Step 1: Write failing adapter contract tests**

Use a recording Qdrant client and real `qdrant_client.models`. Assert both `search_items` and `search_chunks` call `query_points` with two prefetches (`using="dense"` and `using="bm25"`), `Fusion.RRF`, `with_payload=True`, and a mandatory `user_id` condition. Assert chunk search adds an `item_name` match-any condition and maps points to `SearchHit`.

```python
adapter = QdrantVectorSearch(config, user_id="user-a", client=client, models_module=models)
hits = adapter.search_chunks(SearchQuery("判别式", (0.1, 0.2)), ["代数"], top_k=3)
self.assertEqual(client.calls[0]["query"], models.FusionQuery(fusion=models.Fusion.RRF))
self.assertEqual(hits[0].id, "chunk-1")
self.assertEqual(hits[0].content, "一元二次方程")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_qdrant_retrieval -v`

Expected: import failure for `QdrantVectorSearch`.

- [ ] **Step 3: Implement the adapter**

Create `QdrantVectorSearch(config, user_id, client=None, models_module=None)`. Build:

```python
prefetch = [
    models.Prefetch(query=list(query.dense), using="dense", limit=max(top_k, 10)),
    models.Prefetch(
        query=models.Document(text=query.text, model=config.bm25_model),
        using="bm25",
        limit=max(top_k, 10),
    ),
]
response = client.query_points(
    collection_name=collection,
    prefetch=prefetch,
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=filter_value,
    limit=top_k,
    with_payload=True,
)
```

Convert every point into the storage-neutral hit type. Convert client exceptions to `VectorSearchError("Qdrant 知识库检索失败")` without provider details.

- [ ] **Step 4: Run focused and import regression tests**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_qdrant_retrieval app.test.test_qdrant_vector_store -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Qdrant retrieval**

```powershell
git add app/retrieval/qdrant_adapter.py app/test/test_qdrant_retrieval.py
git commit -m "功能：实现Qdrant混合查询适配器"
```

### Task 3: Implement user-bound Milvus search

**Files:**
- Create: `app/retrieval/milvus_adapter.py`
- Create: `app/test/test_milvus_retrieval.py`

- [ ] **Step 1: Write failing Milvus contract tests**

Use a recording client and injected request/ranker factories. Assert every item and chunk expression contains `user_id == "user-a"`, chunk expressions also contain escaped `item_name in [...]`, and returned rows map to the same `SearchHit` shape as Qdrant.

- [ ] **Step 2: Run the test and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_milvus_retrieval -v`

Expected: import failure for `MilvusVectorSearch`.

- [ ] **Step 3: Implement the adapter**

Create a client from `MilvusVectorStoreConfig` only when no client is injected. Require dense and sparse query values. Build two `AnnSearchRequest` values and execute `client.hybrid_search` with `WeightedRanker(0.9, 0.1, norm_score=True)`. Escape user-controlled strings with the existing Milvus escaping helper. Convert all failures to `VectorSearchError("Milvus 知识库检索失败")`.

- [ ] **Step 4: Run focused and import regression tests**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_milvus_retrieval app.test.test_milvus_vector_store -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Milvus retrieval**

```powershell
git add app/retrieval/milvus_adapter.py app/test/test_milvus_retrieval.py
git commit -m "功能：实现用户隔离的Milvus查询适配器"
```

### Task 4: Deepen Retrieval and create QueryRuntime

**Files:**
- Create: `app/query_process/runtime.py`
- Replace: `app/retrieval/local_adapter.py`
- Replace: `app/retrieval/factory.py`
- Modify: `app/application_services.py`
- Create: `app/test/test_query_runtime.py`
- Modify: `app/test/test_retrieval_seam.py`

- [ ] **Step 1: Write failing runtime factory tests**

Inject recording embedding, vector-search and reranker factories. Verify a `siliconflow + qdrant` snapshot creates a runtime bound to snapshot user/version and Qdrant config; verify `local_bge_m3 + milvus` selects Milvus; unsupported incomplete snapshots fail before graph construction.

```python
runtime = create_query_runtime(snapshot, embedding_factory=embedding_factory,
                               qdrant_factory=qdrant_factory,
                               milvus_factory=milvus_factory)
self.assertEqual(runtime.user_id, "user-a")
self.assertEqual(runtime.settings_version, 7)
self.assertIs(runtime.retrieval.vector_search, qdrant_search)
```

- [ ] **Step 2: Run and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_runtime -v`

Expected: missing `app.query_process.runtime`.

- [ ] **Step 3: Implement RetrievalModule and runtime factory**

`RetrievalModule` receives `embedding`, `vector_search` and `reranker` dependencies. It creates `SearchQuery` from embedding output, uses the optional HyDE document in the embedded text, maps item hits into current item-match dictionaries, and delegates reranking without global configuration. `create_query_runtime(snapshot)` selects Qdrant or Milvus from the frozen snapshot.

Rename `ApplicationServices.runtime_factory` to `import_runtime_factory`, add `query_runtime_factory`, and update import callers/tests accordingly.

- [ ] **Step 4: Run runtime, retrieval and application tests**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_runtime app.test.test_retrieval_seam app.test.test_application_services app.test.test_import_api_auth -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the deep runtime module**

```powershell
git add app/query_process/runtime.py app/retrieval/local_adapter.py app/retrieval/factory.py app/application_services.py app/test/test_query_runtime.py app/test/test_retrieval_seam.py app/test/test_application_services.py app/test/test_import_api_auth.py app/import_process/api/file_import_service.py
git commit -m "重构：建立用户级问答运行时"
```

### Task 5: Inject runtime into LangGraph and fix answer quality

**Files:**
- Modify: `app/query_process/agent/main_graph.py`
- Modify: query retrieval nodes under `app/query_process/agent/nodes/`
- Modify: `app/query_process/agent/nodes/node_rrf.py`
- Modify: `app/query_process/agent/nodes/node_rerank.py`
- Modify: `app/query_process/agent/nodes/node_answer_output.py`
- Create: `app/test/test_query_graph_runtime.py`
- Modify: `app/test/test_reranker_degradation.py`

- [ ] **Step 1: Write failing graph-interface tests**

Build the graph with a fake `QueryRuntime`; assert nodes call the injected retrieval without patching globals. Add a direct RRF test where a hit present in two routes ranks above a single-route hit. Add an empty-evidence test asserting the answer LLM is not called and the stable no-evidence response is returned.

- [ ] **Step 2: Run and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_graph_runtime -v`

Expected failures: `build_query_graph` does not accept runtime, RRF returns `['B', 'A']`, and answer LLM is invoked for no evidence.

- [ ] **Step 3: Implement runtime node factories and fixes**

Replace module-global `query_app` construction with `build_query_graph(runtime, checkpointer)`. Create node factories only for nodes that need retrieval. Change RRF assignment to accumulation:

```python
score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + weight / (60 + index)
```

Before answer generation, if `reranked_docs` is empty, set a stable no-evidence answer, emit no fabricated sources, and skip `get_llm_client()`.

- [ ] **Step 4: Run graph and warning regressions**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_graph_runtime app.test.test_reranker_degradation app.test.test_query_warning_flow -v`

Expected: all tests pass.

- [ ] **Step 5: Commit graph injection**

```powershell
git add app/query_process/agent app/test/test_query_graph_runtime.py app/test/test_reranker_degradation.py app/test/test_query_warning_flow.py
git commit -m "重构：向问答图注入用户运行时"
```

### Task 6: Add QueryEngine and authenticated router

**Files:**
- Create: `app/query_process/engine.py`
- Create: `app/query_process/api/router.py`
- Modify: `app/api/server.py`
- Modify: `app/query_process/api/query_server.py`
- Create: `app/test/test_query_engine.py`
- Create: `app/test/test_query_http.py`

- [ ] **Step 1: Write failing lifecycle and HTTP tests**

Verify unauthenticated query/confirm/stream return `401`; snapshot configuration errors return `409`; non-stream answers are read by `request_id`; cross-user stream/confirm return `404`; invalid candidates return `409`. Verify failures produce an error terminal result instead of `200 + empty answer`.

- [ ] **Step 2: Run and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_engine app.test.test_query_http -v`

Expected: missing QueryEngine/router and current unauthenticated routes return non-401 responses.

- [ ] **Step 3: Implement request ownership and router**

`QueryEngine` keeps a process-local request registry keyed by `request_id`, with `user_id`, `session_id`, status and terminal outcome. It builds the runtime from `services.settings.get_snapshot(user.id)`, constructs an owner-qualified thread key, and owns a shared `InMemorySaver`. `create_query_router(services)` injects the same current-user dependency used by import routes and maps stable engine exceptions to `401/404/409/503`.

- [ ] **Step 4: Run lifecycle and auth regressions**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_engine app.test.test_query_http app.test.test_auth_http app.test.test_import_api_auth -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the engine**

```powershell
git add app/query_process/engine.py app/query_process/api/router.py app/query_process/api/query_server.py app/api/server.py app/test/test_query_engine.py app/test/test_query_http.py
git commit -m "功能：统一认证问答生命周期"
```

### Task 7: Isolate history by user

**Files:**
- Modify: `app/clients/mongo_history_utils.py`
- Modify: `app/clients/mongo_history_utils_new.py`
- Create: `app/tools/assign_legacy_history.py`
- Create: `app/test/test_history_user_isolation.py`
- Modify: `app/test/test_query_warning_flow.py`

- [ ] **Step 1: Write failing history adapter tests**

Assert saved documents contain `user_id`; recent messages, session summaries and deletes filter by both user and session; user B sees no rows from user A. Assert legacy rows without `user_id` remain invisible. Test the explicit migration function assigns only rows lacking ownership to the requested user.

- [ ] **Step 2: Run and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_history_user_isolation -v`

Expected: current history functions do not accept mandatory `user_id` and stored documents omit it.

- [ ] **Step 3: Implement mandatory ownership and migration**

Add `user_id` to history interface calls and Mongo documents. Create `(user_id, session_id, ts)` index. Update every query/delete/aggregation match. Add a command requiring both `--user-id` and `--confirm ASSIGN_LEGACY_HISTORY`; it updates only `{"user_id": {"$exists": False}}`.

- [ ] **Step 4: Run history and query regressions**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_history_user_isolation app.test.test_query_warning_flow app.test.test_query_engine app.test.test_query_http -v`

Expected: all tests pass.

- [ ] **Step 5: Commit user-scoped history**

```powershell
git add app/clients/mongo_history_utils.py app/clients/mongo_history_utils_new.py app/tools/assign_legacy_history.py app/test/test_history_user_isolation.py app/test/test_query_warning_flow.py
git commit -m "安全：按用户隔离问答历史"
```

### Task 8: Use authenticated fetch-stream and remove mock answers

**Files:**
- Create: `frontend/src/services/sse.ts`
- Create: `frontend/src/services/sse.test.ts`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/views/ChatView.vue`
- Create: `frontend/src/views/ChatView.test.ts`

- [ ] **Step 1: Write failing frontend tests**

Assert query, confirm and history call `authFetch`; non-2xx responses reject with the public error; no mock answer is returned. Feed split UTF-8 SSE chunks to the parser and assert `delta`, `warning`, `final`, `confirmation_required`, and `error` dispatch correctly. Assert an error terminal event stops loading and displays its message.

- [ ] **Step 2: Run and verify RED**

Run: `npm test -- --run src/services/sse.test.ts src/views/ChatView.test.ts`

Working directory: `frontend`

Expected: missing SSE parser, direct `fetch` calls, and mock fallback cause failures.

- [ ] **Step 3: Implement authenticated streaming**

Create `streamSse(input, init, onEvent)` using `authFetch`, `Response.body.getReader()`, `TextDecoder`, and buffering until `\n\n`. Parse `event:` and `data:` fields and stop on `final`, `confirmation_required`, or `error`. Replace EventSource usage, route all protected calls through `authFetch`, and delete the entire local mock answer branch.

- [ ] **Step 4: Run frontend tests and build**

Run: `npm test -- --run`

Run: `npm run build`

Working directory: `frontend`

Expected: all tests and production build pass.

- [ ] **Step 5: Commit frontend lifecycle**

```powershell
git add frontend/src/services/sse.ts frontend/src/services/sse.test.ts frontend/src/services/api.ts frontend/src/views/ChatView.vue frontend/src/views/ChatView.test.ts
git commit -m "修复：使用认证流式问答并移除仿真答案"
```

### Task 9: Make evaluation and verification truthful

**Files:**
- Modify: `app/eval/run_pipeline.py`
- Modify: `app/eval/run_ragas.py`
- Expand: `app/eval/golden_set.jsonl`
- Create: `app/test/test_query_evaluation.py`
- Modify: `README_DEPLOYMENT.md`

- [ ] **Step 1: Write failing evaluation tests**

Assert confirmation resumes with the candidate ID string expected by the graph; error and empty-evidence samples remain in evaluation rows; failure rate is included in summary data; no default path silently skips failed samples.

- [ ] **Step 2: Run and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest app.test.test_query_evaluation -v`

Expected: current resume payload shape and skip-bad defaults fail assertions.

- [ ] **Step 3: Implement evaluation fixes and deployment documentation**

Use QueryEngine/QueryRuntime for pipeline samples, include every sample with status metadata, compute success/failure/empty-evidence counts before RAGAS metrics, and expand the golden set across multiple imported topics and multi-turn questions. Document authenticated query settings, SSE failures, and the explicit legacy-history migration command.

- [ ] **Step 4: Run complete verification**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s app/test -p 'test_*.py'`

Run: `npm test -- --run`

Run: `npm run build`

Expected: zero backend failures/errors, zero frontend test failures, and a successful production build.

- [ ] **Step 5: Run an opt-in real Qdrant smoke test**

Run only when explicit external credentials are available: import one uniquely worded document, query it as the same user, assert at least one source contains its `document_id`, then query as a second user and assert zero matching sources. Do not place credentials or full vectors in logs.

- [ ] **Step 6: Commit evaluation and operations documentation**

```powershell
git add app/eval/run_pipeline.py app/eval/run_ragas.py app/eval/golden_set.jsonl app/test/test_query_evaluation.py README_DEPLOYMENT.md
git commit -m "测试：增加用户级问答质量门禁"
```
