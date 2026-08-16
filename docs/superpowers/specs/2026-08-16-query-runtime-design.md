# 用户级问答运行时设计

## 目标

将已经完成的用户认证、SiliconFlow 远程 Embedding 和 Qdrant Cloud 文档导入能力接入问答核心，使同一用户的知识文档能够被正确、隔离地检索，并生成带回答依据的可验证回答。

首要纵向链路为：

```text
Bearer JWT
  -> 用户运行配置快照
  -> SiliconFlow BAAI/bge-m3 查询向量
  -> Qdrant dense + BM25 混合检索
  -> 重排与有依据回答
  -> 结构化终态事件
```

Milvus 查询 adapter 继续兼容现有 `local_bge_m3 + milvus` 配置，但不作为首个远程验收路径。

## 范围

### 包含

- 用户级 Query Runtime module，以及由用户运行配置快照创建运行时的 factory。
- Retrieval module 与内部 Vector Search seam。
- Qdrant、Milvus 两个查询 adapter，强制执行用户隔离。
- 通过闭包向 LangGraph 节点注入运行时，不在图状态中保存密钥或客户端。
- Query Engine module，统一查询请求、候选主题确认、任务结果和 SSE 终态。
- 问答、确认、流式事件和历史接口认证。
- MongoDB 历史记录增加用户归属。
- 修复 request/session 标识混用、RRF 分数覆盖、空回答依据继续生成等问题。
- 删除生产环境仿真答案，网络搜索改为显式启用。
- 后端、前端和受控端到端测试。

### 不包含

- 开放注册、多角色授权和刷新令牌。
- 跨向量库故障自动切换或双读。
- 自动迁移 Milvus 数据到 Qdrant。
- 把 LLM 主模型配置迁入用户运行设置。
- 分布式 LangGraph checkpoint 和跨进程 SSE。
- 自动认领没有用户归属的旧 MongoDB 历史记录。

## 方案选择

采用分阶段深化方案：先建立 Query Runtime，再建立 Query Engine，最后让评测直接穿过相同 interface。

不采用最小补丁方案。仅向全局 `get_retrieval()` 增加 Qdrant 分支仍会让用户配置、Embedding、向量库和身份过滤泄漏到节点，删除 factory 后复杂度不会集中，module 仍然 shallow。

不采用一次性重写全部问答代码。Query Runtime 与 Query Engine 可以分别形成可运行、可测试的提交，降低 LangGraph、SSE、历史和前端同时变化的回归范围。

## Module 与 seam

### Query Runtime module

Query Runtime 是每个用户查询请求使用的不可变运行时。外部 interface 只要求 factory 从用户运行配置快照创建运行时：

```python
@dataclass(frozen=True)
class QueryRuntime:
    user_id: str
    settings_version: int
    retrieval: Retrieval


def create_query_runtime(snapshot: UserRuntimeSnapshot) -> QueryRuntime:
    ...
```

factory 创建并组合：

- 与导入阶段相同配置的 Embedding adapter。
- 与 `vector_store_type` 对应的 Vector Search adapter。
- 当前进程配置的 Reranker provider。
- 对外暴露稳定 interface 的 Retrieval module。

运行时在查询请求开始时冻结。用户修改设置后只影响后续查询，不改变已经运行或等待候选主题确认的请求。

### Vector Search seam

Vector Search seam 隐藏 Qdrant 与 Milvus 查询语法、过滤表达式、混合检索和命中格式。两个生产 adapter 证明该 seam 真实存在。

interface 只接受已经生成的查询表示和检索意图：

```python
class VectorSearch(Protocol):
    def search_items(self, query: SearchQuery, *, top_k: int) -> list[SearchHit]: ...
    def search_chunks(
        self,
        query: SearchQuery,
        item_names: list[str],
        *,
        top_k: int,
    ) -> list[SearchHit]: ...
```

adapter 构造时绑定 `user_id`、集合配置和客户端。`user_id` 不是可选调用参数，因此任何调用者都无法漏传身份过滤。

统一 `SearchHit` 至少包含：

- `id`
- `score`
- `content`
- `item_name`
- `file_title`
- `parent_title`
- `source`

### Retrieval module

Retrieval module 在 Vector Search seam 外提供问答图需要的高层行为：

```python
class Retrieval(Protocol):
    def match_item_names(self, item_names: list[str]) -> list[ItemMatch]: ...
    def search_chunks(
        self,
        query: str,
        item_names: list[str],
        *,
        hyde_document: str | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]: ...
    def rerank(self, query: str, hits: list[SearchHit]) -> RerankedDocuments: ...
```

它隐藏以下实现：

- 调用 Embedding adapter。
- 为普通问题和 HyDE 文本生成查询表示。
- 把 Vector Search 命中转换为稳定结果。
- 调用 Reranker 并保留降级告警。
- 校验空问题、向量维度和结果索引。

LangGraph 节点只跨过 Retrieval interface，不接触具体向量库或 Embedding 配置。

### Query Engine module

Query Engine 统一一次查询请求的完整生命周期。外部 interface 为：

```python
class QueryEngine:
    async def ask(self, user: User, request: QueryRequest) -> QueryResponse: ...
    async def confirm(self, user: User, request: ConfirmRequest) -> QueryResponse: ...
    async def events(self, user: User, request_id: str) -> AsyncIterator[QueryEvent]: ...
```

Query Engine 负责：

- 获取并冻结用户运行配置快照。
- 创建 Query Runtime 和查询图。
- 创建并绑定查询请求所有权。
- 管理任务状态、答案和公开错误。
- 校验候选主题确认请求。
- 保证每个请求只有一个终态事件。
- 保存带用户归属的历史消息。

FastAPI router 是 transport adapter，只解析 HTTP 数据、注入当前用户并转换公开异常。复杂度集中在 Query Engine，HTTP 与 interface 测试复用相同行为。

## LangGraph 注入与 checkpoint

查询图改为：

```python
build_query_graph(runtime: QueryRuntime, checkpointer)
```

需要 Retrieval 的节点由 node factory 创建，通过闭包持有 `runtime.retrieval`。运行时、API Key、数据库客户端不进入 `QueryGraphState`，也不进入 checkpoint、日志或历史记录。

checkpoint 继续使用当前进程内存实现。本轮不解决服务重启后恢复候选主题确认。thread key 使用：

```text
user_id:session_id
```

候选主题确认必须同时满足：

- 查询会话属于当前用户。
- `pending_request_id` 属于当前用户和当前查询会话。
- checkpoint 正在等待确认。
- `candidate_id` 存在于 checkpoint 候选集合。

任何条件不满足均返回 `404` 或 `409`，不得宽松放行。

## Qdrant 混合检索

Qdrant adapter 使用导入阶段已经建立的 named vectors：

- `dense`：SiliconFlow `BAAI/bge-m3` 生成的 1024 维向量。
- `bm25`：Qdrant Cloud Inference 根据查询文本生成的 sparse 表示。

主题与知识切片查询都执行 dense 与 BM25 prefetch，再由 Qdrant RRF fusion 生成候选。每次查询的 filter 必须包含：

```text
user_id == current_user.id
```

知识切片存在已确认学习主题时，额外包含 `item_name in [...]`。无学习主题时允许在当前用户范围内全局检索，不允许跨用户全局检索。

Qdrant 响应统一转换为 `SearchHit`，后续 RRF、Reranker、回答依据和引用提取不再处理 Qdrant 原生对象。

## Milvus 兼容

Milvus adapter 使用用户快照中的 URL、token、collection 和 dimension 创建客户端，并使用同一 Embedding adapter 生成 dense+sparse 查询表示。

过滤表达式始终包含 `user_id`；存在学习主题时再附加 `item_name` 条件。旧的全局 `milvus_config` 和全局客户端不参与 Query Runtime。

Milvus adapter 与 Qdrant adapter 必须通过相同 contract tests，返回相同 `SearchHit` 结构。

## 请求标识与状态

- `request_id` 唯一标识一次查询请求，用于任务状态、结果、SSE 队列和所有权校验。
- `session_id` 唯一标识当前用户的一段查询会话，用于历史记录和 LangGraph thread。
- 所有任务结果只能按 `request_id` 读写。
- 所有历史记录只能按 `user_id + session_id` 读写。
- 前端可以提供 `session_id`，服务端必须将它放入当前用户命名空间，不能把它当作全局标识。

## 认证与历史隔离

新增 `create_query_router(services)`，与导入 router 使用相同的 Current User dependency。以下 interface 都要求 Bearer JWT：

- 提问
- 候选主题确认
- 流式事件
- 会话历史读取和删除
- 历史会话列表

MongoDB 新消息增加 `user_id`。索引采用：

```text
(user_id, session_id, ts)
```

没有 `user_id` 的旧历史记录默认不可见，避免错误归属。提供显式运维迁移命令，将旧记录分配给指定用户；迁移不在应用启动时自动执行。

## SSE 与终态契约

每个查询请求必须产生且只产生一个终态：

- `final`：包含答案、回答依据、图片、节点耗时和告警。
- `confirmation_required`：包含候选学习主题和待确认请求标识。
- `error`：包含稳定错误代码、公开消息和是否可重试。

终态发送后立即发送内部 close 信号并释放队列。未知、过期或属于其他用户的 `request_id` 在建立流之前返回 `404`。

错误 payload：

```json
{
  "code": "vector_search_unavailable",
  "message": "知识库检索暂时不可用",
  "retryable": true
}
```

不得返回 provider 响应正文、URL 中的凭据、密钥或完整配置。

前端不再使用原生 `EventSource`，改为带 Authorization Header 的 `fetch + ReadableStream` SSE 客户端。JWT 不放入查询字符串。

## 回答质量规则

- RRF 对同一知识切片的多路倒数排名分数执行累计，不允许后一路覆盖前一路。
- Reranker 不可用时保留原始检索顺序并返回唯一降级告警。
- 没有回答依据时不调用 LLM 生成事实性回答，返回稳定的“知识库中没有足够依据”结果。
- 回答只引用当前 `reranked_docs`，引用来源与送入 Prompt 的内容来自同一结果集合。
- 网络搜索默认关闭。只有显式启用后才进入查询图，失败时产生告警而不是阻塞私有知识库回答。
- 前端删除自动仿真答案。HTTP 或 SSE 失败必须向用户显示真实错误状态。

## 错误语义

- JWT 缺失、无效或过期：`401`。
- 用户配置不存在或组合不完整：`409`。
- 查询会话、查询请求或历史不属于当前用户：`404`。
- 当前查询不等待候选主题确认：`409`。
- Embedding、向量库、Reranker 或 LLM 外部依赖失败：结构化公开错误；Reranker 允许带告警降级，其他核心依赖失败终止查询。
- 非流式查询失败不得返回 `200 + 空答案`。
- 流式查询失败必须发送 `error` 终态并关闭连接。

## 测试设计

### Vector Search contract

Qdrant、Milvus 和内存测试 adapter 使用同一 fixture 验证：

- 所有主题与知识切片查询都包含用户过滤。
- 学习主题过滤正确组合。
- dense 与 sparse/BM25 查询参数正确。
- 不同数据库命中转换为相同 `SearchHit`。
- 数据库异常转换为稳定查询异常。

### Query Runtime interface

- `siliconflow + qdrant` 快照创建远程 Embedding 与 Qdrant 查询 adapter。
- `local_bge_m3 + milvus` 快照创建本地 Embedding 与 Milvus 查询 adapter。
- 不支持或不完整组合在运行图之前失败。
- 设置版本在单次查询中保持不变。

### Query Graph interface

- Fake Query Runtime 可以驱动完整问答图，不 patch 节点全局 factory。
- 普通召回与 HyDE 命中执行正确的累计 RRF。
- 空回答依据不会调用答案 LLM。
- Reranker 降级告警进入最终结果。
- 候选主题确认后只恢复当前用户命名空间中的 checkpoint。

### Query Engine 与 HTTP

- 未认证提问、确认、流式事件和历史请求返回 `401`。
- 用户 A 不能读取、确认、订阅或删除用户 B 的请求与会话。
- 非流式答案按 `request_id` 返回。
- 每个成功、确认或失败请求只有一个终态。
- 配置错误和依赖错误映射为稳定状态与公开消息。

### 前端

- 提问、确认、历史和流式请求都携带 Bearer JWT。
- fetch-stream 正确解析 delta、progress、warning、final、confirmation_required 和 error。
- error 终态停止加载并展示错误。
- 后端不可用或非 2xx 时不生成仿真答案。

### 受控端到端验证

使用当前用户的真实 SiliconFlow 与 Qdrant 配置：

1. 导入一个包含唯一事实的小型知识文档。
2. 确认 Qdrant 主题和知识切片 payload 包含该用户 ID。
3. 用同一用户提问并召回该事实。
4. 确认最终回答包含回答依据和来源。
5. 使用另一用户查询相同问题，确认无法召回第一用户的数据。

自动测试不使用真实外部凭据。真实验证只在显式运行的探针或受控环境执行。

## 实施顺序

1. 定义统一命中类型与 Vector Search seam。
2. 通过 Qdrant contract tests 实现混合检索。
3. 将现有 Milvus 查询迁入用户绑定 adapter。
4. 建立 Retrieval 和 Query Runtime module。
5. 将查询节点改为 runtime 注入并修复 RRF、空回答依据。
6. 建立 Query Engine 与受认证 router。
7. 增加历史用户归属与显式旧数据迁移命令。
8. 完成 SSE 终态和前端 fetch-stream。
9. 删除前端仿真答案和旧全局查询 factory。
10. 扩充黄金集并让失败样本计入评测结果。

## 完成标准

1. 当前 SQLite 中的 `siliconflow + qdrant` 配置能驱动问答检索，不读取旧 `.env` Embedding/Milvus 查询配置。
2. 同一用户导入的知识文档可被问答召回，并生成带回答依据的答案。
3. Qdrant dense、BM25 和 RRF 混合检索生效。
4. 所有向量查询和历史查询都强制用户隔离。
5. 非流式、流式和候选主题确认使用一致的请求标识和终态语义。
6. 后端失败不会返回空成功结果，前端不会生成仿真答案。
7. 多路 RRF 分数正确累计，空回答依据不会触发事实性生成。
8. Qdrant 与 Milvus adapter contract tests、Query Engine 测试、前端测试和完整回归测试全部通过。
