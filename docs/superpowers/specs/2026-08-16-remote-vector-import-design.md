# 远程向量导入最小链路设计

## 目标

在 2 核 2G 部署环境中跑通以下最小纵向链路：

```text
管理员登录
  -> 上传文档
  -> SiliconFlow BAAI/bge-m3 生成 dense 向量
  -> Qdrant Cloud 写入 dense 与服务端 BM25 sparse 数据
```

同时保留 Milvus 作为可选存储 adapter。单次导入只写用户配置选中的一个向量库，目标库失败时任务失败，不双写，也不静默切换存储。

## 本轮范围

### 包含

- 单管理员账号登录，使用 Bearer JWT 识别稳定 `user_id`。
- SQLite 持久化用户和模型、向量库配置。
- API Key 加密落库，读取配置时不向前端回传明文。
- SiliconFlow OpenAI-compatible Embedding adapter，默认模型为 `BAAI/bge-m3`。
- Vector Store module seam，以及 Qdrant、Milvus 两个 adapter。
- 上传任务创建时冻结用户配置，并将可信 `user_id` 传入导入图。
- Qdrant 中写入主题和 chunks，payload 包含 `user_id`。
- Milvus schema 增加 `user_id`，提供显式的一次性清库重建命令。
- 最小前端登录和设置入口，使管理员可以完成配置并上传文件。

### 不包含

- 开放注册、找回密码、刷新令牌和多角色权限。
- 将查询、知识库管理、编辑或删除链路迁移到 Qdrant。
- 迁移现有 Milvus 向量数据。
- Qdrant 与 Milvus 双写或故障自动切换。
- 将现有 MongoDB、MinIO、MinerU 等依赖全部移除。
- 使用 Qdrant 查询时的 dense、BM25 和 RRF 融合；本轮只准备兼容该查询方式的数据结构。

## 方案选择

采用深层 Vector Import module。上传图只处理文档业务数据，不再直接创建 collection、拼接数据库过滤表达式或调用具体数据库客户端。

不采用以下方案：

- 在现有三个节点中增加 Qdrant/Milvus 条件分支。该方案会让存储知识继续泄漏到调用方，locality 较差。
- 复制一套 Qdrant 导入图。该方案会重复主题识别、向量生成、异常传播和任务状态逻辑。

## Module 与 seam

### Current User module

`CurrentUser` seam 从 Bearer JWT 中得到可信的 `user_id`。`/upload`、`/status/{task_id}` 和用户设置接口通过依赖注入使用该 interface，不接受前端传入的 `user_id`。

最小认证流程：

1. 启动时从环境变量读取管理员用户名和初始密码。
2. SQLite 中不存在管理员时创建账号，已存在时不覆盖密码。
3. `POST /api/auth/login` 校验密码并签发有过期时间的 JWT。
4. 前端将 JWT 放在 `Authorization: Bearer <token>` 请求头中。
5. 不提供公开注册接口。

用户表保留稳定 UUID 主键和角色字段，后续增加多用户时不需要修改上传 interface。

### User Runtime Settings module

该 module 通过 `user_id` 读写用户运行配置，并返回不可变的配置快照。SQLite 至少保存：

- Embedding provider、base URL、model、dimension、batch size 和 timeout。
- Vector store 类型：`qdrant` 或 `milvus`。
- Qdrant URL、API Key、collection 名和是否启用 Cloud Inference。
- Milvus URL、token 和 collection 名。
- 配置版本和更新时间。

Embedding API Key、Qdrant API Key 和 Milvus token 使用服务端主密钥加密。设置查询只返回 `configured: true/false` 和掩码，不返回可还原的密钥。

保存设置时执行组合校验。最小版本明确支持：

- `siliconflow + qdrant`：远程 dense 向量由 SiliconFlow 生成，BM25 sparse 由 Qdrant Cloud Inference 生成。
- `local_bge_m3 + milvus`：保留现有 dense+sparse 行为。

其他组合返回明确的配置错误，不在运行中猜测或降级。

### Embedding module

沿用现有 Embedding seam，新增 `SiliconFlowEmbeddingAdapter`。它调用 OpenAI-compatible `/v1/embeddings` interface，只返回 dense 向量，并校验：

- 返回数量必须与输入文本数量一致。
- 每条向量维度必须与用户配置一致。
- 超时、认证失败、限流和响应格式错误转换为稳定的 embedding 异常。
- 日志不包含 API Key、请求头或完整向量。

现有本地 BGE-M3 adapter 保留，用于 Milvus 旧链路。远程 adapter 不伪造 sparse 向量。

### Vector Import module

外部 interface 只暴露一次文档导入操作：

```python
result = vector_store.import_document(document)
```

`document` 包含：

- `user_id`
- `document_id`
- `file_title`
- `item_name`
- 主题 dense 向量
- chunks 原始文本、元数据和 dense 向量
- 可选 sparse 向量

module 内部隐藏 schema、collection 初始化、幂等删除、point ID、批量写入和数据库错误转换。Qdrant 与 Milvus 是该 seam 上的两个真实 adapter。

## 数据流

```text
POST /upload + Bearer JWT
  -> CurrentUser 得到 user_id
  -> User Runtime Settings 读取不可变配置快照
  -> factory 创建本次任务专用的 embedding/vector-store adapters
  -> 后台任务持有 adapters，state 只保存非敏感标识
  -> 文档解析与切片
  -> 主题识别，仅更新 item_name，不再写 Milvus
  -> 向量节点一次性向量化 item_name 与 chunks
  -> Vector Import module 写入用户选中的唯一向量库
  -> 成功标记 completed，任一步异常标记 failed
```

adapter 在任务创建时完成，不在每个节点重新读取 SQLite。这样用户在导入过程中修改设置不会改变当前任务，也不会把密钥放进会被日志打印的 LangGraph state。

## Qdrant 数据结构

最小版本使用两个共享 collection：

- `rag_item_names_v1`
- `rag_chunks_v1`

两个 collection 都配置：

- named dense vector：`dense`，Cosine 距离，默认 1024 维。
- named sparse vector：`bm25`，模型为 `Qdrant/bm25`，启用 IDF modifier。

所有 point payload 都包含：

- `user_id`
- `document_id`
- `file_title`
- `item_name`
- `content` 或主题文本

chunk payload 额外保留 `title`、`parent_title`、`part` 和 chunk 顺序。

`document_id` 由 `user_id + 规范化文件名` 生成稳定哈希。重复上传同一文件时，Qdrant adapter 先按 `user_id + document_id` 删除旧 chunks，再批量 upsert 新数据。point ID 使用稳定 UUID，不依赖数据库自增 ID。

Qdrant Cloud 通过 `Document(text=..., model="Qdrant/bm25")` 接收主题和 chunk 原文并在服务端生成 sparse 向量。若当前免费集群不支持该模型或 Cloud Inference，导入任务直接失败，并在任务错误中说明失败阶段。

## Milvus 数据结构与重建

Milvus adapter 保留两个 collection，并在 schema 中增加：

- `user_id: VARCHAR`
- `document_id: VARCHAR`

幂等删除条件从仅匹配 `item_name` 改为同时匹配 `user_id` 和 `document_id`。主题记录按 `user_id + item_name` 隔离。

不在应用启动或首次导入时自动删除旧 collection。提供显式命令，要求传入确认参数后才删除并重建两个 collection。用户已确认不保留旧向量数据；该命令只负责 schema 重建，不执行数据迁移。

## 最小前端

前端只增加完成纵向链路所需的界面：

- 登录页：用户名、密码和登录操作。
- 设置页：Embedding provider、base URL、model、dimension、API Key、vector store，以及所选存储的 URL、密钥和 collection 配置。
- 路由守卫：未登录时进入登录页。
- 请求封装：上传、状态轮询和设置请求统一附带 Bearer JWT。
- 密钥输入为空表示保留旧密钥；显式清除使用单独操作，避免编辑其他字段时误删密钥。

查询和知识库管理页面暂不迁移到 Qdrant。当用户选择 Qdrant 后，设置页显示“当前仅支持导入，查询迁移将在下一阶段完成”的状态提示，避免形成已可完整使用的误解。

## 错误语义

- JWT 缺失、过期或无效：返回 `401`。
- 用户访问其他用户的任务：返回 `404`，不泄漏任务是否存在。
- 配置缺失或组合不受支持：上传请求返回 `409`，不创建后台任务。
- Embedding 或 Qdrant/Milvus 写入失败：后台任务标记 `failed`，保存稳定错误摘要和失败阶段。
- 不记录密码、JWT、API Key、数据库 token、完整向量或包含密钥的配置对象。
- adapter factory 创建失败时不回退到另一 adapter。

## 测试

测试通过 module interface 和 HTTP interface 验证：

- 首次启动创建管理员，重复启动不覆盖账号。
- 登录成功、密码错误、JWT 过期和 JWT 篡改。
- SQLite 设置按用户隔离，密钥以密文保存，读取接口不返回明文。
- 上传使用 JWT 中的 `user_id`，忽略或拒绝伪造身份字段。
- 配置快照在任务开始后不随用户设置变化。
- SiliconFlow adapter 的批次、数量、维度、超时和错误转换。
- Qdrant adapter 创建 dense+sparse schema，写入 BM25 Document，并按 `user_id + document_id` 幂等替换。
- Milvus adapter schema 和过滤表达式都包含 `user_id`。
- 单库选择严格生效，目标 adapter 失败后另一 adapter 未被调用。
- 导入异常最终把任务标记为 `failed`。
- 前端构建通过，登录、设置和上传请求携带 JWT。

真实远程验证使用独立的手工探针或受控测试账号，日志只输出请求阶段、耗时、向量数量、维度和写入数量。没有真实 SiliconFlow、Qdrant 凭据时，自动测试不得发起外部网络调用。

## 完成标准

1. 管理员可登录并保存 SiliconFlow 与 Qdrant 配置。
2. 上传一个小型文档后，任务状态为 `completed`。
3. SiliconFlow 返回的 1024 维 dense 向量写入 Qdrant。
4. Qdrant 对主题和 chunks 接收 `Qdrant/bm25` 文本推理数据。
5. Qdrant payload 包含稳定 `user_id` 和 `document_id`。
6. 重复上传同一文件不会保留旧 chunks。
7. Qdrant 写入失败时任务为 `failed`，Milvus 没有收到写入。
8. 将用户设置切换为 Milvus 后，factory 选择现有 Milvus adapter。
