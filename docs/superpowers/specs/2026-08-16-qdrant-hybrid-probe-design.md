# Qdrant 混合检索探针设计

## 目标

新增一个仅供手工运行的端到端探针，验证以下真实云链路是否可用：

1. 硅基流动 `BAAI/bge-m3` 生成 1024 维 dense 向量。
2. Qdrant Cloud 使用服务端 `Qdrant/bm25` 生成 sparse 向量。
3. Qdrant 使用 RRF 融合 dense 与 sparse 检索结果。

探针不接入正式导入或查询流程，不改变现有 Milvus collection，也不进入自动单元测试发现范围。

## 运行接口

探针类命名为 `QdrantHybridProbe`，模块路径为：

```text
app/eval/qdrant_hybrid_probe.py
```

手工运行命令：

```powershell
.\.venv\Scripts\python.exe -m app.eval.qdrant_hybrid_probe
```

`run()` 是唯一对外入口，负责完整执行和资源清理。内部步骤保持私有，避免测试脚本形成新的业务 interface。

## 配置

从环境变量读取：

```env
QDRANT_URL=
QDRANT_API_KEY=
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_EMBEDDING_DIMENSION=1024
QDRANT_BM25_MODEL=Qdrant/bm25
QDRANT_CLOUD_INFERENCE=true
```

`SILICONFLOW_API_KEY` 未设置时，兼容读取 `RERANKER_API_KEY`。探针启动时一次性校验配置，缺失字段只输出变量名，不输出其他配置值。

## 数据流

探针使用三个固定的中文测试文档和一个查询问题。每次运行生成随机 collection 名，避免并发运行互相覆盖。

1. 调用硅基流动 `/embeddings`，分别为三个文档生成 dense 向量。
2. 创建包含 `dense_vector` 和 `bm25_sparse_vector` 的 Qdrant collection。
3. dense 向量直接写入；BM25 以 `Document(text=..., model="Qdrant/bm25")` 交给 Qdrant 服务端生成。
4. 查询时调用硅基流动生成 query dense 向量，同时向 Qdrant 提交 BM25 query document。
5. 通过两个 prefetch 和顶层 RRF fusion 返回最终结果。
6. 日志输出结果顺序、融合分数、文档标题和内容摘要。

## Collection 配置

临时 collection 使用两个 named vectors：

- `dense_vector`：1024 维，Cosine 距离。
- `bm25_sparse_vector`：sparse vector，启用 Qdrant `Modifier.IDF`。

测试 point 的 payload 包含 `title`、`content` 和 `probe_run_id`，不包含任何密钥。

## 日志与安全

每一步记录开始、结束和耗时：配置校验、dense 请求、collection 创建、数据写入、RRF 查询和资源清理。

- API Key 仅记录“已设置”以及末尾四位掩码，不记录明文。
- HTTP 错误记录状态码和经过长度限制的响应正文。
- 不记录 Authorization header 或完整请求对象。
- 结果日志不输出向量内容，只输出向量数量与维度。

## 失败与清理

任何步骤失败都抛出稳定的探针异常，并在日志中标明失败阶段。常见错误需要区分：

- 硅基流动认证、模型不可用、限流或向量维度不符。
- Qdrant 认证、collection 创建失败或免费集群不支持 Cloud Inference。
- `Qdrant/bm25` 不在当前集群可用模型列表。
- RRF 查询响应为空或 payload 缺失。

collection 名一旦生成，就在 `finally` 中尝试删除。清理失败只追加错误日志，不覆盖原始失败原因。

## 验证

自动测试使用假的 HTTP client 和假的 Qdrant client，通过探针的 `run()` interface 验证：

- 配置缺失时不会发起网络调用。
- dense 响应维度和数量会被校验。
- collection 同时配置 dense、sparse 和 IDF。
- 写入与查询使用正确的 named vectors 和 RRF。
- 成功和异常路径都会删除临时 collection。
- 日志和异常文本不包含 API Key 明文。

真实云验证由用户手工运行探针完成，不纳入常规单元测试。

## 非目标

- 不迁移现有 Milvus 数据。
- 不接入正式 Retrieval module。
- 不实现前端模型设置或按用户保存。
- 不在本次探针中调用远程 Reranker。
