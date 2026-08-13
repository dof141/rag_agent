# 远程重排序模块设计

## 背景与目标

当前检索 adapter 直接获取 `FlagReranker` 模型对象并调用 `compute_score`，导致检索流程依赖 `FlagEmbedding`、PyTorch 和本地模型文件。该实现不适合 2 核 2G 部署，也会把具体模型接口泄漏给调用方。

本次改造建立独立的 Reranker module seam：问答流程只提交查询与候选文档，module 隐藏平台协议、鉴权、响应映射和降级逻辑。默认远程 adapter 使用硅基流动免费模型 `BAAI/bge-reranker-v2-m3`，同时保留本地 BGE adapter。Embedding 继续使用 DashScope 的 `qwen3.7-text-embedding`，两者独立配置。

## 范围

本次包含：

- 建立通用 Reranker interface、factory、远程 HTTP adapter 和本地 BGE adapter。
- 将检索 adapter 从 `FlagEmbedding` 模型对象中解耦。
- 远程重排失败时按原始 RRF 顺序降级，并让问答继续。
- 通过 SSE、最终结果和聊天历史传递结构化告警。
- 在前端回答下方持久显示黄色降级提示。
- 为 `.env.example` 和本地 `.env` 的 Embedding、Reranker 配置加入中文注释。

本次不包含：

- 支持所有平台的私有重排协议或 SDK。
- 多级远程 provider 自动切换、重试队列或熔断器。
- 多模态重排、instruction 等模型特有能力。
- 调用真实远程接口的收费集成测试。

## Module seam

Reranker module 对调用方暴露一个稳定 interface：接收 `query` 和候选文档文本，返回结构化 `RerankOutcome`。调用方不感知 `FlagEmbedding`、HTTP URL、Bearer 鉴权、批次或平台响应字段。

```text
问答编排
  -> Retrieval module
       -> Reranker interface
            -> Local BGE adapter
            -> 通用 HTTP /v1/rerank adapter
                  -> 硅基流动
                  -> 其他兼容平台
```

`RerankOutcome` 包含：

- `items`：每项包含原输入索引与相关性分数。
- `degraded`：是否使用了降级结果。
- `warning_code`：稳定的机器可读告警代码。
- `warning_message`：可直接展示的中文文案。

远程 adapter 支持 Bearer 鉴权、`POST /rerank` 请求以及 `results[].index`、`results[].relevance_score` 响应。它按 `index` 恢复候选文档映射，不依赖服务端返回文档正文。该兼容范围刻意保持有限，平台私有字段不进入公开 interface。

本地 adapter 仅在 `RERANKER_ADAPTER=local` 时导入并懒加载 `FlagEmbedding`。远程部署不会因为 import 链加载本地模型重依赖。

## 成功与降级数据流

成功流程：

1. `node_rerank` 整理本地召回与网页搜索候选。
2. Retrieval module 调用 Reranker module。
3. adapter 返回所有候选的原索引和相关性分数。
4. Retrieval module 将分数附加到文档副本并按分数降序排列。
5. `node_rerank` 执行现有分数断崖 Top-K。

降级流程：

1. 请求超时、网络异常、HTTP 429、HTTP 5xx、非法 JSON 或响应结构无效时，Reranker module 返回降级 outcome。
2. 降级结果保留原始 RRF 顺序，最多取前 10 条，不伪造重排分数。
3. `node_rerank` 跳过依赖分数的断崖 Top-K，避免当前降级文档缺少 `score` 后再次异常并清空上下文的问题。
4. 问答继续生成，图状态写入结构化 `warnings`。

告警结构为：

```json
{
  "code": "reranker_degraded",
  "message": "重排序服务暂时不可用，本次回答已使用原始检索顺序生成"
}
```

鉴权失败、400 请求错误和配置缺失也不得清空上下文；它们同样降级并记录详细后端日志，但前端只显示稳定且不泄漏密钥或内部响应的通用文案。

## 告警传递与前端展示

`QueryGraphState` 新增 `warnings` 列表。流式模式在降级发生时立即发送 `warning` SSE 事件，并在最终 `final` 事件再次携带 `warnings`，避免前端错过早期事件。非流式模式从最终聊天消息读取相同字段。

助手消息写入 MongoDB 时持久化 `warnings`，会话详情接口原样返回。因此刷新页面或重新打开历史会话后，提示仍然存在。

前端在回答正文下方、引用来源上方显示黄色提示条：

> 重排序服务暂时不可用，本次回答已使用原始检索顺序生成

提示不写入回答 Markdown，不使用短暂 Toast。前端按 `warning.code` 去重，防止实时 `warning` 与 `final.warnings` 重复显示。

## 配置

Embedding 使用现有 DashScope adapter：

```env
# 使用阿里云 DashScope 远程生成稠密和稀疏向量，本地不加载 BGE-M3
EMBEDDING_ADAPTER=dashscope
EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/api/v1
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

Reranker 使用通用 HTTP adapter：

```env
# 使用兼容 POST /v1/rerank 协议的远程重排序平台
RERANKER_ADAPTER=http
# 硅基流动提供的免费多语言重排序模型
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_BASE_URL=https://api.siliconflow.cn/v1
# 远程重排序密钥；未设置时兼容读取 SILICONFLOW_API_KEY
RERANKER_API_KEY=your_siliconflow_api_key_here
# 超过该秒数后保留原始检索顺序并向前端显示降级提示
RERANKER_REQUEST_TIMEOUT=8
# 单次最多发送给重排序平台的候选文档数
RERANKER_MAX_DOCUMENTS=20
```

本地兼容配置保留 `BGE_RERANKER_LARGE`、`BGE_RERANKER_DEVICE` 和 `BGE_RERANKER_FP16`。`.env.example` 使用占位密钥；本地 `.env` 添加中文注释但不改写用户已有真实密钥，也不提交 Git。

## 响应校验

HTTP adapter 必须校验：

- `results` 是列表且结果数量与实际发送的文档数量一致。
- 每项存在整数 `index` 和数值 `relevance_score`。
- 索引不重复、不越界，并完整覆盖输入索引。
- 分数可安全转换为有限浮点数。

任一校验失败均触发统一降级，避免部分结果、错位索引或异常数值污染排序。

## 测试策略

Reranker module interface 是主要测试面，覆盖：

- 远程模式不导入或加载 `FlagEmbedding`。
- 本地 adapter 只初始化一次并保持懒加载。
- 请求 URL、Bearer 鉴权、模型名、文档数组和 `top_n` 正确。
- 响应按原索引映射，文档元数据不丢失。
- 重复索引、越界索引、缺失分数、非有限分数和数量不一致触发降级。
- 超时、429、5xx、网络异常和非法 JSON 触发原顺序降级。
- 空查询和空文档行为明确且不调用远程接口。
- 降级时跳过分数断崖算法，并保留最多 10 条上下文。
- `warning` SSE、`final.warnings`、MongoDB 历史返回和前端黄色提示完整贯通。

HTTP 测试使用注入的模拟客户端，不调用真实硅基流动接口。最终运行后端相关测试、Python 编译检查、前端构建和 `git diff --check`。

## 成功标准

- 2 核 2G 远程模式启动和查询过程中不加载本地 Embedding 或 Reranker 模型。
- 默认使用 DashScope `qwen3.7-text-embedding` 与硅基流动 `BAAI/bge-reranker-v2-m3`。
- 更换兼容重排平台时，只需更改模型、地址和密钥配置。
- 远程重排不可用时问答继续，且当前回答与历史会话都能看到明确降级提示。
- 本地模式仍可通过配置启用，现有调用方无需了解具体 adapter。
