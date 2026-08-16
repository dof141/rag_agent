# 远程向量导入运维说明

本文档只记录正式远程向量上传链路的部署与回滚步骤，不包含任何真实密钥、token、URL 或账号密码。

## 环境变量

以下变量由部署平台注入，示例值统一留空或使用占位符：

```text
RAG_SQLITE_PATH=<由部署平台注入>
RAG_ADMIN_USERNAME=<由部署平台注入>
RAG_ADMIN_PASSWORD=<由部署平台注入>
RAG_JWT_SECRET=<由部署平台注入>
RAG_JWT_TTL_SECONDS=<由部署平台注入>
RAG_SETTINGS_MASTER_KEY=<由部署平台注入>
RAG_OUTPUT_ROOT=<由部署平台注入>
```

首次启动时，服务只在 SQLite 中不存在管理员时创建一个管理员账号。后续启动不会覆盖已有管理员用户名或密码；如需轮换密码，应通过受控维护流程处理数据库记录，而不是依赖环境变量覆盖。

## 配置流程

1. 启动后端和前端服务。
2. 使用管理员账号登录前端。
3. 进入“运行配置”页面。
4. 保存 SiliconFlow embedding 配置与所选向量库配置。
5. 密钥输入框留空表示保留旧密钥；清除密钥必须点击单独的清除按钮。

当前最小远程链路支持 `siliconflow + qdrant`：SiliconFlow 生成 dense 向量，Qdrant Cloud Inference 生成 `Qdrant/bm25` sparse 数据。Qdrant 当前仅支持导入链路，查询迁移将在下一阶段完成。

## Milvus 重建

Milvus 保留为可选 adapter。schema 重建必须显式运行维护命令，并传入确认短语；应用启动和首次导入都不会自动删除 collection。

重建前应确认旧向量数据可以丢弃。该命令只负责 schema 清理和重建，不执行旧数据迁移。

在仓库根目录执行以下命令，其中用户 UUID 使用待重建配置所属管理员的稳定 `user_id`：

```powershell
uv run python -m app.tools.rebuild_milvus_collections --user-id <用户UUID> --confirm DROP_AND_RECREATE_MILVUS_VECTOR_COLLECTIONS
```

命令只有在 `--confirm` 与上述确认短语完全一致，且该用户当前保存了有效 Milvus 配置时才会删除并重建两个 collection。缺少参数、确认短语不匹配或当前配置不是 Milvus 时，命令以非零状态退出且不执行删除。

## 验证

无真实远程凭据时，仅运行自动化测试和无凭据 UI 验收；自动化测试不得访问 SiliconFlow、Qdrant Cloud 或 Milvus 远程实例。

正式远程验收由管理员在前端输入凭据并上传小型 Markdown 文档完成。验收日志只允许记录阶段、耗时、向量数量、维度和写入数量，不得记录密码、JWT、API Key、数据库 token、Authorization header、配置快照或完整向量。

## 回滚

本链路按 Task 形成多个中文提交。需要回滚时，从最新提交开始按反序执行：

```powershell
git revert <提交哈希>
```

禁止使用 `git reset --hard`，也不要覆盖用户未提交的 `main.py`、`.env.example`、被忽略的 `.env` 或手工探针文件。
