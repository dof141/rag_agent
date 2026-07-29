# 🚀 多模态 GraphRAG 智能知识库系统 (RAG Agent)

本系统是一套高性能、支持多模态文档解析与图谱扩展的工业级 **Retrieval-Augmented Generation (RAG)** 系统。结合了 **LangGraph Workflow 工作流引擎**、**MinerU 高精文档解析**、**VLM 视觉大模型多线程图文理解**、**BGE-M3 向量嵌入与重排序**、**MongoDB 状态持久化与自愈**，以及 **Milvus / Neo4j 混合检索**。

---

## 🌟 核心功能特性

1. **📄 多模态文档智能解析流水线 (Import Workflow)**
   - **MinerU 高精转换**：支持 PDF、Docx、PPT 等多格式转 Markdown，自动提取高清晰度图表与公式。
   - **VLM 多线程图片总结**：采用多线程并发 (`ThreadPoolExecutor`) 调用视觉大模型，自动生成图片的结构化 ALT 描述并回填至 Markdown。
   - **标题与语义切片**：基于层级 Markdown 标题进行语义切割，保留上下文链接。
   - **BGE-M3 混合向量生成**：提取 Dense & Sparse 文本向量，批量写入 Milvus 向量数据库。

2. **🤖 智能查询与 HyDE 增强流水线 (Query Workflow)**
   - **HyDE 假设性文档扩展**：通过 LLM 生成回答假设，大幅提升跨语义召回匹配度。
   - **BGE Reranker 重排序**：精细化相关性二次打分，过滤无关切片。
   - **MCP WebSearch 联网增强**：本地知识库无结果时，自动触发 MCP 联网搜索补充实时信息。
   - **打字机流式回答**：流式生成带有引用标记的最终答案。

3. **🛡️ 状态持久化与断点自愈 (Auto-Healing & Retry)**
   - **MongoDB 任务轮转**：支持全流程状态落盘 (`kb002.import_tasks`)。
   - **启动自愈恢复**：服务异常崩溃重启时，自动清理中断的旧任务。
   - **前端一键重试**：节点报错时支持前端一键从 Node 1 无缝重启动流。

---

## 🏗️ 系统整体架构图

```mermaid
graph TD
    User([前端 Vue3/Vite]) -->|HTTP / Stream| API[FastAPI 后端服务]
    
    subgraph ImportFlow ["导入工作流 (Import LangGraph)"]
        API --> NodeEntry[Node 1: 文件校验 & 类型判断]
        NodeEntry --> NodeMinerU[Node 2: MinerU PDF/Docx 解析]
        NodeMinerU --> NodeVLM[Node 3: VLM 视觉多线程处理]
        NodeVLM --> NodeSplit[Node 4: 标题与语义粗切分]
        NodeSplit --> NodeEmbedding[Node 5: BGE-M3 向量生成]
        NodeEmbedding --> Milvus[(Milvus 向量库)]
        NodeEntry -.-> SyncMongo[(MongoDB 状态持久化)]
    end

    subgraph QueryFlow ["查询工作流 (Query LangGraph)"]
        API --> NodeHyDE[Node Query: HyDE 拓展]
        NodeHyDE --> NodeVectorSearch[向量检索 (Milvus)]
        NodeVectorSearch --> NodeRerank[BGE Reranker 重排]
        NodeRerank --> NodeMCP[MCP 联网补全]
        NodeMCP --> NodeLLM[LLM 流式输出]
    end
```

---

## 📦 技术栈清单

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **后端框架** | FastAPI + Python 3.10+ | 异步高效 RESTful 接口与流式响应 |
| **工作流引擎** | LangGraph | 状态机图流控引擎 |
| **向量数据库** | Milvus (v2.4+) | 存储与检索 1024 维 BGE-M3 向量 |
| **图数据库** | Neo4j (v5.26+) | 知识图谱实体与拓扑关系存储 |
| **状态持久化** | MongoDB (v6.0+) | 存储任务节点状态与全流程 Log |
| **对象存储** | MinIO | 本地私有化图片与解析产物文件存储 |
| **嵌入/重排** | BAAI/bge-m3, BAAI/bge-reranker-large | 文本向量化与二次相关性重排 |
| **多模态 VLM** | SenseNova / Qwen-VL / Qwen-Omni | 图像识别与多图多线程并发解析 |
| **前端界面** | Vue 3 + TypeScript + Vite + Lucide Icons | 现代化双模式 (导入/问答) 交互 UI |

---

## 🛠️ 部署与环境准备

### 1. 基础依赖服务启动 (Docker Compose)

在部署后端前，请确保启动 **Milvus**、**Neo4j**、**MongoDB** 以及 **MinIO**。

推荐使用 `docker-compose.yml` 启动核心依赖服务：

```yaml
version: '3.8'

services:
  # Milvus 向量数据库
  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"

  # Neo4j 图数据库
  neo4j:
    image: neo4j:5.26-community
    container_name: neo4j-container
    environment:
      - NEO4J_AUTH=neo4j/your_rotated_neo4j_password_here
    ports:
      - "7474:7474"
      - "7687:7687"

  # MongoDB 数据库
  mongodb:
    image: mongo:6.0
    container_name: mongodb-container
    ports:
      - "27017:27017"

  # MinIO 对象存储
  minio:
    image: minio/minio
    container_name: minio-container
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: your_rotated_minio_password_here
    ports:
      - "9000:9000"
      - "9001:9001"
```

---

### 2. 后端部署 (Python FastAPI)

#### 步骤 1：克隆代码并创建 Python 虚拟环境

```bash
git clone https://github.com/dof141/rag_agent.git
cd rag_agent

# 使用 uv 或 conda 创建 Python 3.10 环境
uv venv .venv
# 激活环境 (Windows)
.venv\Scripts\activate
# 安装依赖
uv pip install -r requirements.txt
```

#### 步骤 2：配置环境变量 (`.env`)

在 `rag_agent` 根目录下创建 `.env` 文件（或修改已现有的配置文件）：

```ini
# ====== 1. LLM 文本大模型配置 ======
OPENAI_BASE_URL=http://198.18.0.1:20128/v1
OPENAI_API_KEY=your_rotated_openai_key_here
LLM_DEFAULT_MODEL=sensenova/deepseek-v4-flash
LLM_DEFAULT_TEMPERATURE=0.1

# ====== 2. VLM 视觉多模态大模型配置 ======
VLM_BASE_URL=http://198.18.0.1:20128/v1
VLM_API_KEY=your_rotated_vlm_key_here
VL_MODEL=sensenova/sensenova-6.7-flash-lite
VLM_TIMEOUT=60.0

# ====== 3. BGE 向量模型配置 ======
BGE_M3_PATH=F:/ai_models/models/BAAI/bge-m3
BGE_DEVICE=cpu

# ====== 4. Milvus 配置 ======
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks

# ====== 5. Neo4j 配置 ======
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_rotated_neo4j_password_here

# ====== 6. MongoDB 配置 ======
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=kb002

# ====== 7. MinIO 配置 ======
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=your_rotated_minio_password_here
MINIO_BUCKET_NAME=knowledge-base-files

# ====== 8. MinerU 配置 ======
MINERU_API_TOKEN=your_mineru_api_token_here
MINERU_BASE_URL=https://mineru.net/api/v4
```

#### 步骤 3：启动 FastAPI 后端服务

```bash
uvicorn app.query_process.api.query_server:app --host 0.0.0.0 --port 8000 --reload
```

后端服务启动后，接口文档地址为：`http://localhost:8000/docs`

---

### 3. 前端部署 (Vue 3 + Vite)

#### 步骤 1：安装依赖与构建

```bash
cd frontend

# 安装依赖
npm install

# 启动本地开发服务
npm run dev

# 构建生产产物
npm run build
```

#### 步骤 2：Nginx 部署生产静态文件 (可选)

```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API 接口代理转发
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## ⚡ 性能与并发参数调优指南

1. **VLM 并发线程数调整**：
   - 配置文件：`app/import_process/agent/nodes/node_md_img.py`
   - 参数：`max_workers = min(5, total_count)`
   - 说明：在保证代理/第三方大模型 API 不触发 `500 internal_error` 的前提下，可将并发调高至 3~5 线程，大图提速 4 倍以上。

2. **向量检索与 Rerank 过滤**：
   - 向量匹配数：`top_k = 10`
   - 重排序筛选：`rerank_top_k = 3`，Score 阈值限制大于 `0.35`。

---

## 📝 许可证与贡献

本项目采用 MIT 许可证，欢迎提交 Issue 与 Pull Request 共同优化提升性能！
