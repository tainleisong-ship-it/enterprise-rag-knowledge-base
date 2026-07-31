# Enterprise RAG Knowledge Base

基于 **混合检索（BM25 + 向量）+ 规则重排序** 的企业级 RAG 问答系统后端，使用 DeepSeek 大模型生成答案。

## 核心特性

- **混合检索**：BM25 关键词检索 + FAISS 向量相似度检索，双路召回后加权融合
- **规则重排序**：基于精确匹配、关键词重叠、位置因子、长度因子的模拟 Reranker，无需额外模型文件
- **缓存降级**：Redis 查询缓存，连接失败时自动降级为无缓存模式，服务不中断
- **一键运行**：PyCharm 中直接右键运行 `app/main.py`，自动启动 Uvicorn 并打开浏览器跳转 Swagger 文档
- **模块化架构**：检索、重排序、链路、缓存逻辑完全分离，易于替换和扩展

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| Web 框架 | FastAPI + Uvicorn |
| 编排链路 | LangChain (LCEL) |
| 大模型 | DeepSeek (`deepseek-chat`) |
| Embedding | HuggingFace 本地模型 (`BAAI/bge-small-zh-v1.5`) |
| 向量存储 | FAISS (本地内存) |
| 关键词检索 | rank-bm25 |
| 缓存 | Redis (可降级) |
| 配置管理 | pydantic-settings |
| 数据校验 | Pydantic v2 |

## 架构说明
```
用户问题
│
▼
┌──────────────┐ 命中
│ Redis 缓存 │──────────────────► 直接返回结果
└──────┬───────┘
│ 未命中
▼
┌──────────────────────────────────────────────┐
│ HybridRetriever │
│ ┌──────────────┐ ┌────────────────────┐ │
│ │ BM25 检索 │ │ FAISS 向量检索 │ │
│ │ (关键词召回) │ │ (语义召回) │ │
│ └──────┬───────┘ └─────────┬──────────┘ │
│ └────── 加权融合 ──────┘ │
│ (0.6*vec + 0.4*bm25) │
└──────────────────────┬───────────────────────┘
│ Top-K 候选
▼
┌────────────────┐
│ RuleBasedReranker │
│ (规则重排序) │
└────────┬───────┘
│ Top-K 精排
▼
┌────────────────┐
│ DeepSeek LLM │
│ (答案生成) │
└────────┬───────┘
│
▼
写入 Redis 缓存
│
▼
返回响应
```

## 项目结构

```
enterprise-rag-knowledge-base/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI 入口 + Uvicorn 一键启动 + 自动开浏览器
│ ├── api/
│ │ ├── init.py
│ │ └── schemas.py # Pydantic v2 请求/响应模型
│ ├── core/
│ │ ├── init.py
│ │ └── config.py # pydantic-settings 配置管理
│ ├── rag/
│ │ ├── init.py
│ │ ├── retriever.py # BM25 + 向量混合检索
│ │ ├── reranker.py # 规则模拟重排序
│ │ ├── chain.py # RAG 链路编排
│ │ └── store.py # 文档加载 + FAISS 向量库初始化
│ └── services/
│ ├── init.py
│ └── cache.py # Redis 缓存封装（含降级）
├── data/
│ └── knowledge_base.txt # 模拟企业知识库数据
├── requirements.txt
├── .env.example
└── README.md
```