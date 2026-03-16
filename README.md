# 企业知识库问答 Agent


## 项目结构
```text
agent_rag_github_ready/
├── app.py
├── ingest.py
├── evals.py
├── prompts.py
├── rag_store.py
├── requirements.txt
├── .env.example
├── kb_docs/
│   ├── company_policy.md
│   └── agent_design.md
├── evals/
│   └── sample_eval.json
├── data/
│   └── kb_index.json   # 执行 ingest 后自动生成
├── logs/
│   └── app.log         # 运行后自动生成
└── static/
    └── index.html
```

## 快速开始
### 1）创建虚拟环境
```bash
python -m venv .venv
```

### 2）激活虚拟环境
Windows：
```bash
.venv\Scripts\activate
```

Linux / macOS：
```bash
source .venv/bin/activate
```

### 3）安装依赖
```bash
pip install -r requirements.txt
```

### 4）配置环境变量
```bash
cp .env.example .env
```
然后编辑 `.env`，填入你自己的 `OPENAI_API_KEY`。

### 5）全量构建知识库索引
```bash
python ingest.py --docs kb_docs
```

### 6）启动项目
```bash
uvicorn app:app --reload
```

启动后可访问：
- API 文档：`http://127.0.0.1:8000/docs`
- 演示页面：`http://127.0.0.1:8000/`

## 增量更新知识库
命令行：
```bash
python ingest.py --docs kb_docs --sync
```

API：
```bash
curl -X POST http://127.0.0.1:8000/admin/sync-index \
  -H "Content-Type: application/json" \
  -d '{
    "docs_dir": "kb_docs",
    "chunk_size": 700,
    "chunk_overlap": 120,
    "delete_missing": true
  }'
```

## 单文档增量写入
```bash
curl -X POST http://127.0.0.1:8000/admin/upsert-document \
  -H "Content-Type: application/json" \
  -d '{
    "source": "faq_remote_work.md",
    "title": "远程办公 FAQ",
    "content": "# 远程办公 FAQ\n\n1. 需要提前报备。\n2. 跨城办公需主管审批。"
  }'
```

## 聊天接口
普通问答：
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "如果知识库证据不足，系统应该如何回答？",
    "session_id": "demo-session"
  }'
```

SSE 流式问答：
```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Agent 项目为什么要优先使用工具检索？",
    "session_id": "demo-stream"
  }'
```

## 评测
仅做检索评测：
```bash
python evals.py
```

同时检查回答与拒答行为：
```bash
python evals.py --with-generation
```

## 这个项目功能简介
- 设计并实现企业知识库问答 Agent，支持多轮对话、流式输出、检索增强生成与引用返回。
- 搭建 RAG 链路，完成文档切分、向量化入库、混合检索、上下文组装与回答生成。
- 基于 Function Calling 封装知识检索、来源查询、片段查询等工具能力，实现模型按任务动态调用外部能力。
- 增加低置信拒答、异常重试、基础提示词注入拦截与管理接口，提升系统稳定性与可演示性。
- 提供增量知识更新和基础评测脚本，支持项目持续迭代。


