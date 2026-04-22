# Context-Retrieval RAG (SlideVQA)

## 完全本地部署

- Elasticsearch
- Qdrant
- Qwen 本地模型服务
  - Qwen3-VL (`vllm`)：OCR + answer
  - Qwen3-Embedding：text embedding
  - Qwen3-VL-Embedding：page-image embedding
  - Qwen3-Reranker：rerank
- Weave（可观测性）
- RAGAS（评测框架）

## 启动依赖

```bash
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.yml ps
```

<!-- 默认端口：

- `http://localhost:8001/v1`：Qwen3-VL（OCR + QA）
- `http://localhost:8002/v1`：Qwen3-Embedding
- `http://localhost:8003/v1`：Qwen3-VL-Embedding
- `http://localhost:8004`：Qwen3-Reranker
- `http://localhost:9200`：Elasticsearch
- `http://localhost:6333`：Qdrant -->

## 环境变量（`.env`）

```dotenv
dataset_name=slidevqa
# dataset_dir=./data/downloads/slidevqa

es_url=http://localhost:9200
qdrant_url=http://localhost:6333

vllm_base_url=http://localhost:8001/v1
vllm_model_name=qwen3-vl
# 可选：RAGAS 单独使用文本模型（建议），避免 VL 在结构化评测输出上不稳定
ragas_llm_base_url=
ragas_llm_model_name=

qwen_embed_base_url=http://localhost:8002/v1
qwen_embed_api_key=EMPTY
qwen_embed_model=Qwen/Qwen3-Embedding-4B
qwen_query_instruction=Represent the query for retrieving relevant document passages.
qwen_document_instruction=Represent the document for retrieval.

qwen_multimodal_embed_base_url=http://localhost:8003/v1
qwen_multimodal_embed_api_key=EMPTY
qwen_multimodal_embed_model=Qwen/Qwen3-VL-Embedding-2B
qwen_multimodal_query_instruction=Represent the query for retrieving relevant document pages.
qwen_multimodal_document_instruction=Represent the document page for retrieval.

qwen_rerank_base_url=http://localhost:8004
qwen_rerank_api_key=
qwen_rerank_model=Qwen/Qwen3-Reranker-0.6B
qwen_rerank_query_instruction=

qdrant_text_vector_size=2560
qdrant_image_vector_size=2048

top_k=20
rerank_top_k=5

# Retrieval/build knobs
use_page_image=true
build_page_workers=4
build_llm_concurrency=6
build_embed_concurrency=2
text_embed_batch_size=256
image_embed_batch_size=64
qdrant_upsert_batch_size=1024
es_bulk_ops_batch_size=4000
ocr_max_tokens=2048
contextual_chunk_max_tokens=512
page_proxy_max_tokens=512

wandb_api_key=...
weave_project=context-rag
```

## 数据准备（SlideVQA）

下载SlideVQA

```
hf download NTT-hil-insight/SlideVQA --repo-type=dataset --local-dir /data/SlideVQA
```

准备数据

```bash
uv sync

uv run scripts/prepare_eval_dataset.py --slidevqa-dir /data/SlideVQA --slidevqa-split test --overwrite

uv run scripts/validate_eval_dataset.py --dataset-dir data/downloads/slidevqa
```

<!-- 默认会生成：

- `data/downloads/slidevqa/samples.jsonl`
- `data/downloads/slidevqa/corpus_pages.jsonl`

每条样本包含：

- `retrieval_scope_doc_page_ids`：该问题对应 deck 的 20 页
- `retrieval_scope_document_ids`：该问题对应 deck id

评测时检索会自动按这个 scope 过滤，即每条问题只在自己的 20 页中检索。 -->

## 运行

<!-- ### 1. 建库

```bash
uv run run-slidevqa-experiment build --variant baseline --rebuild
uv run run-slidevqa-experiment build --variant enhanced --rebuild
```

### 2. 评测

```bash
uv run run-slidevqa-experiment eval --variant baseline
uv run run-slidevqa-experiment eval --variant enhanced

# 如需关闭 RAGAS（仅跑 EM/F1/Recall）
uv run run-slidevqa-experiment eval --variant baseline --no-with-ragas
``` -->

建库 + 评测

```bash
uv run run-slidevqa-experiment run --variant baseline --rebuild
uv run run-slidevqa-experiment run --variant enhanced --rebuild
```

<!-- ## 输出

- 建库产物：`data/experiments/slidevqa/<variant>/`
  - `pages.jsonl`
  - `chunks.jsonl`
  - `manifest.json`
- 评测结果：`data/eval_runs/`
  - `<run_id>.jsonl`（逐样本）
  - `<run_id>.ragas.jsonl`（逐样本 RAGAS 分数）
  - `<run_id>.summary.json`（汇总） -->

## 指标

- 检索层：`Recall@5 (doc_id + page_no)`
- 生成层：`EM`、`F1`
- RAGAS：
  - `faithfulness`（`ragas_faithfulness`）
  - `answer_relevancy`（`ragas_answer_relevancy`）
  - `context_precision`（`ragas_context_precision`）
  - `context_recall`（`ragas_context_recall`）
  - `answer_correctness`（`ragas_answer_correctness`）
- 性能：`avg_latency_s`、`p95_latency_s`
