# Context-Multimodal-RAG

多模态 RAG 实验项目，用于验证在“问题来自单个 slide deck、检索范围可控、回答依赖页面图像与 OCR 文本”的场景下，检索增强策略能否稳定提升最终问答效果。

项目搭建可复现的最小闭环：数据准备、索引构建、检索召回、回答生成与评测都可以通过命令行直接运行，便于对比 `baseline` 和 `enhanced` 两种检索方案在同一数据集、同一生成设定下的差异。

当前实验设定，核心约束如下：

- 检索范围固定在每题对应 deck 的 `20` 页内
- 生成阶段只使用 `top1` 返回页
- 只喂给 VLM：
  - `top1` 页图
  - `top1` raw OCR chunk
- answer model：`qwen3.5-35b-a3b`

## 主要结果

<<<<<<< HEAD
| Variant | EM | F1 | Recall@5 | 
| --- | ---: | ---: | ---: |
| baseline | 0.7222 | 0.7222 | 0.9861 | 
| enhanced | 0.8056 | 0.8056 | 0.9861 | 
=======
测试用的数据集为[SlideVQA](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA)部分子集

| 方案 | EM | F1 | Recall@5 | 平均延迟 | P95 延迟 | 
| --- | ---: | ---: | ---: | ---: | ---: | 
| baseline | 69.44% | 71.53% | 98.61% | 9.784s | 17.133s | 
| enhanced | 80.56% | 80.66% | 98.61% | 10.044s | 17.094s | 

更多细节参考项目的报告: https://wandb.ai/ximo_ml/context-rag/reports/Context-RAG-SlideVQA-2026-04-15---VmlldzoxNjY0NDgzOQ
>>>>>>> 0fb6b9f (update readme)

当前代码对应的核心差异：

- `baseline`：原始 OCR chunk 检索
- `enhanced`：`contextual chunk + page proxy` 检索增强

在 `top1 image + top1 chunk` 的更严格回答设定下，`enhanced` 明显优于 `baseline`。

## 具体框架

- `Elasticsearch`：承担词法检索，提供 OCR chunk 的 BM25 召回能力。
- `Qdrant`：承担向量检索，统一存储文本与页面级多模态 embedding。
- `Qwen3-Embedding-4B`：生成查询和文本 chunk 的 embedding，用于语义检索。
- `Qwen3-VL-Embedding-2B`：生成页面图像与 page proxy 的多模态 embedding，用于页面级检索增强。
- `Qwen3.5-35B-A3B`：负责最终回答生成，也用于部分检索增强文本的构造。
- `Reciprocal Rank Fusion (RRF)`：融合 BM25 与向量检索结果，减少单一路径召回偏差。
- `contextual chunk + page proxy`：在原始 OCR chunk 之外补充更适合检索的上下文化文本与页面代理表示，是 `enhanced` 方案的核心。

## 启动

只启动当前需要的服务：

```bash
docker compose up elasticsearch qdrant qwen_vl qwen_embed qwen_multimodal_embed
docker compose ps
```

## 数据准备

```bash
uv sync
hf download NTT-hil-insight/SlideVQA --repo-type=dataset --local-dir /data/SlideVQA
uv run scripts/prepare_eval_dataset.py --slidevqa-dir /data/SlideVQA --slidevqa-split test --overwrite
```

## 建库

```bash
uv run run-slidevqa-experiment build --variant baseline --rebuild
uv run run-slidevqa-experiment build --variant enhanced --rebuild
```

## 评测

```bash
uv run run-slidevqa-experiment eval --variant baseline
uv run run-slidevqa-experiment eval --variant enhanced
```


