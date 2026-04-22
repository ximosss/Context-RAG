# Context-RAG for SlideVQA

多模态RAG。

- 检索范围固定在每题对应 deck 的 `20` 页内
- 生成阶段只使用 `top1` 返回页
- 只喂给 VLM：
  - `top1` 页图
  - `top1` raw OCR chunk
- answer model：`qwen3.5-35b-a3b`

## 主要结果

| Variant | EM | F1 | Recall@5 | 
| --- | ---: | ---: | ---: |
| baseline | 0.7222 | 0.7222 | 0.9861 | 
| enhanced | 0.8056 | 0.8056 | 0.9861 | 

当前代码对应的核心差异：

- `baseline`：原始 OCR chunk 检索
- `enhanced`：`contextual chunk + page proxy` 检索增强

在 `top1 image + top1 chunk` 的更严格回答设定下，`enhanced` 明显优于 `baseline`。

## 启动

只启动当前需要的服务：

```bash
docker compose up -d elasticsearch qdrant qwen_vl qwen_embed qwen_multimodal_embed
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



