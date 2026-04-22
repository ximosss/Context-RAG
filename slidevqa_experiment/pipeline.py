from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path

from qdrant_client import models as qdrant_models

from .clients import (
    RuntimeClients,
    WeaveTracker,
    bm25_search,
    embed_images,
    embed_multimodal_texts,
    embed_texts,
    ensure_storage,
    qdrant_search,
    rerank_documents,
    vllm_chat,
)
from .config import Settings
from .prompts import ANSWER_PROMPT, ANSWER_REWRITE_PROMPT, CONTEXTUAL_CHUNK_PROMPT, OCR_PAGE_PROMPT, PAGE_PROXY_PROMPT
from .schemas import Chunk, EvalSample, PageRecord
from .utils import (
    approx_tokens,
    candidate_to_rerank_document,
    clean_generated_text,
    chunk_page,
    extract_short_answer,
    first_non_empty_line,
    is_noise_chunk,
    looks_like_reasoning_leak,
    read_jsonl,
    should_trust_env,
    strip_answer_citations,
    write_jsonl,
)

logger = logging.getLogger("slidevqa_experiment")

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

RERANK_DOCUMENT_TOKEN_BUDGET = 500
ANSWER_MAX_HITS = 1
ANSWER_MAX_CHUNKS_PER_HIT = 1
ANSWER_MAX_IMAGES = 1
ANSWER_PROMPT_TOKEN_BUDGET = 6000
ANSWER_RAW_TOKEN_BUDGET = 300
RAGAS_MAX_CONTEXTS = 3
RAGAS_CONTEXT_TOKEN_BUDGET = 300
RAGAS_RESPONSE_TOKEN_BUDGET = 200
RAGAS_REFERENCE_TOKEN_BUDGET = 80


def _positive_int(value: int, fallback: int = 1) -> int:
    return value if value > 0 else fallback


def _stable_point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"context-rag:{raw_id}"))


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    words = normalized.split()
    if len(words) <= token_budget:
        return normalized
    return " ".join(words[:token_budget]).strip()


def _fit_blocks_to_token_budget(blocks: list[str], token_budget: int, min_remaining: int = 32) -> list[str]:
    if token_budget <= 0:
        return []
    selected: list[str] = []
    used = 0
    for block in blocks:
        normalized = str(block or "").strip()
        if not normalized:
            continue
        block_tokens = approx_tokens(normalized)
        if block_tokens <= 0:
            continue
        if used + block_tokens <= token_budget:
            selected.append(normalized)
            used += block_tokens
            continue
        remaining = token_budget - used
        if remaining < min_remaining:
            break
        trimmed = _truncate_to_token_budget(normalized, token_budget=remaining)
        if trimmed:
            selected.append(trimmed)
        break
    return selected


def _source_rrf_weight(settings: Settings, source: str) -> float:
    if source == "bm25":
        return max(0.0, float(settings.bm25_rrf_weight))
    if source == "dense":
        return max(0.0, float(settings.dense_rrf_weight))
    if source == "page_image":
        return max(0.0, float(settings.image_rrf_weight))
    if source == "page_proxy":
        return max(0.0, float(settings.proxy_rrf_weight))
    if source == "page_proxy_dense":
        return max(0.0, float(settings.proxy_dense_rrf_weight))
    return 1.0


class _RagasRelevancyEmbeddingAdapter:
    """Adapter for older RAGAS answer_relevancy metric expecting LangChain-style methods."""

    def __init__(self, embeddings: object):
        self._embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        if hasattr(self._embeddings, "embed_query"):
            return self._embeddings.embed_query(text)  # type: ignore[attr-defined]
        if hasattr(self._embeddings, "embed_text"):
            return self._embeddings.embed_text(text)  # type: ignore[attr-defined]
        raise AttributeError("Embedding object has neither embed_query nor embed_text")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._embeddings, "embed_documents"):
            return self._embeddings.embed_documents(texts)  # type: ignore[attr-defined]
        if hasattr(self._embeddings, "embed_texts"):
            return self._embeddings.embed_texts(texts)  # type: ignore[attr-defined]
        return [self.embed_query(text) for text in texts]


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def _compact_ragas_context(hit: dict, token_budget: int) -> str:
    page_no = int(hit.get("page_no", 0) or 0)
    title = _normalize_whitespace(str(hit.get("page_title") or "(none)"))
    section = _normalize_whitespace(str(hit.get("section_path") or "(none)"))
    raw_text = _normalize_whitespace(_truncate_to_token_budget(str(hit.get("text_raw") or ""), token_budget=max(40, token_budget - 40)))
    contextual_text = _normalize_whitespace(
        _truncate_to_token_budget(str(hit.get("contextual_chunk_text") or ""), token_budget=max(24, token_budget // 3))
    )
    parts = [
        f"Page: {page_no}",
        f"Title: {title}",
        f"Section: {section}",
        f"Text: {raw_text or '(none)'}",
    ]
    if contextual_text:
        parts.append(f"Context: {contextual_text}")
    return _truncate_to_token_budget("\n".join(parts), token_budget=token_budget)


def _dedupe_non_empty(values: list[str] | None, limit: int | None = None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    output: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def _compute_qa_scores(rows: list[dict]) -> tuple[list[dict[str, float]], dict[str, float]]:
    import evaluate

    squad_metric = evaluate.load("squad")
    predictions: list[dict] = []
    references: list[dict] = []

    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"row-{idx}")
        answers = [str(item).strip() for item in row.get("answers", []) if str(item).strip()]
        if not answers:
            answers = [""]
        predictions.append(
            {
                "id": sample_id,
                "prediction_text": str(row.get("prediction", "")),
            }
        )
        references.append(
            {
                "id": sample_id,
                "answers": {"text": answers, "answer_start": [0] * len(answers)},
            }
        )

    aggregate_raw = squad_metric.compute(predictions=predictions, references=references)
    aggregate = {
        "exact_match": float(aggregate_raw["exact_match"]) / 100.0,
        "f1": float(aggregate_raw["f1"]) / 100.0,
    }

    per_row: list[dict[str, float]] = []
    for prediction, reference in zip(predictions, references):
        sample_raw = squad_metric.compute(predictions=[prediction], references=[reference])
        per_row.append(
            {
                "exact_match": float(sample_raw["exact_match"]) / 100.0,
                "f1": float(sample_raw["f1"]) / 100.0,
            }
        )

    return per_row, aggregate


def _compute_recall_at_5_scores(rows: list[dict]) -> tuple[list[float], float]:
    import pytrec_eval

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    qids: list[str] = []

    for idx, row in enumerate(rows):
        qid = f"q{idx}"
        qids.append(qid)
        gold_ids = _dedupe_non_empty([str(item) for item in row.get("relevant_doc_page_ids", [])])
        retrieved_ids = _dedupe_non_empty([str(item) for item in row.get("retrieved_doc_page_ids", [])], limit=5)
        if not gold_ids:
            # pytrec_eval requires at least one relevance judgment per query.
            gold_ids = ["__missing_gold__"]
        qrels[qid] = {doc_id: 1 for doc_id in gold_ids}
        run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(retrieved_ids)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recall.5"})
    raw_scores = evaluator.evaluate(run)
    per_row = [float(raw_scores.get(qid, {}).get("recall_5", 0.0)) for qid in qids]
    aggregate = sum(per_row) / len(per_row) if per_row else 0.0
    return per_row, aggregate


def load_samples(settings: Settings, max_samples: int | None = None) -> list[EvalSample]:
    rows = read_jsonl(settings.dataset_dir / "samples.jsonl")
    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        if max_samples is not None and len(samples) >= max_samples:
            break
        question = str(row.get("question", "")).strip()
        answers = row.get("answers") if isinstance(row.get("answers"), list) else []
        document_id = str(row.get("document_id", "")).strip()
        page_no = int(row.get("page_no", 0))
        image_rel = str(row.get("image_path", "")).strip()
        image_path = str((settings.dataset_dir / image_rel).resolve()) if image_rel else ""
        raw_relevant_doc_page_ids = row.get("relevant_doc_page_ids")
        relevant_doc_page_ids = (
            [str(item).strip() for item in raw_relevant_doc_page_ids if str(item).strip()]
            if isinstance(raw_relevant_doc_page_ids, list)
            else []
        )
        raw_scope_doc_page_ids = row.get("retrieval_scope_doc_page_ids")
        retrieval_scope_doc_page_ids = [
            str(item).strip() for item in raw_scope_doc_page_ids if str(item).strip()
        ] if isinstance(raw_scope_doc_page_ids, list) else []
        raw_scope_document_ids = row.get("retrieval_scope_document_ids")
        retrieval_scope_document_ids = [
            str(item).strip() for item in raw_scope_document_ids if str(item).strip()
        ] if isinstance(raw_scope_document_ids, list) else []
        if not relevant_doc_page_ids and document_id and page_no > 0:
            relevant_doc_page_ids = [f"{document_id}:{page_no:04d}"]
        if not question or not answers or not relevant_doc_page_ids:
            raise ValueError(f"Invalid sample at row {idx}: {row}")
        samples.append(
            EvalSample(
                sample_id=str(row.get("sample_id") or f"row-{idx}"),
                question=question,
                answers=answers,
                document_id=document_id,
                page_no=page_no,
                image_path=image_path,
                relevant_doc_page_ids=relevant_doc_page_ids,
                retrieval_scope_doc_page_ids=retrieval_scope_doc_page_ids,
                retrieval_scope_document_ids=retrieval_scope_document_ids,
            )
        )
    return samples


def load_corpus_pages(settings: Settings, corpus_limit: int | None = None) -> list[dict]:
    rows = read_jsonl(settings.dataset_dir / "corpus_pages.jsonl")
    pages: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        doc_page_id = str(row.get("doc_page_id", "")).strip()
        document_id = str(row.get("document_id", "")).strip()
        page_no = int(row.get("page_no", 0))
        image_rel = str(row.get("image_path", "")).strip()
        image_path = str((settings.dataset_dir / image_rel).resolve()) if image_rel else ""
        if not doc_page_id or not document_id or page_no <= 0 or not image_path:
            continue
        if doc_page_id in seen:
            continue
        seen.add(doc_page_id)
        pages.append(
            {
                "doc_page_id": doc_page_id,
                "document_id": document_id,
                "page_no": page_no,
                "image_path": image_path,
            }
        )
        if corpus_limit is not None and len(pages) >= corpus_limit:
            break
    return pages


async def build_contextual_chunk(
    runtime: RuntimeClients,
    chunk_text: str,
    doc_title: str,
    page_title: str,
    section_path: str,
    page_no: int,
    prev_text: str,
    next_text: str,
    max_tokens: int = 512,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> str:
    prompt = CONTEXTUAL_CHUNK_PROMPT.format(
        doc_title=doc_title,
        page_title=page_title,
        section_path=section_path,
        page_no=page_no,
        prev_text=prev_text[:300] if prev_text else "(none)",
        next_text=next_text[:300] if next_text else "(none)",
        chunk_text=chunk_text,
    )
    try:
        if llm_semaphore is None:
            result = await vllm_chat(
                runtime,
                settings=runtime.settings,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        else:
            async with llm_semaphore:
                result = await vllm_chat(
                    runtime,
                    settings=runtime.settings,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
        return clean_generated_text(result)
    except Exception:
        logger.exception("Failed to build contextual chunk")
        return ""


async def build_page_proxy(
    runtime: RuntimeClients,
    page_text: str,
    doc_title: str,
    page_no: int,
    page_title: str,
    section_path: str,
    prev_title: str,
    next_title: str,
    has_table: bool,
    has_figure: bool,
    image_path: str | None,
    max_tokens: int = 512,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> str:
    prompt = PAGE_PROXY_PROMPT.format(
        doc_title=doc_title,
        page_no=page_no,
        page_title=page_title,
        section_path=section_path,
        prev_title=prev_title or "(none)",
        next_title=next_title or "(none)",
        has_table=has_table,
        has_figure=has_figure,
        page_text=page_text[:3000],
    )
    images = [image_path] if image_path else None
    try:
        if llm_semaphore is None:
            result = await vllm_chat(
                runtime,
                settings=runtime.settings,
                prompt=prompt,
                images=images,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        else:
            async with llm_semaphore:
                result = await vllm_chat(
                    runtime,
                    settings=runtime.settings,
                    prompt=prompt,
                    images=images,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
        return clean_generated_text(result)
    except Exception:
        logger.exception("Failed to build page proxy")
        return ""


async def ocr_page(
    runtime: RuntimeClients,
    image_path: str,
    max_tokens: int = 2048,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> str:
    try:
        if llm_semaphore is None:
            text = await vllm_chat(
                runtime,
                settings=runtime.settings,
                prompt=OCR_PAGE_PROMPT,
                images=[image_path],
                max_tokens=max_tokens,
                temperature=0.0,
            )
        else:
            async with llm_semaphore:
                text = await vllm_chat(
                    runtime,
                    settings=runtime.settings,
                    prompt=OCR_PAGE_PROMPT,
                    images=[image_path],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
    except Exception:
        logger.exception("OCR failed for %s", image_path)
        text = ""
    cleaned = clean_generated_text(text)
    return cleaned.strip() or Path(image_path).stem.replace("_", " ")


async def build_offline(
    runtime: RuntimeClients,
    settings: Settings,
    tracker: WeaveTracker,
    variant: str,
    max_samples: int | None,
    rebuild: bool,
    corpus_limit: int | None,
) -> dict:
    text_index = settings.text_index_name(variant)
    proxy_index = settings.proxy_index_name(variant)
    text_collection = settings.text_collection_name(variant)
    proxy_collection = settings.proxy_collection_name(variant)
    image_collection = settings.image_collection_name(variant)

    await ensure_storage(runtime, settings, variant=variant, rebuild=rebuild)

    samples = load_samples(settings, max_samples=max_samples)
    corpus_pages = load_corpus_pages(settings, corpus_limit=corpus_limit)
    corpus_lookup = {row["doc_page_id"]: row for row in corpus_pages}

    required_ids = {doc_page_id for sample in samples for doc_page_id in sample.relevant_doc_page_ids}
    required_ids.update({doc_page_id for sample in samples for doc_page_id in sample.retrieval_scope_doc_page_ids})
    missing_required = sorted([doc_page_id for doc_page_id in required_ids if doc_page_id not in corpus_lookup])
    if missing_required:
        raise ValueError(f"Missing required corpus pages for samples: {missing_required[:20]}")

    page_workers = _positive_int(settings.build_page_workers, fallback=1)
    llm_concurrency = _positive_int(settings.build_llm_concurrency, fallback=1)
    embed_concurrency = _positive_int(settings.build_embed_concurrency, fallback=1)
    text_batch_size = _positive_int(settings.text_embed_batch_size, fallback=1)
    image_batch_size = _positive_int(settings.image_embed_batch_size, fallback=1)
    qdrant_batch_size = _positive_int(settings.qdrant_upsert_batch_size, fallback=1)
    es_bulk_ops_batch_size = max(2, _positive_int(settings.es_bulk_ops_batch_size, fallback=2000))
    if es_bulk_ops_batch_size % 2 != 0:
        es_bulk_ops_batch_size += 1

    logger.info(
        "Build settings: pages=%d page_workers=%d llm_concurrency=%d embed_concurrency=%d "
        "text_batch=%d image_batch=%d",
        len(corpus_pages),
        page_workers,
        llm_concurrency,
        embed_concurrency,
        text_batch_size,
        image_batch_size,
    )

    llm_semaphore = asyncio.Semaphore(llm_concurrency)
    page_results: list[tuple[int, PageRecord, list[Chunk]]] = []

    async def _prepare_single_page(order: int, row: dict) -> tuple[int, PageRecord, list[Chunk]]:
        doc_page_id = row["doc_page_id"]
        document_id = row["document_id"]
        page_no = row["page_no"]
        image_path = row["image_path"]

        ocr_text = await ocr_page(
            runtime,
            image_path,
            max_tokens=settings.ocr_max_tokens,
            llm_semaphore=llm_semaphore,
        )
        page_title = first_non_empty_line(ocr_text)
        section_path = ""
        has_table = "|" in ocr_text
        has_figure = True

        raw_chunks = chunk_page(ocr_text, page_title=page_title, section_path=section_path)
        contextual_values = [""] * len(raw_chunks)

        if variant == "enhanced" and raw_chunks:
            contextual_values = await asyncio.gather(
                *[
                    build_contextual_chunk(
                        runtime,
                        chunk_text=raw_chunk["text_raw"],
                        doc_title=document_id,
                        page_title=page_title,
                        section_path=section_path,
                        page_no=page_no,
                        prev_text=raw_chunks[idx - 1]["text_raw"] if idx > 0 else "",
                        next_text=raw_chunks[idx + 1]["text_raw"] if idx + 1 < len(raw_chunks) else "",
                        max_tokens=settings.contextual_chunk_max_tokens,
                        llm_semaphore=llm_semaphore,
                    )
                    for idx, raw_chunk in enumerate(raw_chunks)
                ]
            )

        contextual_chunks: list[Chunk] = []
        for idx, raw_chunk in enumerate(raw_chunks):
            text_raw = clean_generated_text(raw_chunk["text_raw"])
            if is_noise_chunk(text_raw):
                continue
            contextual = clean_generated_text(contextual_values[idx] if idx < len(contextual_values) else "")
            text_for_bm25 = f"{contextual}\n{text_raw}".strip() if contextual else text_raw
            text_for_dense = text_for_bm25

            contextual_chunks.append(
                Chunk(
                    chunk_id=f"{doc_page_id}::chunk-{idx:04d}",
                    doc_page_id=doc_page_id,
                    document_id=document_id,
                    page_no=page_no,
                    chunk_index=idx,
                    chunk_type=raw_chunk["chunk_type"],
                    text_raw=text_raw,
                    contextual_chunk_text=contextual,
                    text_for_bm25=text_for_bm25,
                    text_for_dense=text_for_dense,
                    page_title=page_title,
                    section_path=section_path,
                    image_path=image_path,
                )
            )

        page_proxy_text = ""
        if variant == "enhanced":
            page_proxy_text = await build_page_proxy(
                runtime,
                page_text=ocr_text,
                doc_title=document_id,
                page_no=page_no,
                page_title=page_title,
                section_path=section_path,
                prev_title="",
                next_title="",
                has_table=has_table,
                has_figure=has_figure,
                image_path=image_path if settings.use_page_image else None,
                max_tokens=settings.page_proxy_max_tokens,
                llm_semaphore=llm_semaphore,
            )

        page_record = PageRecord(
            doc_page_id=doc_page_id,
            document_id=document_id,
            page_no=page_no,
            image_path=image_path,
            ocr_text=ocr_text,
            page_title=page_title,
            section_path=section_path,
            page_proxy_text=page_proxy_text,
            has_table=has_table,
            has_figure=has_figure,
        )
        return order, page_record, contextual_chunks

    total_pages = len(corpus_pages)
    if total_pages > 0:
        page_prepare_semaphore = asyncio.Semaphore(min(page_workers, total_pages))

        async def _prepare_single_page_bounded(order: int, row: dict) -> tuple[int, PageRecord, list[Chunk]]:
            async with page_prepare_semaphore:
                return await _prepare_single_page(order, row)

        page_tasks = [
            asyncio.create_task(_prepare_single_page_bounded(order, row))
            for order, row in enumerate(corpus_pages)
        ]

        try:
            for idx, future in enumerate(asyncio.as_completed(page_tasks), start=1):
                page_results.append(await future)
                if idx % 50 == 0 or idx == total_pages:
                    logger.info("Indexed page preparation progress: %d/%d", idx, total_pages)
        except Exception:
            for task in page_tasks:
                task.cancel()
            await asyncio.gather(*page_tasks, return_exceptions=True)
            raise

    page_results.sort(key=lambda item: item[0])
    pages = [record for _, record, _ in page_results]
    chunks = [chunk for _, _, page_chunks in page_results for chunk in page_chunks]

    text_docs: list[dict] = []
    for chunk in chunks:
        text_docs.append(
            {
                "_id": chunk.chunk_id,
                "chunk_id": chunk.chunk_id,
                "doc_page_id": chunk.doc_page_id,
                "document_id": chunk.document_id,
                "page_no": chunk.page_no,
                "chunk_type": chunk.chunk_type,
                "page_title": chunk.page_title,
                "section_path": chunk.section_path,
                "image_path": chunk.image_path,
                "text_raw": chunk.text_raw,
                "contextual_chunk_text": chunk.contextual_chunk_text,
                "text_for_bm25": chunk.text_for_bm25,
            }
        )

    operations: list[dict] = []
    for doc in text_docs:
        doc_id = doc.pop("_id")
        operations.append({"index": {"_index": text_index, "_id": doc_id}})
        operations.append(doc)
    if operations:
        for start in range(0, len(operations), es_bulk_ops_batch_size):
            await runtime.es.bulk(operations=operations[start : start + es_bulk_ops_batch_size], refresh=False)
        await runtime.es.indices.refresh(index=text_index)

    if variant == "enhanced":
        proxy_docs: list[dict] = []
        for page in pages:
            if not page.page_proxy_text:
                continue
            proxy_docs.append(
                {
                    "_id": page.doc_page_id,
                    "doc_page_id": page.doc_page_id,
                    "document_id": page.document_id,
                    "page_no": page.page_no,
                    "page_title": page.page_title,
                    "section_path": page.section_path,
                    "image_path": page.image_path,
                    "page_proxy_text": page.page_proxy_text,
                }
            )
        proxy_ops: list[dict] = []
        for doc in proxy_docs:
            doc_id = doc.pop("_id")
            proxy_ops.append({"index": {"_index": proxy_index, "_id": doc_id}})
            proxy_ops.append(doc)
        if proxy_ops:
            for start in range(0, len(proxy_ops), es_bulk_ops_batch_size):
                await runtime.es.bulk(operations=proxy_ops[start : start + es_bulk_ops_batch_size], refresh=False)
            await runtime.es.indices.refresh(index=proxy_index)

        proxy_ids = [doc["doc_page_id"] for doc in proxy_docs]
        proxy_payloads = [
            {
                "doc_page_id": doc["doc_page_id"],
                "document_id": doc["document_id"],
                "page_no": doc["page_no"],
                "page_title": doc["page_title"],
                "section_path": doc["section_path"],
                "image_path": doc["image_path"],
                "page_proxy_text": doc["page_proxy_text"],
            }
            for doc in proxy_docs
        ]
        proxy_vectors: list[list[float] | None] = [None] * len(proxy_payloads)
        proxy_embed_semaphore = asyncio.Semaphore(embed_concurrency)

        async def _embed_proxy_batch(start: int) -> None:
            batch = [payload["page_proxy_text"] for payload in proxy_payloads[start : start + text_batch_size]]
            async with proxy_embed_semaphore:
                vectors = await embed_texts(runtime, settings, batch, input_type="document")
            for offset, vector in enumerate(vectors):
                proxy_vectors[start + offset] = vector

        proxy_embed_tasks = [asyncio.create_task(_embed_proxy_batch(i)) for i in range(0, len(proxy_payloads), text_batch_size)]
        if proxy_embed_tasks:
            await asyncio.gather(*proxy_embed_tasks)

        if proxy_ids:
            if any(vector is None for vector in proxy_vectors):
                raise RuntimeError("Page proxy embedding generation incomplete.")
            for start in range(0, len(proxy_ids), qdrant_batch_size):
                proxy_points = [
                    qdrant_models.PointStruct(
                        id=_stable_point_id(proxy_ids[i]),
                        vector=proxy_vectors[i],  # type: ignore[arg-type]
                        payload=proxy_payloads[i],
                    )
                    for i in range(start, min(start + qdrant_batch_size, len(proxy_ids)))
                ]
                await runtime.qdrant.upsert(collection_name=proxy_collection, points=proxy_points)

    text_ids = [chunk.chunk_id for chunk in chunks]
    text_payloads = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_page_id": chunk.doc_page_id,
            "document_id": chunk.document_id,
            "page_no": chunk.page_no,
            "chunk_type": chunk.chunk_type,
            "page_title": chunk.page_title,
            "section_path": chunk.section_path,
            "image_path": chunk.image_path,
            "text_raw": chunk.text_raw,
            "contextual_chunk_text": chunk.contextual_chunk_text,
        }
        for chunk in chunks
    ]

    text_vectors: list[list[float] | None] = [None] * len(chunks)
    text_embed_semaphore = asyncio.Semaphore(embed_concurrency)

    async def _embed_text_batch(start: int) -> None:
        batch = [chunk.text_for_dense for chunk in chunks[start : start + text_batch_size]]
        async with text_embed_semaphore:
            vectors = await embed_texts(runtime, settings, batch, input_type="document")
        for offset, vector in enumerate(vectors):
            text_vectors[start + offset] = vector

    text_embed_tasks = [asyncio.create_task(_embed_text_batch(i)) for i in range(0, len(chunks), text_batch_size)]
    if text_embed_tasks:
        await asyncio.gather(*text_embed_tasks)

    if text_ids:
        if any(vector is None for vector in text_vectors):
            raise RuntimeError("Text embedding generation incomplete.")
        for start in range(0, len(text_ids), qdrant_batch_size):
            points = [
                qdrant_models.PointStruct(
                    id=_stable_point_id(text_ids[i]),
                    vector=text_vectors[i],  # type: ignore[arg-type]
                    payload=text_payloads[i],
                )
                for i in range(start, min(start + qdrant_batch_size, len(text_ids)))
            ]
            await runtime.qdrant.upsert(collection_name=text_collection, points=points)

    if settings.use_page_image:
        image_ids = [page.doc_page_id for page in pages]
        image_payloads = [
            {
                "doc_page_id": page.doc_page_id,
                "document_id": page.document_id,
                "page_no": page.page_no,
                "page_title": page.page_title,
                "section_path": page.section_path,
                "image_path": page.image_path,
                "page_proxy_text": page.page_proxy_text,
            }
            for page in pages
        ]
        image_vectors: list[list[float] | None] = [None] * len(pages)
        image_embed_semaphore = asyncio.Semaphore(embed_concurrency)

        async def _embed_image_batch(start: int) -> None:
            batch_paths = [page.image_path for page in pages[start : start + image_batch_size]]
            async with image_embed_semaphore:
                vectors = await embed_images(runtime, settings, batch_paths, input_type="document")
            for offset, vector in enumerate(vectors):
                image_vectors[start + offset] = vector

        image_embed_tasks = [asyncio.create_task(_embed_image_batch(i)) for i in range(0, len(pages), image_batch_size)]
        if image_embed_tasks:
            await asyncio.gather(*image_embed_tasks)

        if image_ids:
            if any(vector is None for vector in image_vectors):
                raise RuntimeError("Image embedding generation incomplete.")
            for start in range(0, len(image_ids), qdrant_batch_size):
                image_points = [
                    qdrant_models.PointStruct(
                        id=_stable_point_id(image_ids[i]),
                        vector=image_vectors[i],  # type: ignore[arg-type]
                        payload=image_payloads[i],
                    )
                    for i in range(start, min(start + qdrant_batch_size, len(image_ids)))
                ]
                await runtime.qdrant.upsert(collection_name=image_collection, points=image_points)

    artifact_dir = settings.artifact_dir(variant)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(artifact_dir / "pages.jsonl", [page.__dict__ for page in pages])
    write_jsonl(artifact_dir / "chunks.jsonl", [chunk.__dict__ for chunk in chunks])

    summary = {
        "status": "built",
        "variant": variant,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "num_samples": len(samples),
        "text_index": text_index,
        "proxy_index": proxy_index,
        "text_collection": text_collection,
        "proxy_collection": proxy_collection,
        "image_collection": image_collection,
        "artifact_dir": str(artifact_dir),
        "built_at": int(time.time()),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.publish(summary, name=f"build-{variant}")
    return summary


async def retrieve(
    runtime: RuntimeClients,
    settings: Settings,
    variant: str,
    query: str,
    text_query_vector: list[float],
    image_query_vector: list[float] | None,
    filter_doc_page_ids: list[str] | None = None,
    filter_document_ids: list[str] | None = None,
) -> dict:
    top_k = settings.top_k
    scoped_doc_page_ids = _dedupe_non_empty(filter_doc_page_ids)
    scoped_document_ids = _dedupe_non_empty(filter_document_ids)
    scoped_doc_page_id_set = set(scoped_doc_page_ids)
    scoped_document_id_set = set(scoped_document_ids)

    bm25_hits = await bm25_search(
        runtime,
        settings.text_index_name(variant),
        query,
        field="text_for_bm25",
        top_k=top_k,
        filter_doc_page_ids=scoped_doc_page_ids,
        filter_document_ids=scoped_document_ids,
    )
    dense_hits = await qdrant_search(
        runtime,
        settings.text_collection_name(variant),
        text_query_vector,
        top_k=top_k,
        filter_doc_page_ids=scoped_doc_page_ids,
        filter_document_ids=scoped_document_ids,
    )

    proxy_hits: list[dict] = []
    proxy_dense_hits: list[dict] = []
    if variant == "enhanced":
        proxy_hits = await bm25_search(
            runtime,
            settings.proxy_index_name(variant),
            query,
            field="page_proxy_text",
            top_k=top_k,
            filter_doc_page_ids=scoped_doc_page_ids,
            filter_document_ids=scoped_document_ids,
        )
        proxy_dense_hits = await qdrant_search(
            runtime,
            settings.proxy_collection_name(variant),
            text_query_vector,
            top_k=top_k,
            filter_doc_page_ids=scoped_doc_page_ids,
            filter_document_ids=scoped_document_ids,
        )

    image_hits: list[dict] = []
    if settings.use_page_image and image_query_vector is not None:
        image_hits = await qdrant_search(
            runtime,
            settings.image_collection_name(variant),
            image_query_vector,
            top_k=top_k,
            filter_doc_page_ids=scoped_doc_page_ids,
            filter_document_ids=scoped_document_ids,
        )

    hit_lists = [
        [{"source": "bm25", **hit} for hit in bm25_hits],
        [{"source": "dense", **hit} for hit in dense_hits],
        [{"source": "page_image", **hit} for hit in image_hits],
        [{"source": "page_proxy", **hit} for hit in proxy_hits],
        [{"source": "page_proxy_dense", **hit} for hit in proxy_dense_hits],
    ]

    candidate_map: dict[str, dict] = {}
    k = 60

    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            doc_page_id = str(hit.get("doc_page_id") or "").strip()
            if not doc_page_id:
                continue
            if scoped_doc_page_id_set and doc_page_id not in scoped_doc_page_id_set:
                continue
            hit_document_id = str(hit.get("document_id") or "").strip()
            if scoped_document_id_set and hit_document_id not in scoped_document_id_set:
                continue
            candidate = candidate_map.setdefault(
                doc_page_id,
                {
                    "doc_page_id": doc_page_id,
                    "document_id": hit_document_id,
                    "page_no": int(hit.get("page_no", 0) or 0),
                    "page_title": hit.get("page_title", ""),
                    "section_path": hit.get("section_path", ""),
                    "image_path": hit.get("image_path", ""),
                    "page_proxy_text": hit.get("page_proxy_text", ""),
                    "text_raw": "",
                    "contextual_chunk_text": "",
                    "chunk_ids": [],
                    "supporting_chunks": [],
                    "sources": [],
                    "source_scores": {},
                    "score": 0.0,
                },
            )
            source = hit.get("source", "")
            source_weight = _source_rrf_weight(settings, str(source))
            if source_weight <= 0.0:
                continue
            candidate["score"] += source_weight / (k + rank + 1)
            if source and source not in candidate["sources"]:
                candidate["sources"].append(source)
            raw_score = float(hit.get("_score", hit.get("score", 0.0)) or 0.0)
            candidate["source_scores"][source] = max(candidate["source_scores"].get(source, float("-inf")), raw_score)

            if hit.get("chunk_id") and hit["chunk_id"] not in candidate["chunk_ids"]:
                candidate["chunk_ids"].append(hit["chunk_id"])
                candidate["supporting_chunks"].append(
                    {
                        "chunk_id": hit["chunk_id"],
                        "chunk_type": hit.get("chunk_type", ""),
                        "text_raw": hit.get("text_raw", ""),
                        "contextual_chunk_text": hit.get("contextual_chunk_text", ""),
                    }
                )

            if hit.get("page_proxy_text") and not candidate["page_proxy_text"]:
                candidate["page_proxy_text"] = hit["page_proxy_text"]

    fused = sorted(candidate_map.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for candidate in fused:
        raw_snippets = [chunk["text_raw"].strip() for chunk in candidate["supporting_chunks"] if chunk.get("text_raw", "").strip()]
        ctx_snippets = [chunk["contextual_chunk_text"].strip() for chunk in candidate["supporting_chunks"] if chunk.get("contextual_chunk_text", "").strip()]
        candidate["text_raw"] = "\n\n".join(raw_snippets[:2])
        candidate["contextual_chunk_text"] = "\n\n".join(ctx_snippets[:2])

    return {
        "bm25_hits": bm25_hits,
        "dense_hits": dense_hits,
        "image_hits": image_hits,
        "proxy_hits": proxy_hits,
        "proxy_dense_hits": proxy_dense_hits,
        "fused_hits": fused,
    }


async def rerank_candidates(runtime: RuntimeClients, settings: Settings, query: str, fused_hits: list[dict]) -> list[dict]:
    if not fused_hits:
        return []
    top_k = settings.rerank_top_k
    if not settings.use_reranker:
        return fused_hits[:top_k]
    # Keep each rerank candidate compact to avoid endpoint max-length 400 errors.
    documents = [
        _truncate_to_token_budget(candidate_to_rerank_document(hit), token_budget=RERANK_DOCUMENT_TOKEN_BUDGET)
        for hit in fused_hits
    ]
    try:
        results = await rerank_documents(runtime, settings, query=query, documents=documents, top_k=top_k)
        reranked: list[dict] = []
        for row in results:
            index = int(row["index"])
            if index < 0 or index >= len(fused_hits):
                continue
            hit = fused_hits[index]
            hit = dict(hit)
            hit["score"] = row["relevance_score"]
            reranked.append(hit)
        if not reranked:
            return fused_hits[:top_k]
        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return reranked[:top_k]
    except Exception:
        logger.exception("Rerank failed, fallback to fused order")
        return fused_hits[:top_k]


async def generate_answer(runtime: RuntimeClients, settings: Settings, query: str, hits: list[dict]) -> dict:
    hits_for_answer = hits[: max(1, min(ANSWER_MAX_HITS, len(hits)))]
    chunk_evidence_blocks: list[str] = []
    page_images: list[str] = []
    citations: list[dict] = []

    for hit in hits_for_answer:
        citations.append(
            {
                "document_id": hit.get("document_id", ""),
                "doc_page_id": hit.get("doc_page_id", ""),
                "page_no": hit.get("page_no", 0),
                "chunk_ids": hit.get("chunk_ids", []),
                "snippet": (hit.get("text_raw") or hit.get("page_title") or "")[:240],
            }
        )

        for chunk in hit.get("supporting_chunks", [])[:ANSWER_MAX_CHUNKS_PER_HIT]:
            raw_text = _truncate_to_token_budget(
                str(chunk.get("text_raw") or "(none)"),
                token_budget=ANSWER_RAW_TOKEN_BUDGET,
            )
            chunk_evidence_blocks.append(
                raw_text or "(no OCR evidence found)"
            )

    if settings.use_page_image:
        seen_image_paths: set[str] = set()
        for hit in hits_for_answer:
            image_path = hit.get("image_path", "")
            if image_path and image_path not in seen_image_paths and Path(image_path).exists():
                page_images.append(image_path)
                seen_image_paths.add(image_path)
                if len(page_images) >= ANSWER_MAX_IMAGES:
                    break

    question_tokens = approx_tokens(query)
    prompt_token_budget = max(800, ANSWER_PROMPT_TOKEN_BUDGET - min(question_tokens, 512))
    chunk_evidence = _truncate_to_token_budget(
        "\n\n".join(chunk_evidence_blocks),
        token_budget=max(320, prompt_token_budget),
    ) or "(no OCR evidence found)"
    prompt = ANSWER_PROMPT.format(chunk_evidence=chunk_evidence, question=query)
    compact_chunk_evidence = _truncate_to_token_budget(
        chunk_evidence,
        token_budget=max(160, prompt_token_budget // 2),
    )
    compact_prompt = ANSWER_PROMPT.format(
        chunk_evidence=compact_chunk_evidence or "(no OCR evidence found)",
        question=query,
    )

    attempts = [
        {"prompt": prompt, "images": page_images or None, "max_tokens": 256},
        {"prompt": compact_prompt, "images": page_images or None, "max_tokens": 128},
    ]
    answer = "Error generating answer."
    page_images_used: list[str] = []
    final_context_text = chunk_evidence

    for attempt_idx, attempt in enumerate(attempts, start=1):
        try:
            answer = await vllm_chat(
                runtime,
                settings=settings,
                prompt=attempt["prompt"],
                images=attempt["images"],
                max_tokens=attempt["max_tokens"],
                temperature=0.1,
            )
            page_images_used = attempt["images"] or []
            if attempt_idx > 1:
                final_context_text = compact_chunk_evidence
            break
        except Exception:
            logger.exception("Answer generation failed on attempt %d/%d", attempt_idx, len(attempts))
            if attempt_idx < len(attempts):
                await asyncio.sleep(0.5)

    return {
        "answer": answer,
        "citations": citations,
        "page_images_used": page_images_used,
        "context_text": final_context_text,
    }


def _prediction_needs_rewrite(prediction: str, raw_answer: str) -> bool:
    normalized_prediction = strip_answer_citations(prediction)
    normalized_raw = strip_answer_citations(raw_answer)
    if not normalized_prediction:
        return True
    if "\n" in normalized_prediction:
        return True
    if len(normalized_prediction.split()) > 8:
        return True
    if looks_like_reasoning_leak(normalized_prediction):
        return True
    return normalized_prediction == normalized_raw and len(normalized_prediction.split()) > 4


async def finalize_prediction(
    runtime: RuntimeClients,
    settings: Settings,
    question: str,
    raw_answer: str,
) -> str:
    initial = extract_short_answer(raw_answer)
    if not _prediction_needs_rewrite(initial, raw_answer):
        return initial

    prompt = ANSWER_REWRITE_PROMPT.format(
        question=question,
        draft_answer=_truncate_to_token_budget(strip_answer_citations(raw_answer), token_budget=180) or "Unknown",
    )
    try:
        rewritten = await vllm_chat(
            runtime,
            settings=settings,
            prompt=prompt,
            images=None,
            max_tokens=32,
            temperature=0.0,
            max_retries=2,
        )
        final = extract_short_answer(rewritten)
        if final and not looks_like_reasoning_leak(final):
            return final
    except Exception:
        logger.exception("Answer rewrite failed")

    return initial


def _to_finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _build_ragas_contexts(hits: list[dict]) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()
    for hit in hits[:RAGAS_MAX_CONTEXTS]:
        context = _compact_ragas_context(hit, token_budget=RAGAS_CONTEXT_TOKEN_BUDGET).strip()
        if context and context not in seen:
            seen.add(context)
            contexts.append(context)
    return contexts


async def _run_ragas_eval(settings: Settings, run_id: str, ragas_samples: list[dict], sample_ids: list[str]) -> dict:
    if not ragas_samples:
        return {
            "enabled": False,
            "aggregate": {},
            "per_sample": [],
            "results_path": "",
            "error": "No samples for RAGAS evaluation.",
        }

    try:
        import warnings

        import httpx
        from openai import AsyncOpenAI
        from ragas import EvaluationDataset, aevaluate
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig
    except Exception as exc:
        return {
            "enabled": False,
            "aggregate": {},
            "per_sample": [],
            "results_path": "",
            "error": f"RAGAS dependencies unavailable: {exc}",
        }

    ragas_llm_base_url = settings.ragas_llm_base_url.strip() or settings.vllm_base_url
    ragas_llm_model_name = settings.ragas_llm_model_name.strip() or settings.vllm_model_name

    llm_client = AsyncOpenAI(
        base_url=ragas_llm_base_url,
        api_key="EMPTY",
        http_client=httpx.AsyncClient(trust_env=should_trust_env(ragas_llm_base_url), timeout=180.0),
    )
    embedding_client = AsyncOpenAI(
        base_url=settings.qwen_embed_base_url,
        api_key=settings.qwen_embed_api_key or "EMPTY",
        http_client=httpx.AsyncClient(trust_env=should_trust_env(settings.qwen_embed_base_url), timeout=120.0),
    )

    try:
        ragas_llm = llm_factory(
            model=ragas_llm_model_name,
            provider="openai",
            client=llm_client,
            temperature=0.0,
            max_tokens=256,
            max_retries=1,
            top_p=0.1,
        )
        ragas_embeddings = embedding_factory(
            provider="openai",
            model=settings.qwen_embed_model,
            client=embedding_client,
        )
        if "vl" in ragas_llm_model_name.lower():
            message = (
                f"Skipping RAGAS because multimodal judge model '{ragas_llm_model_name}' is unstable for "
                "structured RAGAS prompts. Set RAGAS_LLM_MODEL_NAME to a text-only model to enable RAGAS."
            )
            logger.warning(message)
            return {
                "enabled": False,
                "aggregate": {},
                "per_sample": [],
                "results_path": "",
                "error": message,
            }
        ragas_metrics = [
            copy.deepcopy(faithfulness),
            copy.deepcopy(answer_relevancy),
            copy.deepcopy(context_precision),
            copy.deepcopy(context_recall),
            copy.deepcopy(answer_correctness),
        ]
        for metric in ragas_metrics:
            if hasattr(metric, "llm"):
                metric.llm = ragas_llm
            if hasattr(metric, "embeddings"):
                if getattr(metric, "name", "") == "answer_relevancy":
                    metric.embeddings = _RagasRelevancyEmbeddingAdapter(ragas_embeddings)
                    # Lower question generation fan-out to reduce eval latency/timeouts.
                    if hasattr(metric, "strictness"):
                        metric.strictness = 1
                else:
                    metric.embeddings = ragas_embeddings
        dataset = EvaluationDataset.from_list(ragas_samples, name=f"{run_id}-ragas")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
            run_config = RunConfig(timeout=180, max_retries=1, max_workers=4)
            eval_result = await aevaluate(
                dataset=dataset,
                metrics=ragas_metrics,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=True,
            )

        per_sample: list[dict] = []
        for raw_scores in eval_result.scores:
            row_scores: dict[str, float | None] = {}
            for metric_name in RAGAS_METRIC_NAMES:
                row_scores[metric_name] = _to_finite_float(raw_scores.get(metric_name))
            per_sample.append(row_scores)

        aggregate: dict[str, float | None] = {}
        for metric_name in RAGAS_METRIC_NAMES:
            values = [row[metric_name] for row in per_sample if row.get(metric_name) is not None]
            aggregate[f"ragas_{metric_name}"] = sum(values) / len(values) if values else None

        ragas_rows: list[dict] = []
        for idx, sample_id in enumerate(sample_ids):
            ragas_rows.append(
                {
                    "sample_id": sample_id,
                    "scores": per_sample[idx] if idx < len(per_sample) else {},
                }
            )

        ragas_path = settings.eval_runs_dir / f"{run_id}.ragas.jsonl"
        write_jsonl(ragas_path, ragas_rows)
        return {
            "enabled": True,
            "aggregate": aggregate,
            "per_sample": per_sample,
            "results_path": str(ragas_path),
            "error": "",
        }
    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        return {
            "enabled": False,
            "aggregate": {},
            "per_sample": [],
            "results_path": "",
            "error": str(exc),
        }
    finally:
        for name, client in [("ragas_llm_client", llm_client), ("ragas_embedding_client", embedding_client)]:
            try:
                await client.close()
            except RuntimeError as exc:
                # Guard against anyio/httpx transport shutdown race in some environments.
                logger.debug("Ignoring %s close RuntimeError: %s", name, exc)
            except Exception:
                logger.exception("Failed to close %s", name)


async def run_eval(
    runtime: RuntimeClients,
    settings: Settings,
    tracker: WeaveTracker,
    variant: str,
    max_samples: int | None,
    run_name: str | None,
    with_ragas: bool = True,
) -> dict:
    samples = load_samples(settings, max_samples=max_samples)
    if not samples:
        raise ValueError("No samples loaded")

    unique_questions = list(dict.fromkeys(sample.question for sample in samples))
    text_query_vectors = await embed_texts(runtime, settings, unique_questions, input_type="query")
    text_query_by_question = dict(zip(unique_questions, text_query_vectors))

    image_query_by_question: dict[str, list[float]] = {}
    if settings.use_page_image:
        image_query_vectors = await embed_multimodal_texts(runtime, settings, unique_questions, input_type="query")
        image_query_by_question = dict(zip(unique_questions, image_query_vectors))

    latencies: list[float] = []
    rows: list[dict] = []
    ragas_samples: list[dict] = []
    ragas_sample_ids: list[str] = []

    run_id = run_name or f"{variant}-{int(time.time())}"
    results_path = settings.eval_runs_dir / f"{run_id}.jsonl"

    for idx, sample in enumerate(samples, start=1):
        if idx % 20 == 0:
            logger.info("Eval progress %d/%d", idx, len(samples))
        t0 = time.monotonic()

        retrieval = await retrieve(
            runtime,
            settings,
            variant=variant,
            query=sample.question,
            text_query_vector=text_query_by_question[sample.question],
            image_query_vector=image_query_by_question.get(sample.question),
            filter_doc_page_ids=sample.retrieval_scope_doc_page_ids,
            filter_document_ids=sample.retrieval_scope_document_ids,
        )
        reranked = await rerank_candidates(runtime, settings, query=sample.question, fused_hits=retrieval["fused_hits"])
        answer_result = await generate_answer(runtime, settings, query=sample.question, hits=reranked)

        latency_s = time.monotonic() - t0
        latencies.append(latency_s)

        prediction = await finalize_prediction(
            runtime,
            settings,
            question=sample.question,
            raw_answer=answer_result["answer"],
        )
        retrieved_doc_page_ids = [hit.get("doc_page_id", "") for hit in reranked]

        row = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "answers": sample.answers,
            "relevant_doc_page_ids": sample.relevant_doc_page_ids,
            "retrieved_doc_page_ids": retrieved_doc_page_ids,
            "prediction": prediction,
            "scores": {},
            "latency_s": latency_s,
            "citations": answer_result["citations"],
        }
        rows.append(row)

        if with_ragas:
            ragas_samples.append(
                {
                    "user_input": sample.question,
                    "response": _truncate_to_token_budget(str(prediction or "").strip(), token_budget=RAGAS_RESPONSE_TOKEN_BUDGET),
                    "reference": _truncate_to_token_budget(
                        str(sample.answers[0] if sample.answers else "").strip(),
                        token_budget=RAGAS_REFERENCE_TOKEN_BUDGET,
                    ),
                    "retrieved_contexts": _build_ragas_contexts(reranked),
                }
            )
            ragas_sample_ids.append(sample.sample_id)

    logger.info("Computing EM/F1 using evaluate[squad] and Recall@5 using pytrec_eval")
    qa_per_row, qa_aggregate = _compute_qa_scores(rows)
    recall_per_row, recall_aggregate = _compute_recall_at_5_scores(rows)
    if len(qa_per_row) != len(rows) or len(recall_per_row) != len(rows):
        raise RuntimeError("Metric rows are misaligned with evaluation samples.")

    for idx, row in enumerate(rows):
        row["scores"]["exact_match"] = qa_per_row[idx]["exact_match"]
        row["scores"]["f1"] = qa_per_row[idx]["f1"]
        row["scores"]["recall_at_5"] = recall_per_row[idx]

    ragas_output = {
        "enabled": False,
        "aggregate": {},
        "per_sample": [],
        "results_path": "",
        "error": "",
    }
    if with_ragas:
        logger.info("Running RAGAS evaluation on %d samples", len(ragas_samples))
        ragas_output = await _run_ragas_eval(settings, run_id=run_id, ragas_samples=ragas_samples, sample_ids=ragas_sample_ids)
        per_sample_ragas = ragas_output.get("per_sample", [])
        for idx, row in enumerate(rows):
            ragas_scores = per_sample_ragas[idx] if idx < len(per_sample_ragas) else {}
            if not ragas_scores:
                continue
            row["ragas"] = ragas_scores
            for metric_name, metric_value in ragas_scores.items():
                if metric_value is not None:
                    row["scores"][f"ragas_{metric_name}"] = metric_value

    with open(results_path, "w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "status": "completed",
        "variant": variant,
        "dataset": settings.dataset_name,
        "num_samples": len(samples),
        "exact_match": qa_aggregate["exact_match"],
        "f1": qa_aggregate["f1"],
        "recall_at_5": recall_aggregate,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_s": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "results_path": str(results_path),
        "generated_at": int(time.time()),
    }

    metrics["ragas_enabled"] = with_ragas
    if with_ragas:
        metrics.update(ragas_output.get("aggregate", {}))
        metrics["ragas_results_path"] = ragas_output.get("results_path", "")
        if ragas_output.get("error"):
            metrics["ragas_error"] = ragas_output["error"]

    summary_path = settings.eval_runs_dir / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.publish(metrics, name=f"eval-{run_id}")
    return metrics
