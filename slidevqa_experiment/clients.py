from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models as qdrant_models

from .config import Settings
from .utils import normalize_base_url, sanitize_llm_output, should_trust_env

logger = logging.getLogger("slidevqa_experiment")


KB_TEXT_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_page_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "page_no": {"type": "integer"},
            "chunk_type": {"type": "keyword"},
            "page_title": {"type": "text"},
            "section_path": {"type": "text"},
            "image_path": {"type": "keyword"},
            "text_raw": {"type": "text", "analyzer": "standard"},
            "contextual_chunk_text": {"type": "text", "analyzer": "standard"},
            "text_for_bm25": {"type": "text", "analyzer": "standard"},
        }
    }
}

KB_PAGE_PROXY_MAPPING = {
    "mappings": {
        "properties": {
            "doc_page_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "page_no": {"type": "integer"},
            "page_title": {"type": "text"},
            "section_path": {"type": "text"},
            "image_path": {"type": "keyword"},
            "page_proxy_text": {"type": "text", "analyzer": "standard"},
        }
    }
}


class RuntimeClients:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.es = AsyncElasticsearch(settings.es_url)
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url, trust_env=should_trust_env(settings.qdrant_url))
        self.embed_client = AsyncOpenAI(
            base_url=settings.qwen_embed_base_url,
            api_key=settings.qwen_embed_api_key or "EMPTY",
            http_client=httpx.AsyncClient(trust_env=should_trust_env(settings.qwen_embed_base_url), timeout=120.0),
        )
        self.vllm_client = AsyncOpenAI(
            base_url=settings.vllm_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(trust_env=should_trust_env(settings.vllm_base_url), timeout=600.0),
        )
        self.multimodal_client = httpx.AsyncClient(
            trust_env=should_trust_env(settings.qwen_multimodal_embed_base_url),
            timeout=180.0,
        )

    async def close(self) -> None:
        await self.es.close()
        await self.qdrant.close()
        await self.embed_client.close()
        await self.vllm_client.close()
        await self.multimodal_client.aclose()


def _prepend_instruction(texts: list[str], instruction: str) -> list[str]:
    prefix = instruction.strip()
    if not prefix:
        return texts
    return [f"{prefix}\n{text}" if text else prefix for text in texts]


def _multimodal_urls(base_url: str) -> tuple[list[str], str]:
    base = normalize_base_url(base_url)
    legacy = [f"{base}/multimodalembeddings"]
    if base.endswith("/v1"):
        root = base[: -len("/v1")]
        if root:
            legacy.append(f"{root}/multimodalembeddings")
        openai = f"{base}/embeddings"
    else:
        openai = f"{base}/v1/embeddings"
    return legacy, openai


def _extract_embedding_rows(payload: dict) -> list[list[float]]:
    if isinstance(payload.get("embeddings"), list):
        return payload["embeddings"]
    data = payload.get("data")
    if isinstance(data, list):
        rows = [item.get("embedding") for item in data if isinstance(item, dict) and isinstance(item.get("embedding"), list)]
        if rows:
            return rows
    raise ValueError(f"Unsupported embedding response payload: {payload}")


async def embed_texts(runtime: RuntimeClients, settings: Settings, texts: list[str], input_type: str) -> list[list[float]]:
    instruction = settings.qwen_query_instruction if input_type == "query" else settings.qwen_document_instruction
    prepared = _prepend_instruction(texts, instruction)
    resp = await runtime.embed_client.embeddings.create(model=settings.qwen_embed_model, input=prepared)
    return [row.embedding for row in resp.data]


async def embed_multimodal_texts(
    runtime: RuntimeClients,
    settings: Settings,
    texts: list[str],
    input_type: str,
) -> list[list[float]]:
    instruction = (
        settings.qwen_multimodal_query_instruction if input_type == "query" else settings.qwen_multimodal_document_instruction
    )
    prepared = _prepend_instruction(texts, instruction)
    payload = {
        "model": settings.qwen_multimodal_embed_model,
        "input_type": input_type,
        "inputs": [[text] for text in prepared],
    }
    headers = {}
    if settings.qwen_multimodal_embed_api_key:
        headers["Authorization"] = f"Bearer {settings.qwen_multimodal_embed_api_key}"
    legacy_urls, openai_url = _multimodal_urls(settings.qwen_multimodal_embed_base_url)

    errors: list[str] = []
    for url in legacy_urls:
        try:
            response = await runtime.multimodal_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return _extract_embedding_rows(response.json())
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    # vLLM-compatible fallback
    vllm_payload = {
        "model": settings.qwen_multimodal_embed_model,
        "input": prepared,
    }
    try:
        response = await runtime.multimodal_client.post(openai_url, json=vllm_payload, headers=headers)
        response.raise_for_status()
        return _extract_embedding_rows(response.json())
    except Exception as exc:
        errors.append(f"{openai_url}: {exc}")
        raise RuntimeError("Qwen multimodal text embedding request failed: " + " | ".join(errors)) from exc


def _image_to_data_url(image_path: str | Path) -> str:
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


async def embed_images(
    runtime: RuntimeClients,
    settings: Settings,
    image_paths: list[str],
    input_type: str = "document",
) -> list[list[float]]:
    image_data_urls = [_image_to_data_url(path) for path in image_paths]
    payload = {
        "model": settings.qwen_multimodal_embed_model,
        "input_type": input_type,
        "inputs": [[data_url] for data_url in image_data_urls],
    }
    headers = {}
    if settings.qwen_multimodal_embed_api_key:
        headers["Authorization"] = f"Bearer {settings.qwen_multimodal_embed_api_key}"
    legacy_urls, openai_url = _multimodal_urls(settings.qwen_multimodal_embed_base_url)

    errors: list[str] = []
    for url in legacy_urls:
        try:
            response = await runtime.multimodal_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return _extract_embedding_rows(response.json())
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    # vLLM-compatible fallback for image embeddings.
    embeddings: list[list[float]] = []
    try:
        for image_data_url in image_data_urls:
            vllm_payload = {
                "model": settings.qwen_multimodal_embed_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": image_data_url}}],
                    }
                ],
            }
            response = await runtime.multimodal_client.post(openai_url, json=vllm_payload, headers=headers)
            response.raise_for_status()
            rows = _extract_embedding_rows(response.json())
            if not rows:
                raise ValueError("Empty embedding rows from vLLM multimodal endpoint.")
            embeddings.append(rows[0])
        return embeddings
    except Exception as exc:
        errors.append(f"{openai_url}: {exc}")
        raise RuntimeError("Qwen multimodal image embedding request failed: " + " | ".join(errors)) from exc


async def vllm_chat(
    runtime: RuntimeClients,
    settings: Settings,
    prompt: str,
    images: list[str] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    content: list[dict] = []
    if images:
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
    content.append({"type": "text", "text": prompt})

    import asyncio as _asyncio
    import random

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            request_kwargs = {
                "model": settings.vllm_model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if settings.vllm_seed is not None:
                request_kwargs["seed"] = settings.vllm_seed
            resp = await runtime.vllm_client.chat.completions.create(
                **request_kwargs,
            )
            return sanitize_llm_output(resp.choices[0].message.content)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.random()
                logger.warning("vllm_chat attempt %d/%d failed: %s — retrying in %.1fs", attempt + 1, max_retries, exc, delay)
                await _asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


async def ensure_storage(runtime: RuntimeClients, settings: Settings, variant: str, rebuild: bool) -> None:
    text_index = settings.text_index_name(variant)
    proxy_index = settings.proxy_index_name(variant)

    for index_name, mapping in [(text_index, KB_TEXT_MAPPING), (proxy_index, KB_PAGE_PROXY_MAPPING)]:
        exists = await runtime.es.indices.exists(index=index_name)
        if exists and rebuild:
            await runtime.es.indices.delete(index=index_name)
            exists = False
        if not exists:
            await runtime.es.indices.create(index=index_name, body=mapping)

    text_collection = settings.text_collection_name(variant)
    proxy_collection = settings.proxy_collection_name(variant)
    image_collection = settings.image_collection_name(variant)

    collections: list[tuple[str, int]] = [
        (text_collection, settings.qdrant_text_vector_size),
        (image_collection, settings.qdrant_image_vector_size),
    ]
    if variant == "enhanced":
        collections.insert(1, (proxy_collection, settings.qdrant_text_vector_size))

    for collection_name, vec_size in collections:
        try:
            if rebuild:
                await runtime.qdrant.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        try:
            await runtime.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(size=vec_size, distance=qdrant_models.Distance.COSINE),
            )
        except Exception:
            pass


def _dedupe_non_empty(values: list[str] | None) -> list[str]:
    if not values:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _build_qdrant_filter(
    *,
    filter_doc_page_ids: list[str] | None,
    filter_document_ids: list[str] | None,
) -> qdrant_models.Filter | None:
    must_conditions: list[qdrant_models.FieldCondition] = []
    doc_page_ids = _dedupe_non_empty(filter_doc_page_ids)
    document_ids = _dedupe_non_empty(filter_document_ids)
    if doc_page_ids:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="doc_page_id",
                match=qdrant_models.MatchAny(any=doc_page_ids),
            )
        )
    if document_ids:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchAny(any=document_ids),
            )
        )
    return qdrant_models.Filter(must=must_conditions) if must_conditions else None


async def bm25_search(
    runtime: RuntimeClients,
    index: str,
    query: str,
    field: str,
    top_k: int,
    filter_doc_page_ids: list[str] | None = None,
    filter_document_ids: list[str] | None = None,
) -> list[dict]:
    filters: list[dict] = []
    doc_page_ids = _dedupe_non_empty(filter_doc_page_ids)
    document_ids = _dedupe_non_empty(filter_document_ids)
    if doc_page_ids:
        filters.append({"terms": {"doc_page_id": doc_page_ids}})
    if document_ids:
        filters.append({"terms": {"document_id": document_ids}})

    query_bool: dict = {"must": [{"match": {field: query}}]}
    if filters:
        query_bool["filter"] = filters

    body = {"query": {"bool": query_bool}, "size": top_k}
    resp = await runtime.es.search(index=index, body=body)
    return [{"_id": hit["_id"], "_score": hit["_score"], **hit["_source"]} for hit in resp["hits"]["hits"]]


async def qdrant_search(
    runtime: RuntimeClients,
    collection: str,
    query_vector: list[float],
    top_k: int,
    filter_doc_page_ids: list[str] | None = None,
    filter_document_ids: list[str] | None = None,
) -> list[dict]:
    query_filter = _build_qdrant_filter(
        filter_doc_page_ids=filter_doc_page_ids,
        filter_document_ids=filter_document_ids,
    )
    results = await runtime.qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    return [{"id": str(point.id), "score": point.score, **point.payload} for point in results.points]
