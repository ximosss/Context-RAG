from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_get(keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def _env_int(keys: tuple[str, ...], default: int) -> int:
    raw = _env_get(keys, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(keys: tuple[str, ...], default: bool) -> bool:
    raw = _env_get(keys, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(keys: tuple[str, ...], default: float) -> float:
    raw = _env_get(keys, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_optional_int(keys: tuple[str, ...]) -> int | None:
    raw = _env_get(keys, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _normalize_dataset_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "slidevqa"


@dataclass
class Settings:
    project_root: Path
    dataset_name: str
    dataset_key: str
    data_dir: Path
    downloads_dir: Path
    dataset_dir: Path
    experiment_dir: Path
    eval_runs_dir: Path
    es_url: str
    qdrant_url: str
    top_k: int
    bm25_rrf_weight: float
    dense_rrf_weight: float
    image_rrf_weight: float
    proxy_rrf_weight: float
    proxy_dense_rrf_weight: float
    qdrant_text_vector_size: int
    qdrant_image_vector_size: int
    use_page_image: bool
    build_page_workers: int
    build_llm_concurrency: int
    build_embed_concurrency: int
    text_embed_batch_size: int
    image_embed_batch_size: int
    qdrant_upsert_batch_size: int
    es_bulk_ops_batch_size: int
    ocr_max_tokens: int
    contextual_chunk_max_tokens: int
    page_proxy_max_tokens: int
    qwen_embed_base_url: str
    qwen_embed_api_key: str
    qwen_embed_model: str
    qwen_query_instruction: str
    qwen_document_instruction: str
    qwen_multimodal_embed_base_url: str
    qwen_multimodal_embed_api_key: str
    qwen_multimodal_embed_model: str
    qwen_multimodal_query_instruction: str
    qwen_multimodal_document_instruction: str
    vllm_base_url: str
    vllm_model_name: str
    vllm_seed: int | None

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        load_env_file(project_root / ".env")

        raw_data_dir = Path(_env_get(("data_dir", "DATA_DIR"), "data"))
        data_dir = raw_data_dir if raw_data_dir.is_absolute() else (project_root / raw_data_dir).resolve()
        downloads_dir = data_dir / "downloads"
        dataset_name = _env_get(("dataset_name", "DATASET_NAME"), "slidevqa")
        dataset_key = _normalize_dataset_name(dataset_name)

        dataset_dir_env = _env_get(("dataset_dir", "DATASET_DIR"), "")
        if dataset_dir_env.strip():
            raw_dataset_dir = Path(dataset_dir_env)
            dataset_dir = raw_dataset_dir if raw_dataset_dir.is_absolute() else (project_root / raw_dataset_dir).resolve()
        else:
            dataset_dir = downloads_dir / dataset_key

        experiment_dir = data_dir / "experiments" / dataset_key
        eval_runs_dir = data_dir / "eval_runs"

        return cls(
            project_root=project_root,
            dataset_name=dataset_name,
            dataset_key=dataset_key,
            data_dir=data_dir,
            downloads_dir=downloads_dir,
            dataset_dir=dataset_dir,
            experiment_dir=experiment_dir,
            eval_runs_dir=eval_runs_dir,
            es_url=_env_get(("es_url", "ES_URL"), "http://localhost:9200"),
            qdrant_url=_env_get(("qdrant_url", "QDRANT_URL"), "http://localhost:6333"),
            top_k=_env_int(("top_k", "TOP_K"), 20),
            bm25_rrf_weight=_env_float(("bm25_rrf_weight", "BM25_RRF_WEIGHT"), 1.0),
            dense_rrf_weight=_env_float(("dense_rrf_weight", "DENSE_RRF_WEIGHT"), 1.0),
            image_rrf_weight=_env_float(("image_rrf_weight", "IMAGE_RRF_WEIGHT"), 1.0),
            proxy_rrf_weight=_env_float(("proxy_rrf_weight", "PROXY_RRF_WEIGHT"), 0.35),
            proxy_dense_rrf_weight=_env_float(("proxy_dense_rrf_weight", "PROXY_DENSE_RRF_WEIGHT"), 0.5),
            qdrant_text_vector_size=_env_int(("qdrant_text_vector_size", "QDRANT_TEXT_VECTOR_SIZE"), 2560),
            qdrant_image_vector_size=_env_int(("qdrant_image_vector_size", "QDRANT_IMAGE_VECTOR_SIZE"), 2048),
            use_page_image=_env_bool(("use_page_image", "USE_PAGE_IMAGE"), True),
            build_page_workers=_env_int(("build_page_workers", "BUILD_PAGE_WORKERS"), 16),
            build_llm_concurrency=_env_int(("build_llm_concurrency", "BUILD_LLM_CONCURRENCY"), 16),
            build_embed_concurrency=_env_int(("build_embed_concurrency", "BUILD_EMBED_CONCURRENCY"), 4),
            text_embed_batch_size=_env_int(("text_embed_batch_size", "TEXT_EMBED_BATCH_SIZE"), 256),
            image_embed_batch_size=_env_int(("image_embed_batch_size", "IMAGE_EMBED_BATCH_SIZE"), 64),
            qdrant_upsert_batch_size=_env_int(("qdrant_upsert_batch_size", "QDRANT_UPSERT_BATCH_SIZE"), 128),
            es_bulk_ops_batch_size=_env_int(("es_bulk_ops_batch_size", "ES_BULK_OPS_BATCH_SIZE"), 4000),
            ocr_max_tokens=_env_int(("ocr_max_tokens", "OCR_MAX_TOKENS"), 4096),
            contextual_chunk_max_tokens=_env_int(("contextual_chunk_max_tokens", "CONTEXTUAL_CHUNK_MAX_TOKENS"), 768),
            page_proxy_max_tokens=_env_int(("page_proxy_max_tokens", "PAGE_PROXY_MAX_TOKENS"), 768),
            qwen_embed_base_url=_env_get(("qwen_embed_base_url", "QWEN_EMBED_BASE_URL"), "http://localhost:8002/v1"),
            qwen_embed_api_key=_env_get(("qwen_embed_api_key", "QWEN_EMBED_API_KEY"), "EMPTY"),
            qwen_embed_model=_env_get(("qwen_embed_model", "QWEN_EMBED_MODEL"), "Qwen/Qwen3-Embedding-4B"),
            qwen_query_instruction=_env_get(
                ("qwen_query_instruction", "QWEN_QUERY_INSTRUCTION"),
                "Represent the query for retrieving relevant document passages.",
            ),
            qwen_document_instruction=_env_get(
                ("qwen_document_instruction", "QWEN_DOCUMENT_INSTRUCTION"),
                "Represent the document for retrieval.",
            ),
            qwen_multimodal_embed_base_url=_env_get(
                ("qwen_multimodal_embed_base_url", "QWEN_MULTIMODAL_EMBED_BASE_URL"),
                "http://localhost:8003/v1",
            ),
            qwen_multimodal_embed_api_key=_env_get(
                ("qwen_multimodal_embed_api_key", "QWEN_MULTIMODAL_EMBED_API_KEY"),
                "EMPTY",
            ),
            qwen_multimodal_embed_model=_env_get(
                ("qwen_multimodal_embed_model", "QWEN_MULTIMODAL_EMBED_MODEL"),
                "Qwen/Qwen3-VL-Embedding-2B",
            ),
            qwen_multimodal_query_instruction=_env_get(
                ("qwen_multimodal_query_instruction", "QWEN_MULTIMODAL_QUERY_INSTRUCTION"),
                "Represent the query for retrieving relevant document pages.",
            ),
            qwen_multimodal_document_instruction=_env_get(
                ("qwen_multimodal_document_instruction", "QWEN_MULTIMODAL_DOCUMENT_INSTRUCTION"),
                "Represent the document page for retrieval.",
            ),
            vllm_base_url=_env_get(("vllm_base_url", "VLLM_BASE_URL"), "http://localhost:8001/v1"),
            vllm_model_name=_env_get(("vllm_model_name", "VLLM_MODEL_NAME"), "qwen3.5-35b-a3b"),
            vllm_seed=_env_optional_int(("vllm_seed", "VLLM_SEED")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.eval_runs_dir.mkdir(parents=True, exist_ok=True)

    def text_index_name(self, variant: str) -> str:
        return f"kb_text_{self.dataset_key}_{variant}"

    def proxy_index_name(self, variant: str) -> str:
        return f"kb_page_proxy_{self.dataset_key}_{variant}"

    def text_collection_name(self, variant: str) -> str:
        return f"kb_text_dense_{self.dataset_key}_{variant}"

    def proxy_collection_name(self, variant: str) -> str:
        return f"kb_page_proxy_dense_{self.dataset_key}_{variant}"

    def image_collection_name(self, variant: str) -> str:
        return f"kb_page_image_{self.dataset_key}_{variant}"

    def artifact_dir(self, variant: str) -> Path:
        return self.experiment_dir / variant
