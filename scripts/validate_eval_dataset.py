#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared SlideVQA dataset layout for this RAG pipeline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/downloads/slidevqa"))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def validate_dataset(dataset_dir: Path, max_samples: int | None) -> dict:
    samples_path = dataset_dir / "samples.jsonl"
    corpus_path = dataset_dir / "corpus_pages.jsonl"
    if not samples_path.exists():
        raise SystemExit(f"Missing file: {samples_path}")
    if not corpus_path.exists():
        raise SystemExit(f"Missing file: {corpus_path}")

    samples = _read_jsonl(samples_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    corpus_rows = _read_jsonl(corpus_path)

    corpus_doc_page_ids: set[str] = set()
    missing_corpus_images: list[str] = []
    for row in corpus_rows:
        doc_page_id = str(row.get("doc_page_id", "")).strip()
        image_rel = str(row.get("image_path", "")).strip()
        if doc_page_id:
            corpus_doc_page_ids.add(doc_page_id)
        if image_rel:
            image_abs = dataset_dir / image_rel
            if not image_abs.exists():
                missing_corpus_images.append(str(image_abs))

    missing_sample_images: list[str] = []
    unresolved_gold_doc_page_ids: list[str] = []
    unresolved_scope_doc_page_ids: list[str] = []
    scope_sizes: list[int] = []

    for sample in samples:
        image_rel = str(sample.get("image_path", "")).strip()
        if image_rel and not (dataset_dir / image_rel).exists():
            missing_sample_images.append(str(dataset_dir / image_rel))

        relevant_raw = sample.get("relevant_doc_page_ids", [])
        relevant_doc_page_ids = (
            [str(item).strip() for item in relevant_raw if str(item).strip()] if isinstance(relevant_raw, list) else []
        )
        for doc_page_id in relevant_doc_page_ids:
            if doc_page_id not in corpus_doc_page_ids:
                unresolved_gold_doc_page_ids.append(doc_page_id)

        scope_raw = sample.get("retrieval_scope_doc_page_ids", [])
        scope_doc_page_ids = [str(item).strip() for item in scope_raw if str(item).strip()] if isinstance(scope_raw, list) else []
        if scope_doc_page_ids:
            scope_sizes.append(len(scope_doc_page_ids))
        for doc_page_id in scope_doc_page_ids:
            if doc_page_id not in corpus_doc_page_ids:
                unresolved_scope_doc_page_ids.append(doc_page_id)

    payload = {
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_dir.name,
        "samples": len(samples),
        "corpus_pages": len(corpus_rows),
        "missing_sample_images": len(missing_sample_images),
        "missing_corpus_images": len(missing_corpus_images),
        "unresolved_gold_doc_page_ids": len(unresolved_gold_doc_page_ids),
        "unresolved_scope_doc_page_ids": len(unresolved_scope_doc_page_ids),
        "scope_size_min": min(scope_sizes) if scope_sizes else 0,
        "scope_size_max": max(scope_sizes) if scope_sizes else 0,
        "scope_size_avg": (sum(scope_sizes) / len(scope_sizes)) if scope_sizes else 0.0,
        "missing_sample_image_examples": missing_sample_images[:10],
        "missing_corpus_image_examples": missing_corpus_images[:10],
        "unresolved_gold_doc_page_id_examples": unresolved_gold_doc_page_ids[:10],
        "unresolved_scope_doc_page_id_examples": unresolved_scope_doc_page_ids[:10],
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = validate_dataset(args.dataset_dir, args.max_samples)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if (
        payload["missing_sample_images"]
        or payload["missing_corpus_images"]
        or payload["unresolved_gold_doc_page_ids"]
        or payload["unresolved_scope_doc_page_ids"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
