#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SlideVQA into this project's normalized eval format.")
    parser.add_argument("--slidevqa-dir", type=Path, default=Path("/data/SlideVQA"))
    parser.add_argument("--slidevqa-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-root", type=Path, default=Path("data/downloads"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="<=0 means keep full split. >0 limits processed samples (useful for smoke test).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail when rows are skipped due to invalid fields.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace prepared directory if it exists (default: enabled).",
    )
    return parser.parse_args()


def _jsonl_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_staging_dir(target_dir: Path, overwrite: bool) -> Path:
    if target_dir.exists() and not overwrite:
        raise SystemExit(f"Output directory already exists: {target_dir}. Use --overwrite to replace it.")
    staging_dir = target_dir.parent / f".{target_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def _commit_staging_dir(staging_dir: Path, target_dir: Path) -> None:
    backup_dir = target_dir.parent / f".{target_dir.name}.backup-{os.getpid()}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if target_dir.exists():
        target_dir.rename(backup_dir)
    try:
        staging_dir.rename(target_dir)
    except Exception:
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.rename(target_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def _sanitize_document_id(raw: str, fallback: str) -> str:
    normalized = re.sub(r"[\\/:]+", "_", raw.strip())
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized).strip("._-")
    return normalized or fallback


def _image_suffix_from_payload(image_value: object) -> str:
    if isinstance(image_value, dict):
        path_value = str(image_value.get("path", "")).strip()
        suffix = Path(path_value).suffix.lower()
        if suffix in VALID_IMAGE_SUFFIXES:
            return suffix
    return ".png"


def _save_image_payload(image_value: object, output_path: Path) -> None:
    if isinstance(image_value, dict):
        blob = image_value.get("bytes")
        if isinstance(blob, (bytes, bytearray)):
            output_path.write_bytes(bytes(blob))
            return
        source_path = str(image_value.get("path", "")).strip()
        if source_path:
            source = Path(source_path)
            if source.exists():
                shutil.copyfile(source, output_path)
                return

    save_fn = getattr(image_value, "save", None)
    if callable(save_fn):
        save_fn(output_path)
        return

    raise ValueError(f"Unsupported image payload type: {type(image_value)}")


def prepare_slidevqa(
    *,
    slidevqa_dir: Path,
    split: str,
    output_root: Path,
    overwrite: bool,
    strict: bool,
    max_samples: int,
) -> dict:
    if not slidevqa_dir.exists():
        raise SystemExit(f"SlideVQA dataset directory not found: {slidevqa_dir}")

    prepared_dir = output_root / "slidevqa"
    staging_dir = _prepare_staging_dir(prepared_dir, overwrite)

    try:
        dataset = load_dataset(str(slidevqa_dir), split=split)
        page_fields = [f"page_{i}" for i in range(1, 21)]
        for field in page_fields:
            if field in dataset.column_names:
                try:
                    dataset = dataset.cast_column(field, HFImage(decode=False))
                except Exception:
                    pass

        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        prepared_corpus_rows: list[dict] = []
        prepared_samples: list[dict] = []

        skipped_sample_rows = 0
        copied_pages = 0

        seen_doc_page_ids: set[str] = set()
        seen_decks: set[str] = set()
        deck_id_by_raw: dict[str, str] = {}
        raw_deck_by_id: dict[str, str] = {}

        for idx, item in enumerate(dataset):
            deck_name_raw = str(item.get("deck_name", "")).strip()
            if not deck_name_raw:
                deck_name_raw = f"deck_{idx:06d}"
            document_id = deck_id_by_raw.get(deck_name_raw, "")
            if not document_id:
                base_document_id = _sanitize_document_id(deck_name_raw, f"deck_{idx:06d}")
                document_id = base_document_id
                suffix = 2
                while document_id in raw_deck_by_id and raw_deck_by_id[document_id] != deck_name_raw:
                    document_id = f"{base_document_id}_{suffix}"
                    suffix += 1
                deck_id_by_raw[deck_name_raw] = document_id
                raw_deck_by_id[document_id] = deck_name_raw

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            qa_id = item.get("qa_id")
            qa_id_text = str(qa_id).strip() if qa_id is not None else ""

            evidence_pages_raw = item.get("evidence_pages", [])
            evidence_pages: list[int] = []
            if isinstance(evidence_pages_raw, list):
                for value in evidence_pages_raw:
                    try:
                        page_no = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= page_no <= 20 and page_no not in evidence_pages:
                        evidence_pages.append(page_no)

            page_path_by_no: dict[int, str] = {}
            retrieval_scope_doc_page_ids: list[str] = []

            for page_no in range(1, 21):
                page_field = f"page_{page_no}"
                image_value = item.get(page_field)
                if image_value is None:
                    continue

                suffix = _image_suffix_from_payload(image_value)
                page_rel_path = Path("images") / document_id / f"page_{page_no:04d}{suffix}"
                page_abs_path = staging_dir / page_rel_path
                doc_page_id = f"{document_id}:{page_no:04d}"

                retrieval_scope_doc_page_ids.append(doc_page_id)
                page_path_by_no[page_no] = str(page_rel_path)

                if doc_page_id in seen_doc_page_ids:
                    continue

                page_abs_path.parent.mkdir(parents=True, exist_ok=True)
                _save_image_payload(image_value, page_abs_path)

                seen_doc_page_ids.add(doc_page_id)
                copied_pages += 1
                prepared_corpus_rows.append(
                    {
                        "doc_page_id": doc_page_id,
                        "document_id": document_id,
                        "page_no": page_no,
                        "image_path": str(page_rel_path),
                    }
                )

            relevant_doc_page_ids = [
                f"{document_id}:{page_no:04d}" for page_no in evidence_pages if page_no in page_path_by_no
            ]

            if not question or not answer or not retrieval_scope_doc_page_ids or not relevant_doc_page_ids:
                skipped_sample_rows += 1
                continue

            primary_page_no = next((page_no for page_no in evidence_pages if page_no in page_path_by_no), 0)
            if primary_page_no <= 0:
                primary_page_no = min(page_path_by_no) if page_path_by_no else 0
            sample_image_path = page_path_by_no.get(primary_page_no, "")

            sample_id_parts = [split, str(idx), document_id]
            if qa_id_text:
                sample_id_parts.append(qa_id_text)
            sample_id = ":".join(sample_id_parts)

            prepared_samples.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "answers": [answer],
                    "document_id": document_id,
                    "page_no": primary_page_no,
                    "image_path": sample_image_path,
                    "relevant_doc_page_ids": relevant_doc_page_ids,
                    "retrieval_scope_doc_page_ids": retrieval_scope_doc_page_ids,
                    "retrieval_scope_document_ids": [document_id],
                    "deck_name": deck_name_raw,
                    "deck_url": str(item.get("deck_url", "")).strip(),
                    "qa_id": qa_id,
                    "evidence_pages": evidence_pages,
                }
            )
            seen_decks.add(document_id)

        if strict and skipped_sample_rows:
            raise SystemExit(f"Skipped {skipped_sample_rows} invalid sample rows in strict mode.")
        if not prepared_samples:
            raise SystemExit("No valid SlideVQA samples found.")

        _jsonl_write(staging_dir / "samples.jsonl", prepared_samples)
        _jsonl_write(staging_dir / "corpus_pages.jsonl", prepared_corpus_rows)

        manifest = {
            "dataset": "slidevqa",
            "source_dir": str(slidevqa_dir),
            "split": split,
            "requested_max_samples": max_samples,
            "num_samples": len(prepared_samples),
            "num_corpus_pages": len(prepared_corpus_rows),
            "num_unique_decks": len(seen_decks),
            "num_copied_images": copied_pages,
            "skipped_sample_rows": skipped_sample_rows,
        }
        with open(staging_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _commit_staging_dir(staging_dir, prepared_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = prepare_slidevqa(
        slidevqa_dir=args.slidevqa_dir,
        split=args.slidevqa_split,
        output_root=args.output_root,
        overwrite=args.overwrite,
        strict=args.strict,
        max_samples=args.max_samples,
    )

    print(json.dumps({"prepared": [manifest]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
