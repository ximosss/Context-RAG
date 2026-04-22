from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_page_id: str
    document_id: str
    page_no: int
    chunk_index: int
    chunk_type: str
    text_raw: str
    contextual_chunk_text: str
    text_for_bm25: str
    text_for_dense: str
    page_title: str
    section_path: str
    image_path: str


@dataclass
class PageRecord:
    doc_page_id: str
    document_id: str
    page_no: int
    image_path: str
    ocr_text: str
    page_title: str
    section_path: str
    page_proxy_text: str
    has_table: bool
    has_figure: bool


@dataclass
class EvalSample:
    sample_id: str
    question: str
    answers: list[str]
    document_id: str
    page_no: int
    image_path: str
    relevant_doc_page_ids: list[str]
    retrieval_scope_doc_page_ids: list[str]
    retrieval_scope_document_ids: list[str]
