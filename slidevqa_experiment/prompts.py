from __future__ import annotations

OCR_PAGE_PROMPT = """\
Extract all readable text from this document page in normal reading order.

Rules:
- Return plain text only
- Preserve line breaks when they help readability
- Do not add explanations or markup
- Do not describe the image
- Do not say what you are about to do
- Do not mention the instructions
- If the page is mostly non-textual, return any visible labels, numbers, and headings you can read"""

CONTEXTUAL_CHUNK_PROMPT = """\
You are generating retrieval-only context for a chunk from a document page.
Write 2-4 concise sentences that make this chunk self-explanatory for search.

Document title: {doc_title}
Page title: {page_title}
Section: {section_path}
Page number: {page_no}

Neighboring context (before): {prev_text}
Neighboring context (after): {next_text}

Current chunk:
---
{chunk_text}
---

Focus on:
- what document/page/section this chunk belongs to
- the entities, dates, countries, products, metrics, and units it mentions or implies
- the table/chart/figure/topic this chunk relates to
- short lexical anchors that would help both BM25 and semantic retrieval

Do not answer questions. Do not restate instructions. Output ONLY the contextual sentences, nothing else.
Write in the same language as the chunk."""

PAGE_PROXY_PROMPT = """\
You are generating a retrieval-only page proxy for a slide or document page.
Use the page image and OCR text together to produce a compact page-level description that helps search find this page.

Document title: {doc_title}
Page number: {page_no}
Page title: {page_title}
Section: {section_path}
Previous page title: {prev_title}
Next page title: {next_title}
Has table: {has_table}
Has chart/figure: {has_figure}

Page text:
---
{page_text}
---

Include, when visible or inferable:
- the overall topic of the page
- key entities such as countries, companies, products, regions, years, currencies, units, metrics
- chart/table subjects, axis concepts, legend categories, compared groups, and notable values or ranges
- question styles this page can answer, such as comparison, lookup, ranking, amount, year, country, or category questions

Prefer concrete retrieval anchors over generic prose.
Do not answer any user question directly.
Output ONLY the proxy text, nothing else. Write in the same language as the page text."""

ANSWER_PROMPT = """\
You are a document QA assistant. Answer the question using only the provided page image and the raw OCR/text evidence from that same page.

OCR Evidence:
{chunk_evidence}

Question: {question}

Instructions:
- Output ONLY the final answer, nothing else
- Do NOT explain reasoning or show your work
- Do NOT add citations, page numbers, or references
- Keep the answer as short as possible: a number, name, date, or short phrase
- If the question gives explicit options, output one option exactly
- If the question asks for a year, output only the year
- If the question asks for a value, output only the value with units if present
- Always attempt an answer based on the best available evidence
- Only output "Unknown" if the evidence is completely unrelated to the question
"""

ANSWER_REWRITE_PROMPT = """\
You are finalizing a QA response. Convert the draft answer into the shortest possible final answer using the question.

Question: {question}

Draft answer:
---
{draft_answer}
---

Rules:
- Output ONLY the final answer
- No explanations, no citations, no page numbers
- Keep the answer as short as possible
- If the question gives explicit options, return one option exactly
- If the question asks for a year, return only the year
- If the draft gives a numeric value with units or symbols, preserve them
- If the draft is truly uncertain or unsupported, output "Unknown"
"""
