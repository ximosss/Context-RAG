from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse


# Patterns that indicate untagged thinking/reasoning leaked into the output.
_UNTAGGED_THINKING_RE = re.compile(
    r"^(?:Got it|Okay|Alright|Sure|Let me|Let's|So,?\s|First,?\s|Looking at|The question|The user|The evidence|The answer|The problem|I need to|I'll|Let us|Hmm)",
    flags=re.IGNORECASE,
)
_META_REASONING_RE = re.compile(
    r"(?:\b(?:the problem says|the user says|return plain text only|normal reading order|let me|let's|i need to|i should|check the image|looking at the image|go step by step|list all visible text|the image has|the chart has|the page has|start with|starts with|output only|need to be concise|final answer should be|the legend is|the image shows|the text for the legend|logo text)\b|(?:^|\s)(?:wait,|but in the image|then the|also,|so the ))",
    flags=re.IGNORECASE,
)
_FOOTER_RE = re.compile(
    r"^(?:\d+\s*\|\s*.+|©\s*\d{4}.+|all rights reserved\.?|source:\s*.+)$",
    flags=re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(
    r"^(?:[-*]\s*)?(?:\d+\.\s*)?(title|legend|note|footer|caption|heading|header|page title|main title|subtitle|source)\s*:\s*(.+)$",
    flags=re.IGNORECASE,
)
_EXTENDED_LABEL_PREFIX_RE = re.compile(
    r"^(?:[-*]\s*)?(?:\d+\.\s*)?(?:the\s+)?(note below the chart|top title|bottom right|legend labels?|x-axis labels?|y-axis values?|bar values?|function names?|copyright)\s*:\s*(.+)$",
    flags=re.IGNORECASE,
)
_CITATION_SUFFIX_RE = re.compile(r"\s*(?:\[\s*Page\s+\d+\s*\]|\(\s*Page\s+\d+\s*\))\s*$", flags=re.IGNORECASE)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def should_trust_env(url: str) -> bool:
    host = urlparse(url).hostname
    return host not in {"localhost", "127.0.0.1", "::1"}


def sanitize_llm_output(text: str | None) -> str:
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Core rule: if there's a </think> anywhere, everything after the LAST one
    # is the real content. Handles all variants:
    #   <think>...</think>real    — normal tagged thinking
    #   ...</think>real           — open tag swallowed by chat template
    #   <think>...</think>        — thinking only, no real content
    last_close = cleaned.rfind("</think>")
    if last_close == -1:
        last_close = cleaned.rfind("</thinking>")
    if last_close != -1:
        tag_end = cleaned.index(">", last_close) + 1
        after = cleaned[tag_end:].strip()
        if after:
            cleaned = after
        else:
            # Nothing after close tag — try content before first open tag
            first_open = re.search(r"<\s*think(?:ing)?\b", cleaned, re.IGNORECASE)
            if first_open:
                before = cleaned[: first_open.start()].strip()
                if before:
                    cleaned = before
                else:
                    cleaned = ""

    # Fallback: strip untagged thinking preamble ("Got it, let's...", "Okay, ...")
    if _UNTAGGED_THINKING_RE.match(cleaned):
        for sep in ["\n\n", "\n---\n", "\n- "]:
            parts = cleaned.split(sep, 1)
            if len(parts) == 2 and len(parts[1].strip()) > len(parts[0].strip()) // 4:
                cleaned = parts[1]
                break

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def looks_like_reasoning_leak(text: str | None) -> bool:
    normalized = sanitize_llm_output(text)
    if not normalized:
        return False
    return bool(_UNTAGGED_THINKING_RE.search(normalized) or _META_REASONING_RE.search(normalized))


def _strip_balanced_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _clean_generated_line(line: str) -> str:
    candidate = line.strip().strip("-* ").strip()
    if not candidate:
        return ""

    candidate = re.sub(
        r"\([^)]*(?:but in the image|with the labels|with the color indicators|logo text|the image shows)[^)]*\)",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()

    label_match = _LABEL_PREFIX_RE.match(candidate)
    if label_match:
        label = label_match.group(1).lower()
        value = _strip_balanced_quotes(label_match.group(2))
        if label == "source":
            return f"Source: {value}" if value else ""
        return value

    extended_match = _EXTENDED_LABEL_PREFIX_RE.match(candidate)
    if extended_match:
        value = _strip_balanced_quotes(extended_match.group(2))
        return value

    if _META_REASONING_RE.search(candidate) or _UNTAGGED_THINKING_RE.match(candidate):
        return ""

    candidate = re.sub(r"\([^)]*\)", "", candidate).strip()
    candidate = re.sub(r"\s+(?:as the legend|with the labels|with the color indicators)\.?$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:\d+\.\s*)", "", candidate).strip()
    candidate = _strip_balanced_quotes(candidate)
    if candidate.lower().startswith(("then ", "also ", "so the ", "so ", "but ", "next, ")):
        return ""
    return candidate


def clean_generated_text(text: str | None) -> str:
    normalized = sanitize_llm_output(text)
    if not normalized:
        return ""

    cleaned_lines: list[str] = []
    seen: set[str] = set()
    changed = False
    for raw_line in normalized.splitlines():
        line = _clean_generated_line(raw_line)
        if line != raw_line.strip():
            changed = True
        if not line:
            changed = True
            continue
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key in seen:
            changed = True
            continue
        seen.add(key)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned:
        return cleaned if changed else normalized
    return ""


def strip_answer_citations(text: str | None) -> str:
    cleaned = sanitize_llm_output(text)
    if not cleaned:
        return ""
    return _CITATION_SUFFIX_RE.sub("", cleaned).strip()


def is_noise_chunk(text: str | None) -> bool:
    normalized = clean_generated_text(text)
    if not normalized:
        return True
    if looks_like_reasoning_leak(normalized):
        return True
    if _FOOTER_RE.match(normalized):
        return True

    stripped = normalized.strip()
    if len(stripped.split()) <= 1:
        alpha_chars = sum(ch.isalpha() for ch in stripped)
        return alpha_chars < 4

    alpha_chars = sum(ch.isalpha() for ch in stripped)
    digit_chars = sum(ch.isdigit() for ch in stripped)
    if alpha_chars == 0:
        return True
    if digit_chars > alpha_chars * 3 and alpha_chars < 8:
        return True
    if re.fullmatch(r"[\d\s.%|:/()+-]+", stripped):
        return True
    return False


_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*[\"']?(.+?)(?:[\"'.\n]|$)",
    flags=re.IGNORECASE,
)
_THEREFORE_RE = re.compile(
    r"(?:therefore|thus|hence|so|in conclusion)[,:]?\s+(?:the answer is\s+)?(.+?)(?:[.\n]|$)",
    flags=re.IGNORECASE,
)


def extract_short_answer(raw: str) -> str:
    """Post-process a QA model output to extract just the short answer."""
    text = strip_answer_citations(raw)
    if not text:
        return ""

    compact = re.sub(r"\s+", " ", text).strip()

    # Already short and clean — return as-is.
    if len(text.split()) <= 15 and not _UNTAGGED_THINKING_RE.match(text):
        return text

    # Try explicit "the answer is X" pattern.
    for pattern in (_FINAL_ANSWER_RE, _THEREFORE_RE):
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if candidate and candidate.lower() not in ("unknown", ""):
                return candidate

    # Take the last short non-empty line — models often put the final answer last.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip().rstrip(".")
        if len(last.split()) <= 15 and not _UNTAGGED_THINKING_RE.match(last):
            return last

    # Some models emit "answer should be X" / "is X" style phrasing.
    extra_patterns = [
        re.compile(r"(?:answer\s+should\s+be|should\s+be|is)\s+([A-Za-z0-9.%$€£¥,/\-+ ]{1,40})$", re.IGNORECASE),
        re.compile(r"^\s*(\d+(?:\.\d+)?\s*[%$€£¥]?(?:\s*[A-Za-z]+)?)\s*$", re.IGNORECASE),
    ]
    for pattern in extra_patterns:
        m = pattern.search(compact)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if candidate:
                return candidate

    # Fallback: return the full sanitized text (will hurt F1 but at least
    # partial matches can still score if the answer is somewhere in there).
    return text


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:512]
    return ""


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_paragraphs(text: str) -> list[str]:
    return re.split(r"\n{2,}", text)


def classify_paragraph(text: str) -> str:
    lower = text.lower().strip()
    if text.count("|") > 3 or text.count("\t") > 3:
        return "table"
    if re.match(r"^(figure|fig\.|table|图|表)\s*\d", lower):
        return "caption"
    if len(text) < 100 and not text.endswith(".") and re.match(r"^[\d.]+\s", text):
        return "title"
    return "body"


def approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_page(
    page_text: str,
    page_title: str = "",
    section_path: str = "",
    max_chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    del page_title
    if not page_text.strip():
        return []

    paragraphs = split_paragraphs(page_text)
    chunks: list[dict] = []
    buffer = ""
    idx = 0
    stride = max(1, max_chunk_size - max(0, overlap))

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        chunk_type = classify_paragraph(para)
        if buffer and approx_tokens(buffer + "\n" + para) > max_chunk_size:
            chunks.append(
                {
                    "text_raw": buffer.strip(),
                    "chunk_type": "body",
                    "chunk_index": idx,
                    "section_path": section_path,
                }
            )
            idx += 1
            words = buffer.split()
            buffer = " ".join(words[-overlap:]) if len(words) > overlap else ""

        if chunk_type == "body" and approx_tokens(para) > max_chunk_size:
            if buffer.strip():
                chunks.append(
                    {
                        "text_raw": buffer.strip(),
                        "chunk_type": "body",
                        "chunk_index": idx,
                        "section_path": section_path,
                    }
                )
                idx += 1
                buffer = ""
            words = para.split()
            start = 0
            while start < len(words):
                end = min(start + max_chunk_size, len(words))
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    chunks.append(
                        {
                            "text_raw": chunk_text,
                            "chunk_type": "body",
                            "chunk_index": idx,
                            "section_path": section_path,
                        }
                    )
                    idx += 1
                if end >= len(words):
                    break
                start += stride
            continue

        if chunk_type in {"table", "caption", "title"}:
            if buffer.strip():
                chunks.append(
                    {
                        "text_raw": buffer.strip(),
                        "chunk_type": "body",
                        "chunk_index": idx,
                        "section_path": section_path,
                    }
                )
                idx += 1
                buffer = ""
            chunks.append(
                {
                    "text_raw": para,
                    "chunk_type": chunk_type,
                    "chunk_index": idx,
                    "section_path": section_path,
                }
            )
            idx += 1
        else:
            buffer = (buffer + "\n" + para).strip() if buffer else para

    if buffer.strip():
        chunks.append(
            {
                "text_raw": buffer.strip(),
                "chunk_type": "body",
                "chunk_index": idx,
                "section_path": section_path,
            }
        )

    return chunks
