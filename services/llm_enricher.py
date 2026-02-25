"""
LLM Metadata Enricher - Contextual Retrieval for Tin học (CS) SGK chunks

Inspired by Anthropic's Contextual Retrieval technique:
https://www.anthropic.com/news/contextual-retrieval

Specialised for Vietnamese Computer Science textbooks (lớp 6-12).
For each OCR chunk the LLM extracts:
  - chapter / section / lesson name
  - content_type: CS-specific (algorithm, code_snippet, pseudocode, ...)
  - cs_domain: which CS sub-field (programming, algorithm, networking, ...)
  - programming_language: if code/pseudocode is present (Python, Pascal, ...)
  - key topics (CS keywords)
  - a brief Vietnamese summary

The enriched text  =  [context header]  +  [original chunk text]
is then used for embedding, making vector search far more accurate.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là trợ lý AI chuyên phân tích nội dung sách giáo khoa Tin học Việt Nam (lớp 6-12).
Các sách này bao gồm các chủ đề: lập trình (Python, Scratch, Pascal), giải thuật và
cấu trúc dữ liệu, cơ sở dữ liệu (SQL), mạng máy tính, phần cứng/phần mềm,
công cụ văn phòng (Word, Excel), an toàn thông tin, trí tuệ nhân tạo.

Nhiệm vụ của bạn là trích xuất metadata cấu trúc từ một đoạn văn bản và
trả về JSON thuần túy (không có markdown, không có ```json ```...).
"""

_USER_PROMPT_TEMPLATE = """\
THÔNG TIN SÁCH TIN HỌC:
- Tên sách: {book_name}
- Nhà xuất bản: {publisher}
- Lớp/Cấp độ: {grade}

ĐOẠN VĂN BẢN (chunk {chunk_index}/{total_chunks}):
\"\"\"
{chunk_text}
\"\"\"

Hãy phân tích đoạn trên và trả về JSON với đúng các trường sau:
{{
  "chapter": "tên chương hoặc chủ đề lớn (ví dụ: 'Chương 3: Lập trình Python', để trống nếu không rõ)",
  "section": "tên bài học hoặc mục cụ thể (ví dụ: 'Bài 7: Câu lệnh lặp', để trống nếu không rõ)",
  "content_type": "chọn MỘT trong: definition | algorithm_steps | code_snippet | pseudocode | flowchart_description | example | exercise | explanation | comparison_table | hardware_concept | software_concept | introduction | summary | other",
  "cs_domain": "chọn MỘT trong: programming | algorithm_data_structure | database | networking | hardware_software | office_tools | security | ai_ml | computational_thinking | general | other",
  "programming_language": "ngôn ngữ lập trình nếu có code/pseudocode trong đoạn (Python | Pascal | Scratch | SQL | HTML | JavaScript | để trống nếu không có code)",
  "topics": ["từ khoá Tin học 1", "từ khoá 2", "từ khoá 3"],
  "summary": "mô tả ngắn 1-2 câu về nội dung đoạn này dưới góc độ môn Tin học"
}}

Lưu ý:
- content_type 'algorithm_steps': mô tả các bước của giải thuật bằng ngôn ngữ tự nhiên
- content_type 'pseudocode': giả mã hoặc mã lệnh ví dụ
- content_type 'code_snippet': đoạn code thực tế (Python, Pascal...)
- cs_domain 'computational_thinking': tư duy máy tính, phân tích bài toán
- Chỉ trả về JSON, không giải thích thêm.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Result schema
# ──────────────────────────────────────────────────────────────────────────────

CONTENT_TYPES = {
    "definition",
    "algorithm_steps",
    "code_snippet",
    "pseudocode",
    "flowchart_description",
    "example",
    "exercise",
    "explanation",
    "comparison_table",
    "hardware_concept",
    "software_concept",
    "introduction",
    "summary",
    "other",
}

CS_DOMAINS = {
    "programming",
    "algorithm_data_structure",
    "database",
    "networking",
    "hardware_software",
    "office_tools",
    "security",
    "ai_ml",
    "computational_thinking",
    "general",
    "other",
}

PROGRAMMING_LANGUAGES = {"Python", "Pascal", "Scratch", "SQL", "HTML", "JavaScript"}

_FALLBACK_META: Dict[str, Any] = {
    "chapter": "",
    "section": "",
    "content_type": "other",
    "cs_domain": "other",
    "programming_language": "",
    "topics": [],
    "summary": "",
}


# Human-readable labels for CS-specific values
_CONTENT_TYPE_LABEL = {
    "algorithm_steps": "Các bước giải thuật",
    "code_snippet": "Đoạn code",
    "pseudocode": "Giả mã",
    "flowchart_description": "Lưu đồ",
    "comparison_table": "Bảng so sánh",
    "hardware_concept": "Khái niệm phần cứng",
    "software_concept": "Khái niệm phần mềm",
    "definition": "Định nghĩa",
    "example": "Ví dụ",
    "exercise": "Bài tập",
    "explanation": "Giải thích",
    "introduction": "Giới thiệu",
    "summary": "Tổng kết",
}

_CS_DOMAIN_LABEL = {
    "programming": "Lập trình",
    "algorithm_data_structure": "Giải thuật & CTDL",
    "database": "Cơ sở dữ liệu",
    "networking": "Mạng máy tính",
    "hardware_software": "Phần cứng/Phần mềm",
    "office_tools": "Công cụ văn phòng",
    "security": "An toàn thông tin",
    "ai_ml": "Trí tuệ nhân tạo",
    "computational_thinking": "Tư duy máy tính",
}


def _build_context_header(meta: Dict[str, Any]) -> str:
    """
    Build a short plain-text header that gets prepended to the chunk
    before embedding.  Example output:

        [Chương: Chương 3: Lập trình Python | Bài: Câu lệnh lặp
         Loại: Đoạn code | Lĩnh vực: Lập trình | Ngôn ngữ: Python
         Chủ đề: vòng lặp for, range, danh sách
         Tóm tắt: Hướng dẫn dùng vòng lặp for để duyệt qua các phần tử.]
    """
    parts = []

    if meta.get("chapter"):
        parts.append(f"Chương: {meta['chapter']}")
    if meta.get("section"):
        parts.append(f"Bài: {meta['section']}")

    ct = meta.get("content_type", "")
    if ct and ct != "other":
        parts.append(f"Loại: {_CONTENT_TYPE_LABEL.get(ct, ct)}")

    domain = meta.get("cs_domain", "")
    if domain and domain not in ("other", "general"):
        parts.append(f"Lĩnh vực: {_CS_DOMAIN_LABEL.get(domain, domain)}")

    lang = meta.get("programming_language", "")
    if lang:
        parts.append(f"Ngôn ngữ: {lang}")

    header_lines = []
    if parts:
        header_lines.append(" | ".join(parts))
    if meta.get("topics"):
        header_lines.append("Chủ đề: " + ", ".join(meta["topics"][:6]))
    if meta.get("summary"):
        header_lines.append("Tóm tắt: " + meta["summary"])

    if not header_lines:
        return ""

    return "[" + "\n ".join(header_lines) + "]\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main enricher class
# ──────────────────────────────────────────────────────────────────────────────

class LLMEnricher:
    """
    Uses an OpenAI chat model to extract structured metadata from each chunk,
    then builds a *contextual_text*  =  [header]  +  [chunk]  for embedding.

    Parameters
    ----------
    api_key:        OpenAI API key
    model:          Chat model to use (default: gpt-4o-mini for cost efficiency)
    max_concurrency: Max parallel LLM calls (rate-limit friendly)
    enabled:        If False, enrich() is a no-op (returns original chunks unchanged)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
        enabled: bool = True,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_concurrency = max_concurrency
        self.enabled = enabled
        self._semaphore = asyncio.Semaphore(max_concurrency)
        logger.info(
            f"LLMEnricher initialized | model={model} | "
            f"concurrency={max_concurrency} | enabled={enabled}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def enrich_chunks(
        self,
        chunks_with_pages: List[Dict[str, Any]],
        book_name: str,
        publisher: str,
        grade: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich a list of chunk dicts (as produced by DocumentProcessor).

        Each input dict has at least: {"text": str, "pages": list, ...}
        Each output dict gains:
          - "llm_metadata":   raw dict from LLM  (chapter, section, ...)
          - "contextual_text": enriched text used for embedding
          - original keys preserved unchanged

        If enrichment is disabled or an individual call fails, the chunk is
        returned as-is with empty llm_metadata and contextual_text = text.
        """
        if not self.enabled:
            for chunk in chunks_with_pages:
                chunk.setdefault("llm_metadata", _FALLBACK_META.copy())
                chunk.setdefault("contextual_text", chunk["text"])
            return chunks_with_pages

        total = len(chunks_with_pages)
        logger.info(f"Enriching {total} chunks via LLM ({self.model}) ...")

        tasks = [
            self._enrich_single(chunk, idx, total, book_name, publisher, grade)
            for idx, chunk in enumerate(chunks_with_pages)
        ]
        enriched = await asyncio.gather(*tasks, return_exceptions=False)

        # Count successes
        successes = sum(1 for c in enriched if c.get("llm_metadata") != _FALLBACK_META)
        logger.info(f"LLM enrichment complete: {successes}/{total} chunks enriched")
        return enriched

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    async def _enrich_single(
        self,
        chunk: Dict[str, Any],
        idx: int,
        total: int,
        book_name: str,
        publisher: str,
        grade: Optional[str],
    ) -> Dict[str, Any]:
        """Call LLM for one chunk, with semaphore and error fallback."""
        async with self._semaphore:
            try:
                meta = await self._call_llm(
                    chunk_text=chunk["text"],
                    chunk_index=idx + 1,
                    total_chunks=total,
                    book_name=book_name,
                    publisher=publisher,
                    grade=grade or "",
                )
                header = _build_context_header(meta)
                contextual_text = header + chunk["text"] if header else chunk["text"]

                return {
                    **chunk,
                    "llm_metadata": meta,
                    "contextual_text": contextual_text,
                }

            except Exception as e:
                logger.warning(
                    f"LLM enrichment failed for chunk {idx + 1}/{total}: {e} — using raw text"
                )
                return {
                    **chunk,
                    "llm_metadata": _FALLBACK_META.copy(),
                    "contextual_text": chunk["text"],
                }

    async def _call_llm(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        book_name: str,
        publisher: str,
        grade: str,
    ) -> Dict[str, Any]:
        """Single LLM call; returns parsed metadata dict."""
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            book_name=book_name,
            publisher=publisher,
            grade=grade or "Không xác định",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            # Limit chunk to 2000 chars to reduce token cost
            chunk_text=chunk_text[:2000],
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        meta = json.loads(raw)

        # Normalise / validate
        meta.setdefault("chapter", "")
        meta.setdefault("section", "")
        meta.setdefault("content_type", "other")
        meta.setdefault("cs_domain", "other")
        meta.setdefault("programming_language", "")
        meta.setdefault("topics", [])
        meta.setdefault("summary", "")

        if meta["content_type"] not in CONTENT_TYPES:
            meta["content_type"] = "other"
        if meta["cs_domain"] not in CS_DOMAINS:
            meta["cs_domain"] = "other"
        if meta["programming_language"] not in PROGRAMMING_LANGUAGES:
            meta["programming_language"] = ""
        if not isinstance(meta["topics"], list):
            meta["topics"] = []

        return meta
