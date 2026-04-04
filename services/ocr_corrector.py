"""
Post-OCR text correction using LLM (GPT-4o-mini).

Fixes common Tesseract OCR errors in Vietnamese textbook text:
  - Sai dau tieng Viet (a->ă, o->ô, u->ư, sai thanh dieu)
  - Nham ky tu giong nhau (rn->m, l->1, O->0, cl->d)
  - Dinh / tach tu sai
  - Cong thuc toan bi hong

Only applied to pages where ``ocr_used=True``.  Text-extracted pages
are left untouched.
"""

import asyncio
import logging
from typing import List, Dict, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── System prompt — tightly scoped to OCR correction only ────────────
_SYSTEM_PROMPT = (
    "Bạn là công cụ sửa lỗi OCR cho sách giáo khoa Việt Nam.\n"
    "\n"
    "Text đầu vào được trích xuất bằng Tesseract OCR từ PDF scan, "
    "có thể chứa các lỗi sau:\n"
    "- Sai dấu tiếng Việt (ă↔â, ơ↔o, ư↔u, sai thanh điệu)\n"
    "- Nhầm ký tự giống nhau (rn↔m, l↔1, O↔0, cl↔d)\n"
    "- Dính hoặc tách từ sai\n"
    "- Công thức toán bị hỏng\n"
    "\n"
    "Quy tắc BẮT BUỘC:\n"
    "1. CHỈ sửa lỗi chính tả và dấu tiếng Việt.\n"
    "2. KHÔNG thay đổi nghĩa, KHÔNG thêm bớt nội dung.\n"
    "3. KHÔNG diễn giải, tóm tắt, hay giải thích.\n"
    "4. Giữ nguyên định dạng gốc (xuống dòng, khoảng trắng, bullet, số thứ tự).\n"
    "5. Giữ nguyên thuật ngữ tiếng Anh, tên riêng, code, và ký hiệu toán học.\n"
    "6. Nếu không chắc một từ có sai hay không → giữ nguyên bản gốc.\n"
    "7. Trả về CHỈ text đã sửa, không kèm giải thích hay ghi chú nào khác."
)


class OCRCorrector:
    """
    Correct OCR errors in Vietnamese text using a cheap LLM pass.

    Usage::

        corrector = OCRCorrector(api_key="sk-...")
        corrected_text, updated_pages = await corrector.correct_pages(
            full_text, page_info
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 5,
    ):
        """
        Args:
            api_key:        OpenAI API key.
            model:          Model to use for correction.
            max_concurrent: Max pages corrected in parallel.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"OCRCorrector initialized (model={model})")

    # ── Public API ────────────────────────────────────────────────────

    async def correct_pages(
        self,
        full_text: str,
        page_info: List[Dict],
    ) -> Tuple[str, List[Dict]]:
        """
        Correct OCR errors in pages that were scanned/OCR'd.

        - Pages where ``ocr_used=False`` are **not sent** to the LLM.
        - Corrected pages have their ``text``, ``char_start``, and
          ``char_end`` updated.
        - The full text is rebuilt from the (possibly corrected) pages.

        Returns:
            (corrected_full_text, updated_page_info)
        """
        ocr_indices = [
            i for i, p in enumerate(page_info)
            if p.get('ocr_used', False) and p.get('text', '').strip()
        ]

        if not ocr_indices:
            logger.info("No OCR pages to correct — skipping LLM pass")
            return full_text, page_info

        logger.info(
            f"Correcting {len(ocr_indices)} OCR page(s) with {self.model}..."
        )

        # ── Parallel correction with semaphore ──
        tasks = [
            self._correct_one(page_info[i]['text'])
            for i in ocr_indices
        ]
        corrected_texts = await asyncio.gather(*tasks)

        # ── Rebuild page_info with corrected text ──
        updated = [dict(p) for p in page_info]  # shallow copy each dict
        corrected_count = 0

        for idx, corrected in zip(ocr_indices, corrected_texts):
            original = updated[idx]['text']
            if corrected != original:
                corrected_count += 1
            updated[idx]['text'] = corrected

        # ── Recalculate char positions & rebuild full text ──
        all_page_texts = [p['text'] for p in updated]
        new_full_text = "\n\n".join(all_page_texts)

        pos = 0
        for p in updated:
            p['char_start'] = pos
            p['char_end'] = pos + len(p['text'])
            pos = p['char_end'] + 2  # +2 for the "\n\n" separator

        logger.info(
            f"OCR correction done: {corrected_count}/{len(ocr_indices)} "
            f"pages had changes"
        )
        return new_full_text, updated

    # ── Internal ──────────────────────────────────────────────────────

    async def _correct_one(self, text: str) -> str:
        """Correct a single page, with semaphore and error fallback."""
        async with self._semaphore:
            try:
                return await self._call_llm(text)
            except Exception as e:
                logger.warning(
                    f"OCR correction failed, keeping original: {e}"
                )
                return text  # fallback to uncorrected

    async def _call_llm(self, text: str) -> str:
        """Send text to the LLM for correction."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0,    # deterministic — pure correction, no creativity
            max_tokens=4096,  # a single page rarely exceeds this
        )
        result = response.choices[0].message.content
        if not result or not result.strip():
            logger.warning("LLM returned empty — keeping original")
            return text
        return result.strip()

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.close()
