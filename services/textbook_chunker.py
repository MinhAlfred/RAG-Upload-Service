"""
Textbook-aware semantic chunker for Vietnamese educational content (SGK).

Thay vi cat chunk co dinh theo so token (RecursiveCharacterTextSplitter),
module nay:

  1. Phat hien cau truc sach giao khoa (Chuong > Bai > Muc > Tieu muc)
  2. Nhan dien loai noi dung (Dinh nghia, Vi du, Bai tap, Thuc hanh, ...)
  3. Giu nguyen moi section la 1 chunk neu du nho
  4. Chi split tai ranh gioi cau / doan van khi section qua lon
  5. Gan heading context "[Chuong X > Bai Y > Muc Z]" de embedding co ngu canh
  6. Tra ve metadata day du cho downstream filtering

Designed for: SGK Tin hoc (Vietnamese Computer Science textbooks)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

import tiktoken

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """Count tokens using the embedding model's tokenizer (cl100k_base)."""
    return len(_TOKENIZER.encode(text))


# =====================================================================
# Heading detection
# =====================================================================
# Ordered from highest to lowest hierarchy level.
# Each rule: (compiled regex, level_name, depth_int)

_HEADING_RULES: List[Tuple[re.Pattern, str, int]] = [
    # ── Chuong (Chapter) — with and without diacritics (OCR fallback) ──
    (re.compile(
        r'^(?:CHƯƠNG|Chương|CHU\u01A0NG|CHUONG|Chuong)\s+(\d+|[IVXLC]+)[.:\-–\s]\s*(.+)',
    ), 'chapter', 1),

    # ── Bai (Lesson) / § — with and without diacritics ──
    (re.compile(
        r'^(?:BÀI|Bài|BAI|Bai)\s+(\d+)[.:\-–\s]\s*(.+)',
    ), 'lesson', 2),
    (re.compile(r'^§\s*(\d+)[.:\-–\s]\s*(.+)'), 'lesson', 2),

    # ── Muc: Roman numerals (I. II. III. IV. ...) ──
    (re.compile(r'^([IVXLC]{1,4})\.\s+(.{3,})'), 'section', 3),

    # ── Tieu muc: "1. Title..." (number + title >= 10 chars total) ──
    # Requires letter after the number to avoid matching list items like "1. x=5"
    (re.compile(
        r'^(\d{1,2})\.\s+([A-Za-z\u00C0-\u1EF9\u0110\u0111].{8,})'
    ), 'subsection', 4),
]


# =====================================================================
# Content-type detection
# =====================================================================

_CONTENT_TYPES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(?:Định nghĩa|Khái niệm)',              re.I), 'definition'),
    (re.compile(r'Ví dụ',                                  re.I), 'example'),
    (re.compile(r'(?:Bài tập|Luyện tập|Câu hỏi|Bài \d)',  re.I), 'exercise'),
    (re.compile(r'Thực hành',                               re.I), 'practice'),
    (re.compile(r'(?:Nhận xét|Chú ý|Lưu ý|Ghi nhớ)',      re.I), 'note'),
    (re.compile(r'Hoạt động',                               re.I), 'activity'),
    (re.compile(r'(?:Thuật toán|Giải thuật|Algorithm)',     re.I), 'algorithm'),
    (re.compile(r'(?:Hướng dẫn|Lời giải)',                 re.I), 'solution'),
    (re.compile(r'(?:Tóm tắt|Ôn tập)',                     re.I), 'summary'),
    (re.compile(r'(?:Đọc thêm|Tìm hiểu thêm|Em có biết)', re.I), 'supplementary'),
]


# =====================================================================
# Module-level helpers
# =====================================================================

def _detect_content_type(text: str) -> str:
    """Detect pedagogical content type from the first 300 chars."""
    sample = text[:300]
    for pattern, ctype in _CONTENT_TYPES:
        if pattern.search(sample):
            return ctype
    return 'explanation'


def _match_heading(line: str) -> Optional[Tuple[str, str, str, int]]:
    """
    Check whether *line* is a structural heading.

    Returns ``(level, number, title, depth)`` or ``None``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return None
    for pattern, level, depth in _HEADING_RULES:
        m = pattern.match(stripped)
        if m:
            return level, m.group(1), m.group(2).strip(), depth
    return None


def _build_path(ctx: Dict[str, str]) -> str:
    """Build a breadcrumb string like 'Chuong 1 > Bai 2 > I. Khai niem'."""
    parts = [v for v in (ctx.get('chapter'), ctx.get('lesson'),
                         ctx.get('section')) if v]
    return ' > '.join(parts)


def _find_breaks(text: str) -> List[int]:
    """
    Identify valid split positions in *text*, returned **sorted ascending**.

    A break at position ``p`` means ``text[:p]`` is one chunk candidate
    and ``text[p:]`` starts the next.

    Priority (implicit — the greedy algorithm picks the *latest* that fits):
      paragraph boundary  >  line break  >  sentence end  >  semicolon
    """
    breaks: set = set()

    # Paragraph boundaries (\n followed by optional whitespace and another \n)
    for m in re.finditer(r'\n\s*\n', text):
        breaks.add(m.end())

    # Single line breaks
    for m in re.finditer(r'\n', text):
        breaks.add(m.end())

    # Sentence-ending punctuation followed by whitespace
    for m in re.finditer(r'[.!?]\s', text):
        breaks.add(m.start() + 1)

    # Sentence-ending punctuation at end of text
    if text and text[-1] in '.!?':
        breaks.add(len(text))

    # Semicolons (common in lists / enumerations)
    for m in re.finditer(r';\s', text):
        breaks.add(m.start() + 1)

    return sorted(breaks)


def _make_chunk(
    text: str,
    raw_text: str,
    section: Dict,
    char_start: int,
    char_end: int,
) -> Dict:
    """Construct a chunk dict with all metadata fields."""
    return {
        'text': text,
        'raw_text': raw_text,
        'chapter': section['chapter'],
        'lesson': section['lesson'],
        'section': section['section'],
        'content_type': section['content_type'],
        'heading_path': section['heading_path'],
        'char_start': char_start,
        'char_end': char_end,
        'pages': [],  # populated by caller after all chunks are built
    }


# =====================================================================
# TextbookChunker
# =====================================================================

class TextbookChunker:
    """
    Semantic chunker for Vietnamese textbooks.

    Chunks by pedagogical structure (Chuong / Bai / Muc) with
    sentence-boundary safety, instead of fixed-size token windows.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 80,
    ):
        """
        Args:
            max_chunk_tokens: soft cap per chunk.  Should stay well under the
                              8 192-token embedding-model limit.
            min_chunk_tokens: chunks smaller than this are merged with their
                              predecessor to avoid noise.
        """
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens

    # ─── Public API ───────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        page_info: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Main entry point.

        Parameters
        ----------
        text : str
            Full extracted text of the textbook (pages joined by ``\\n\\n``).
        page_info : list[dict], optional
            Per-page ``{page_number, char_start, char_end, ...}`` from the
            PDF extractor.  Used to map each chunk to page numbers.

        Returns
        -------
        list[dict]
            Each dict contains:

            ======== ======================================================
            Key      Description
            ======== ======================================================
            text         chunk WITH heading prefix (for **embedding**)
            raw_text     chunk WITHOUT prefix (for **storage / display**)
            chapter      e.g. ``"Chuong 2: Phan mem may tinh"``
            lesson       e.g. ``"Bai 5: Ngon ngu lap trinh"``
            section      e.g. ``"I. Khai niem lap trinh"``
            content_type definition | example | exercise | practice | ...
            heading_path breadcrumb ``"Chuong 2 > Bai 5 > I. Khai niem"``
            pages        ``[3, 4]``  page numbers this chunk spans
            char_start   start offset in the original *text*
            char_end     end offset in the original *text*
            ======== ======================================================
        """
        # 1. Parse structural sections
        sections = self._parse_sections(text)

        # 2. Convert each section into one or more chunks
        chunks: List[Dict] = []
        for section in sections:
            chunks.extend(self._section_to_chunks(section))

        # 3. Drop garbage chunks (< 20 tokens)
        chunks = [c for c in chunks if _token_len(c['raw_text']) >= 20]

        # 4. Map chunks to page numbers
        if page_info:
            for c in chunks:
                c['pages'] = self._find_pages(
                    c['char_start'], c['char_end'], page_info,
                )

        logger.info(
            f"TextbookChunker: {len(sections)} sections -> {len(chunks)} chunks "
            f"(max {self.max_tokens} tok)"
        )
        return chunks

    # ─── Section parsing ──────────────────────────────────────────────

    def _parse_sections(self, text: str) -> List[Dict]:
        """
        Walk through lines, detect headings, and group consecutive lines
        into hierarchical sections.
        """
        lines = text.split('\n')
        sections: List[Dict] = []

        # Running hierarchy context
        ctx: Dict[str, str] = {'chapter': '', 'lesson': '', 'section': ''}

        # Accumulator for current section
        buf_lines: List[str] = []
        buf_start: int = 0
        buf_heading: str = ''
        buf_level: str = 'content'

        char_pos = 0

        for line in lines:
            stripped = line.strip()
            heading = _match_heading(stripped)

            if heading:
                # ── flush previous section ──
                self._flush_section(
                    buf_lines, buf_start, char_pos,
                    buf_heading, buf_level, ctx, sections,
                )

                # ── update hierarchy ──
                level, _number, _title, _depth = heading
                if level == 'chapter':
                    ctx = {'chapter': stripped, 'lesson': '', 'section': ''}
                elif level == 'lesson':
                    ctx['lesson'] = stripped
                    ctx['section'] = ''
                elif level in ('section', 'subsection'):
                    ctx['section'] = stripped

                # ── start new section ──
                buf_lines = [stripped]
                buf_start = char_pos
                buf_heading = stripped
                buf_level = level
            else:
                buf_lines.append(line)

            char_pos += len(line) + 1  # +1 accounts for the '\n'

        # Flush the last section
        self._flush_section(
            buf_lines, buf_start, char_pos,
            buf_heading, buf_level, ctx, sections,
        )
        return sections

    @staticmethod
    def _flush_section(
        buf_lines: List[str],
        buf_start: int,
        char_pos: int,
        heading: str,
        level: str,
        ctx: Dict[str, str],
        out: List[Dict],
    ) -> None:
        """Flush buffered lines into a section dict and append to *out*."""
        if not buf_lines:
            return
        content = '\n'.join(buf_lines).strip()
        if not content:
            return
        out.append({
            'content': content,
            'heading': heading,
            'level': level,
            'chapter': ctx['chapter'],
            'lesson': ctx['lesson'],
            'section': ctx['section'],
            'content_type': _detect_content_type(content),
            'heading_path': _build_path(ctx),
            'char_start': buf_start,
            'char_end': char_pos,
        })

    # ─── Section -> chunk(s) ──────────────────────────────────────────

    def _section_to_chunks(self, section: Dict) -> List[Dict]:
        """Turn one structural section into one or more chunks."""
        content = section['content']
        prefix = (
            f"[{section['heading_path']}]\n"
            if section['heading_path'] else ''
        )

        # Happy path: section fits in a single chunk
        if _token_len(prefix + content) <= self.max_tokens:
            return [_make_chunk(
                prefix + content, content, section,
                section['char_start'], section['char_end'],
            )]

        # Section is oversized -> split at natural boundaries
        return self._split_oversized(section, prefix)

    def _split_oversized(self, section: Dict, prefix: str) -> List[Dict]:
        """
        Split an oversized section at paragraph / sentence boundaries.

        Algorithm:
        - Find all valid break positions.
        - Greedily take the *latest* break that keeps the chunk under
          ``max_tokens``.
        - If nothing fits, take the next break regardless (oversized chunk
          is better than an infinite loop).
        - After building all chunks, merge a tiny trailing chunk back into
          its predecessor if possible.
        """
        content = section['content']
        breaks = _find_breaks(content)

        # Ensure the end of the text is always a candidate break
        if not breaks or breaks[-1] < len(content):
            breaks.append(len(content))

        chunks: List[Dict] = []
        start = 0

        while start < len(content):
            # Greedy: find the furthest break that fits within max_tokens
            best: Optional[int] = None
            for bp in breaks:
                if bp <= start:
                    continue
                candidate = content[start:bp].strip()
                if not candidate:
                    continue
                if _token_len(prefix + candidate) <= self.max_tokens:
                    best = bp
                else:
                    break  # breaks are sorted — no point continuing

            if best is None:
                # Nothing fits — take the very next break to avoid looping
                for bp in breaks:
                    if bp > start:
                        best = bp
                        break
                if best is None:
                    best = len(content)

            raw = content[start:best].strip()
            if raw:
                chunks.append(_make_chunk(
                    prefix + raw, raw, section,
                    section['char_start'] + start,
                    section['char_start'] + best,
                ))

            start = best
            # Skip inter-chunk whitespace so next chunk starts cleanly
            while start < len(content) and content[start] in ' \t\n\r':
                start += 1

        # Merge a tiny trailing chunk into its predecessor
        if len(chunks) > 1 and _token_len(chunks[-1]['raw_text']) < self.min_tokens:
            last = chunks.pop()
            prev = chunks[-1]
            merged_raw = prev['raw_text'] + '\n' + last['raw_text']
            if _token_len(prefix + merged_raw) <= int(self.max_tokens * 1.15):
                chunks[-1] = _make_chunk(
                    prefix + merged_raw, merged_raw, section,
                    prev['char_start'], last['char_end'],
                )
            else:
                chunks.append(last)  # can't merge — keep separate

        return chunks

    # ─── Page mapping ─────────────────────────────────────────────────

    @staticmethod
    def _find_pages(
        start: int, end: int, page_info: List[Dict],
    ) -> List[int]:
        """Map a character range to the page numbers it overlaps with."""
        pages = []
        for p in page_info:
            if start < p.get('char_end', 0) and end > p.get('char_start', 0):
                pages.append(p['page_number'])
        return sorted(set(pages))
