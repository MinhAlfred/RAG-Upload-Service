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
    # ── Chuong / Chu de (Chapter / Topic) — with and without diacritics ──
    (re.compile(
        r'^(?:CHƯƠNG|Chương|CHU\u01A0NG|CHUONG|Chuong)\s+(\d+|[IVXLC]+)[.:\-–\s]\s*(.+)',
    ), 'chapter', 1),
    (re.compile(
        r'^(?:CHỦ ĐỀ|Chủ đề|CHU DE|Chu de)\s+(\d+|[IVXLC]+)[.:\-–\s]\s*(.+)',
        re.IGNORECASE,
    ), 'chapter', 1),

    # ── Bai (Lesson) / § — with and without diacritics ──
    (re.compile(
        r'^(?:BÀI|Bài|BAI|Bai)\s+(\d+[a-zA-Z]?)[.:\-–\s]\s*(.+)',
    ), 'lesson', 2),
    (re.compile(r'^§\s*(\d+)[.:\-–\s]\s*(.+)'), 'lesson', 2),

    # section / subsection removed — Roman numerals and numbered headings
    # cause too many false positives with OCR'd Vietnamese textbook content.
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

        # ── Debug: log all final chunks ──
        for idx, c in enumerate(chunks):
            tok = _token_len(c['text'])
            logger.debug(
                f"  chunk[{idx}] tokens={tok}, "
                f"type={c['content_type']}, "
                f"path={c['heading_path']!r}, "
                f"pages={c.get('pages', [])}, "
                f"preview={c['raw_text'][:80]!r}"
            )

        return chunks

    # ─── TOC detection ───────────────────────────────────────────────

    @staticmethod
    def _find_toc_end_line(lines: List[str]) -> int:
        """
        Detect a Table of Contents region and return the line index of
        its last line, or ``-1`` if no TOC is found.

        A TOC is a dense cluster of heading matches packed into a small
        region of the text.  Body headings, by contrast, are separated
        by pages of content.

        Algorithm:
          1. Collect all line indices where ``_match_heading`` fires.
          2. Walk through them looking for the first large gap (> 30 lines).
             Everything before the gap with ≥ 4 headings is the TOC.
          3. Fallback: if ALL headings sit in < 10% of total lines, the
             whole set is treated as a TOC.
        """
        heading_lines = [
            i for i, line in enumerate(lines)
            if _match_heading(line.strip())
        ]

        if len(heading_lines) < 4:
            return -1

        # Find first big gap — headings before it are the TOC
        for j in range(1, len(heading_lines)):
            gap = heading_lines[j] - heading_lines[j - 1]
            if gap > 30 and j >= 4:
                return heading_lines[j - 1]

        # Fallback: all headings in a tiny region → entire cluster is TOC
        total_lines = len(lines)
        span = heading_lines[-1] - heading_lines[0] + 1
        if span < total_lines * 0.10:
            return heading_lines[-1]

        return -1

    # ─── Section parsing ──────────────────────────────────────────────

    def _parse_sections(self, text: str) -> List[Dict]:
        """
        Walk through lines, detect headings, and group consecutive lines
        into hierarchical sections.

        Character positions are tracked by scanning through the original
        *text* string so that they match the ``char_start`` / ``char_end``
        values produced by the PDF extractor (which joins pages with
        ``\\n\\n``).
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

        # Track real character offset in the original text.
        # After split('\n'), the offset of line[i] in the original text is
        # sum(len(line[0..i-1])) + i   (each split consumed one '\n').
        char_pos = 0

        for i, line in enumerate(lines):
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
                logger.info(
                    f"[HeadingDetect] level={level}, number={_number}, "
                    f"title={_title!r}, line={i+1}"
                )
                if level == 'chapter':
                    ctx = {'chapter': stripped, 'lesson': '', 'section': ''}
                elif level == 'lesson':
                    ctx['lesson'] = stripped
                    ctx['section'] = ''

                # ── start new section ──
                buf_lines = [stripped]
                buf_start = char_pos
                buf_heading = stripped
                buf_level = level
            else:
                buf_lines.append(line)

            # Advance by the length of this line + 1 for the '\n' that
            # split() consumed.  This keeps char_pos aligned with the
            # original text even when pages are joined by '\n\n'.
            char_pos += len(line)
            if i < len(lines) - 1:
                char_pos += 1  # the '\n' between this line and the next

        # Flush the last section
        self._flush_section(
            buf_lines, buf_start, char_pos,
            buf_heading, buf_level, ctx, sections,
        )

        # ── Debug: log lines that ALMOST look like headings but didn't match ──
        _NEAR_MISS = re.compile(
            r'(?:chương|chủ\s*đề|bài|chu\s*de|chuong)\s+\d',
            re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            stripped = line.strip()
            if _NEAR_MISS.search(stripped) and not _match_heading(stripped):
                logger.debug(
                    f"[NearMiss] line={i+1}, len={len(stripped)}, "
                    f"text={stripped[:150]!r}"
                )

        # ── Debug: log all detected sections ──
        logger.info(f"[SectionSummary] Total sections detected: {len(sections)}")
        for idx, sec in enumerate(sections):
            logger.info(
                f"  [{idx}] level={sec['level']}, "
                f"content_type={sec['content_type']}, "
                f"chapter={sec['chapter']!r}, "
                f"lesson={sec['lesson']!r}, "
                f"heading={sec['heading']!r}, "
                f"path={sec['heading_path']!r}, "
                f"chars={sec['char_start']}-{sec['char_end']} "
                f"(~{sec['char_end'] - sec['char_start']} chars)"
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

    # ─── TOC-aware rebuild ───────────────────────────────────────────

    _PAGE_NUM_RE = re.compile(r'^(.+?)\s+(\d+)\s*$')

    def _rebuild_from_toc(
        self,
        sections: List[Dict],
        text: str,
        page_info: List[Dict],
    ) -> List[Dict]:
        """
        Detect if headings came from the Table of Contents instead of
        actual body headings.  If so, use TOC page numbers + page_info
        to create properly mapped sections.

        The offset between book page numbers and PDF page numbers is
        auto-detected by searching for a TOC entry title in the body text.
        """
        if not page_info:
            return sections

        heading_secs = [s for s in sections if s['level'] != 'content']
        if len(heading_secs) < 3:
            return sections

        # ── Detect TOC pattern ──────────────────────────────────────
        sizes = sorted(s['char_end'] - s['char_start'] for s in heading_secs)
        median_size = sizes[len(sizes) // 2]
        max_size = sizes[-1]

        if median_size >= 200 or max_size < 5000:
            return sections  # sections have real content → not TOC

        # ── Extract TOC entries ─────────────────────────────────────
        toc_entries: List[Dict] = []
        for s in heading_secs:
            m = self._PAGE_NUM_RE.match(s['heading'])
            if m:
                toc_entries.append({
                    'heading': m.group(1).strip(),
                    'page': int(m.group(2)),
                    'level': s['level'],
                })
            else:
                toc_entries.append({
                    'heading': s['heading'].strip(),
                    'page': None,
                    'level': s['level'],
                })

        with_pages = [e for e in toc_entries if e['page'] is not None]
        if len(with_pages) < len(toc_entries) * 0.5:
            return sections

        # ── Auto-detect page offset (book page → PDF page) ─────────
        # Strategy: search for each TOC entry title in the body text,
        # find which PDF page the match is on, compute offset.
        toc_body_start = max(s['char_end'] for s in heading_secs)
        page_starts = {p['page_number']: p['char_start'] for p in page_info}
        page_ends = {p['page_number']: p.get('char_end', len(text)) for p in page_info}

        offset: Optional[int] = None
        for entry in with_pages:
            pos = text.find(entry['heading'], int(toc_body_start))
            if pos < 0:
                continue
            for p in page_info:
                if p['char_start'] <= pos < p.get('char_end', len(text)):
                    offset = p['page_number'] - entry['page']
                    logger.info(
                        f"[TOC Offset] Found '{entry['heading'][:50]}' "
                        f"on PDF page {p['page_number']} "
                        f"(TOC says page {entry['page']}). "
                        f"Offset = {offset}"
                    )
                    break
            if offset is not None:
                break

        if offset is None:
            # Fallback: first PDF page after TOC = first TOC page
            min_toc_page = min(e['page'] for e in with_pages)
            for p in sorted(page_info, key=lambda x: x['char_start']):
                if p['char_start'] >= toc_body_start:
                    offset = p['page_number'] - min_toc_page
                    logger.info(
                        f"[TOC Offset] Estimated offset = {offset} "
                        f"(PDF page {p['page_number']} ≈ book page {min_toc_page})"
                    )
                    break
            if offset is None:
                offset = 0

        logger.info(
            f"[TOC Detected] {len(toc_entries)} entries, offset={offset}"
        )

        # ── Build new sections ──────────────────────────────────────
        ctx: Dict[str, str] = {'chapter': '', 'lesson': '', 'section': ''}
        new_sections: List[Dict] = []

        for i, entry in enumerate(toc_entries):
            # Always update hierarchy context
            if entry['level'] == 'chapter':
                ctx = {'chapter': entry['heading'], 'lesson': '', 'section': ''}
            elif entry['level'] == 'lesson':
                ctx['lesson'] = entry['heading']
                ctx['section'] = ''

            if entry['page'] is None:
                continue

            pdf_page = entry['page'] + offset
            if pdf_page not in page_starts:
                continue

            start = page_starts[pdf_page]

            # End = char_start of next entry with a later PDF page
            end = len(text)
            for j in range(i + 1, len(toc_entries)):
                np = toc_entries[j].get('page')
                if np is not None:
                    next_pdf = np + offset
                    if next_pdf in page_starts and page_starts[next_pdf] > start:
                        end = page_starts[next_pdf]
                        break

            # Chapter shares page with first lesson → skip chapter section
            if entry['level'] == 'chapter' and i + 1 < len(toc_entries):
                next_page = toc_entries[i + 1].get('page')
                if next_page == entry['page']:
                    continue

            content = text[start:end].strip()
            if not content:
                continue

            new_sections.append({
                'content': content,
                'heading': entry['heading'],
                'level': entry['level'],
                'chapter': ctx['chapter'],
                'lesson': ctx['lesson'],
                'section': ctx['section'],
                'content_type': _detect_content_type(content),
                'heading_path': _build_path(ctx),
                'char_start': start,
                'char_end': end,
            })

        if not new_sections:
            logger.warning(
                "[TOC Rebuild] No sections created — keeping original."
            )
            return sections

        logger.info(
            f"[TOC Rebuild] {len(new_sections)} sections from TOC"
        )
        for idx, sec in enumerate(new_sections):
            logger.info(
                f"  [{idx}] level={sec['level']}, "
                f"chapter={sec['chapter']!r}, "
                f"lesson={sec['lesson']!r}, "
                f"path={sec['heading_path']!r}, "
                f"chars={sec['char_start']}-{sec['char_end']} "
                f"(~{sec['char_end'] - sec['char_start']} chars)"
            )

        return new_sections

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
