"""
Document processor for extracting text from various file formats
"""
import logging
from typing import List, Dict, Tuple
import io
import re
from concurrent.futures import ThreadPoolExecutor  # kept for embedding_service
from PIL import Image
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from config.config import settings
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)

# Use the same tokenizer as the embedding model to measure chunk size in tokens
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_TOKENIZER.encode(text))


class DocumentProcessor:
    """Process documents and extract text"""

    def __init__(self):
        """Initialize document processor"""
        # Initialize OCR service (Tesseract)
        self.ocr = OCRService()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=_token_len
        )

    def extract_from_pdf_with_pages(self, file_content: bytes) -> Tuple[str, List[Dict]]:
        """
        Extract text from PDF and return text with page information.

        Pass 1 (single thread): iterate pages with PyMuPDF to collect direct text
                                or rendered image bytes – fitz.Document is NOT thread-safe.
        Pass 2 (thread pool):   run Tesseract OCR on all scanned pages in parallel.

        Returns: (full_text, page_info_list)
        """
        try:
            pdf_stream = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            total_pages = len(doc)
            logger.info(f"Processing PDF with {total_pages} pages")

            # ------------------------------------------------------------------
            # Pass 1: collect page data without OCR (single thread – fitz safety)
            # ------------------------------------------------------------------
            # OCR every page — Vietnamese PDFs with legacy fonts (VNI/TCVN3)
            # produce garbled text from direct extraction, so OCR all pages
            # for consistent quality. GPT-4o-mini corrects OCR errors afterward.
            page_data: List[Dict] = []
            for page_num in range(total_pages):
                page = doc[page_num]
                logger.info(f"Page {page_num + 1}/{total_pages}: rendering for OCR")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")

                page_data.append({
                    "page_num": page_num,
                    "direct_text": "",
                    "img_bytes": img_bytes,
                    "needs_ocr": True,
                })

            doc.close()

            # ------------------------------------------------------------------
            # Pass 2: Sequential OCR — process pages in order to guarantee
            # correct page-to-text mapping.  Parallel OCR with as_completed
            # caused page content to be associated with wrong page numbers.
            # ------------------------------------------------------------------
            ocr_results: Dict[int, str] = {}
            logger.info(f"Running OCR on {total_pages} pages sequentially")

            for i, p in enumerate(page_data):
                try:
                    ocr_results[i] = self.extract_from_image(p["img_bytes"])
                    logger.info(
                        f"Page {p['page_num'] + 1}/{total_pages}: "
                        f"OCR extracted {len(ocr_results[i])} chars"
                    )
                except Exception as exc:
                    logger.error(f"OCR failed for page {p['page_num'] + 1}: {exc}")
                    ocr_results[i] = ""

            # ------------------------------------------------------------------
            # Assemble results in original page order
            # ------------------------------------------------------------------
            full_text = []
            page_info = []

            for i, p in enumerate(page_data):
                page_text = ocr_results[i]
                full_text.append(page_text)
                page_info.append({
                    "page_number": p["page_num"] + 1,
                    "text": page_text,
                    "char_start": sum(len(t) + 2 for t in full_text[:-1]),
                    "char_end": sum(len(t) + 2 for t in full_text[:-1]) + len(page_text),
                    "ocr_used": True,
                })

            result = "\n\n".join(full_text)
            logger.info(
                f"PDF processed: {total_pages} pages (all OCR), "
                f"total {len(result)} chars"
            )
            return result, page_info

        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}", exc_info=True)
            raise

    def extract_from_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF (handles both text and scanned PDFs)
        Backward compatibility method
        """
        text, _ = self.extract_from_pdf_with_pages(file_content)
        return text

    def extract_from_image(self, image_content: bytes) -> str:
        """
        Extract text from image using Tesseract OCR
        Better for code screenshots and multilingual text
        """
        try:
            # Use OCR service with automatic language detection
            text = self.ocr.extract_text(
                image_content=image_content,
                enhance=True,  # Enhance image for better OCR
                language='auto'  # Auto-detect language
            )

            # Clean up OCR artifacts
            text = self._clean_ocr_text(text)

            return text

        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            raise

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up common OCR errors"""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]

        # Reconstruct text
        cleaned = '\n'.join(lines)

        return cleaned

    def _is_text_garbled(self, text: str) -> bool:
        """
        Detect whether PDF-extracted text is garbled due to wrong font encoding.

        Common cause: Vietnamese PDFs using legacy encodings (VNI, TCVN3/ABC,
        Windows-1258) whose font glyph maps are not Unicode-compliant.
        PyMuPDF reads the raw code points, which land in Cyrillic or other
        non-Latin ranges instead of proper Unicode Vietnamese.

        Returns True when the text should be discarded and OCR used instead.
        """
        if not text:
            return True

        # Work on a condensed sample (skip whitespace/newlines)
        sample = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(sample) == 0:
            return True

        # Vietnamese Unicode lives in:
        #   Basic Latin        0x0000-0x007F
        #   Latin Extended-A   0x0100-0x017F  (Ă, ă, Đ, đ)
        #   Latin Extended-B   Ơ(01A0), ơ(01A1), Ư(01AF), ư(01B0) ONLY
        #   Latin Ext Additional 0x1E00-0x1EFF (ắ, ề, ỏ, ...)
        #   Combining Diacritics 0x0300-0x036F (tone marks)
        #
        # Anything else in a "Vietnamese" PDF is garbled.
        _VIET_EXTENDED_B = {0x01A0, 0x01A1, 0x01AF, 0x01B0}

        suspicious = 0
        for ch in sample:
            cp = ord(ch)
            # Cyrillic block (most common garbled artifact)
            if 0x0400 <= cp <= 0x04FF:
                suspicious += 1
            # Cyrillic Supplement
            elif 0x0500 <= cp <= 0x052F:
                suspicious += 1
            # Private Use Area
            elif 0xE000 <= cp <= 0xF8FF:
                suspicious += 1
            # Combining Diacritical Marks Supplement
            elif 0x1DC0 <= cp <= 0x1DFF:
                suspicious += 1
            # IPA Extensions (ɑ, ɨ, ɥ, ...) — never in Vietnamese
            elif 0x0250 <= cp <= 0x02AF:
                suspicious += 1
            # Spacing Modifier Letters (ʰ, ʲ, ...) — never in Vietnamese
            elif 0x02B0 <= cp <= 0x02FF:
                suspicious += 1
            # Latin Extended-B — except Ơ/ơ/Ư/ư which ARE Vietnamese
            elif 0x0180 <= cp <= 0x024F and cp not in _VIET_EXTENDED_B:
                suspicious += 1

        ratio = suspicious / len(sample)
        if ratio > 0.05:  # >5% suspicious chars → garbled (lowered from 8%)
            logger.debug(
                f"Garbled text detected: {suspicious}/{len(sample)} suspicious chars "
                f"({ratio:.1%}), sample={sample[:60]!r}"
            )
            return True
        return False

    def _find_chunk_pages(self, chunk_start: int, chunk_end: int, page_info: List[Dict]) -> List[int]:
        """
        Find which pages a chunk spans across
        """
        chunk_pages = []
        for page in page_info:
            # Check if chunk overlaps with page
            if (chunk_start < page["char_end"] and chunk_end > page["char_start"]):
                chunk_pages.append(page["page_number"])
        return sorted(list(set(chunk_pages)))

    def chunk_text_with_pages(
        self,
        text: str,
        page_info: List[Dict],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict]:
        """
        Split text into chunks with page information
        Returns list of dicts with chunk text and page numbers
        """
        try:
            if chunk_size or chunk_overlap:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size or settings.CHUNK_SIZE,
                    chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=_token_len
                )
            else:
                splitter = self.text_splitter

            # Get chunks with their positions
            chunks = splitter.split_text(text)
            
            # Calculate character positions for each chunk
            chunks_with_pages = []
            current_pos = 0
            
            for chunk in chunks:
                if len(chunk.strip()) <= 20:  # Skip very short chunks
                    current_pos = text.find(chunk, current_pos) + len(chunk)
                    continue
                    
                chunk_start = text.find(chunk, current_pos)
                chunk_end = chunk_start + len(chunk)
                
                # Find which pages this chunk spans
                chunk_pages = self._find_chunk_pages(chunk_start, chunk_end, page_info)
                
                chunks_with_pages.append({
                    "text": chunk,
                    "pages": chunk_pages,
                    "char_start": chunk_start,
                    "char_end": chunk_end
                })
                
                current_pos = chunk_end

            return chunks_with_pages

        except Exception as e:
            logger.error(f"Error chunking text with pages: {e}")
            raise

    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks with overlap
        """
        try:
            if chunk_size or chunk_overlap:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size or settings.CHUNK_SIZE,
                    chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=_token_len
                )
            else:
                splitter = self.text_splitter

            chunks = splitter.split_text(text)

            # Filter out very short chunks
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]

            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise

    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from markdown or text
        Useful for processing documentation with code examples
        """
        import re

        # Pattern for markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                'language': lang or 'unknown',
                'code': code.strip()
            })

        return code_blocks

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Vietnamese or English
        """
        try:
            from langdetect import detect
            lang = detect(text)
            return lang
        except:
            return "unknown"