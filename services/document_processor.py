"""
Document processor for extracting text from various file formats
"""
import logging
from typing import List, Dict, Tuple
import io
import re
from PIL import Image
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import settings
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)


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
            length_function=len
        )

    def extract_from_pdf_with_pages(self, file_content: bytes) -> Tuple[str, List[Dict]]:
        """
        Extract text from PDF and return text with page information
        Returns: (full_text, page_info_list)
        """
        try:
            pdf_stream = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            full_text = []
            page_info = []
            total_pages = len(doc)
            scanned_pages = 0
            text_pages = 0

            logger.info(f"Processing PDF with {total_pages} pages")

            for page_num in range(total_pages):
                page = doc[page_num]

                # Try to extract text directly
                text = page.get_text()

                # If no text, very little text, or garbled (encoding issue) → use OCR
                if len(text.strip()) < 50 or self._is_text_garbled(text):
                    if len(text.strip()) >= 50:
                        logger.info(f"Page {page_num + 1}/{total_pages}: Text garbled/wrong encoding, falling back to OCR")
                    else:
                        logger.info(f"Page {page_num + 1}/{total_pages}: Scanned page detected, using OCR")
                    scanned_pages += 1

                    # Convert page to image and OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    img_data = pix.tobytes("png")

                    # OCR the image
                    ocr_text = self.extract_from_image(img_data)
                    page_text = ocr_text
                    logger.info(f"Page {page_num + 1}/{total_pages}: OCR extracted {len(ocr_text)} chars")
                else:
                    text_pages += 1
                    page_text = text
                    logger.info(f"Page {page_num + 1}/{total_pages}: Text extracted {len(text)} chars")

                full_text.append(page_text)
                page_info.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_start": sum(len(t) + 2 for t in full_text[:-1]),  # +2 for \n\n
                    "char_end": sum(len(t) + 2 for t in full_text[:-1]) + len(page_text),
                    "ocr_used": len(text.strip()) < 50
                })

            doc.close()

            result = "\n\n".join(full_text)
            logger.info(f"PDF processed: {text_pages} text pages, {scanned_pages} scanned pages, total {len(result)} chars")
            
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

        suspicious = 0
        for ch in sample:
            cp = ord(ch)
            # Cyrillic block – the most common artifact from legacy Vietnamese fonts
            if 0x0400 <= cp <= 0x04FF:
                suspicious += 1
            # Cyrillic Supplement
            elif 0x0500 <= cp <= 0x052F:
                suspicious += 1
            # Private Use Area – some PDFs embed glyphs here
            elif 0xE000 <= cp <= 0xF8FF:
                suspicious += 1
            # Combining Diacritical Marks Supplement (unusual outside linguistics)
            elif 0x1DC0 <= cp <= 0x1DFF:
                suspicious += 1

        ratio = suspicious / len(sample)
        if ratio > 0.08:  # >8 % suspicious chars → garbled
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
                    length_function=len
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
                    length_function=len
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