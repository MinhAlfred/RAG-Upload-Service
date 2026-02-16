"""
Embedding service - Simplified to use only OpenAI
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from config.config import settings
from services.document_processor import DocumentProcessor
from services.embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for document processing and embedding generation"""

    def __init__(self):
        """Initialize embedding service with OpenAI"""
        self.document_processor = DocumentProcessor()
        self.embedder = OpenAIEmbedder(api_key=settings.OPENAI_API_KEY)
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("Initialized EmbeddingService with OpenAI text-embedding-3-small")

    async def process_textbook_document(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        book_name: str,
        publisher: str,
        grade: Optional[str] = None,
        product_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a textbook document with enhanced metadata including page numbers
        
        Returns:
            Dict containing chunks, embeddings, and metadata with textbook info
        """
        try:
            # Validate file type
            if file_type not in settings.SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Validate file size
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
                )

            logger.info(f"Processing textbook {filename} ({file_type}, {file_size_mb:.2f}MB)")

            # Create book metadata from API parameters
            book_metadata = {
                "book_name": book_name.strip(),
                "publisher": publisher.strip(),
                "grade": grade.strip() if grade else None,
                "full_name": f"{book_name.strip()} - {publisher.strip()}" + (f" - {grade.strip()}" if grade else "")
            }

            # Extract text with page information (only for PDF currently)
            if file_type == "application/pdf":
                extracted_text, page_info = await self._extract_text_with_pages(
                    file_content=file_content,
                    filename=filename,
                    file_type=file_type
                )

                if not extracted_text or not extracted_text.strip():
                    raise ValueError("No text could be extracted from the document")

                logger.info(f"Extracted {len(extracted_text)} characters from {len(page_info)} pages")

                # Chunk the text with page information
                chunks_with_pages = self.document_processor.chunk_text_with_pages(
                    text=extracted_text,
                    page_info=page_info,
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP
                )

                logger.info(f"Created {len(chunks_with_pages)} chunks with page information")

                # Extract just the text for embedding
                chunks = [chunk["text"] for chunk in chunks_with_pages]

                # Generate embeddings using OpenAI
                embeddings = await self.embed_batch(chunks)

                # Prepare enhanced metadata for each chunk
                file_hash = hashlib.md5(file_content).hexdigest()
                metadata_list = []

                for idx, chunk_info in enumerate(chunks_with_pages):
                    meta = {
                        "filename": filename,
                        "file_type": file_type,
                        "file_hash": file_hash,
                        "chunk_index": idx,
                        "total_chunks": len(chunks_with_pages),
                        # Product name from API input
                        "product_name": product_name or book_metadata["full_name"],
                        # Textbook specific metadata
                        "book_name": book_metadata["book_name"],
                        "publisher": book_metadata["publisher"],
                        "grade": book_metadata["grade"],
                        "book_full_name": book_metadata["full_name"],
                        # Page information
                        "pages": chunk_info["pages"],
                        "page_range": f"{min(chunk_info['pages'])}-{max(chunk_info['pages'])}" if len(chunk_info['pages']) > 1 else f"Trang {chunk_info['pages'][0]}" if chunk_info['pages'] else "",
                        "char_start": chunk_info["char_start"],
                        "char_end": chunk_info["char_end"]
                    }
                    metadata_list.append(meta)

                return {
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "metadata": metadata_list,
                    "original_text": extracted_text,
                    "page_info": page_info,
                    "book_metadata": book_metadata
                }

            else:
                # For non-PDF files, fall back to regular processing
                logger.info(f"Non-PDF file, using regular processing for {filename}")
                # Prepare additional metadata for non-PDF files
                additional_meta = {
                    "book_metadata": book_metadata,
                    "product_name": product_name or book_metadata["full_name"]
                }
                return await self.process_document(
                    file_content=file_content,
                    filename=filename,
                    file_type=file_type,
                    additional_metadata=f'{additional_meta}'
                )

        except Exception as e:
            logger.error(f"Error processing textbook document: {e}")
            raise

    async def _extract_text_with_pages(
        self,
        file_content: bytes,
        filename: str,
        file_type: str
    ) -> tuple[str, List[Dict]]:
        """Extract text from PDF with page information"""
        try:
            # Run extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            if file_type == "application/pdf":
                text, page_info = await loop.run_in_executor(
                    self.executor,
                    self.document_processor.extract_from_pdf_with_pages,
                    file_content
                )
                return text, page_info
            else:
                # For other file types, simulate single page
                text = await self._extract_text(file_content, filename, file_type)
                page_info = [{
                    "page_number": 1,
                    "text": text,
                    "char_start": 0,
                    "char_end": len(text),
                    "ocr_used": False
                }]
                return text, page_info

        except Exception as e:
            logger.error(f"Error extracting text with pages: {e}")
            raise

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        additional_metadata: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a regular document and generate embeddings (backward compatibility)

        Returns:
            Dict containing chunks, embeddings, and metadata
        """
        try:
            # Validate file type
            if file_type not in settings.SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Validate file size
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
                )

            logger.info(f"Processing {filename} ({file_type}, {file_size_mb:.2f}MB)")

            # Extract text from document
            extracted_text = await self._extract_text(
                file_content=file_content,
                filename=filename,
                file_type=file_type
            )

            if not extracted_text or not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")

            logger.info(f"Extracted {len(extracted_text)} characters")

            # Chunk the text
            chunks = self.document_processor.chunk_text(
                text=extracted_text,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            logger.info(f"Created {len(chunks)} chunks")

            # Generate embeddings using OpenAI
            embeddings = await self.embed_batch(chunks)

            # Prepare metadata for each chunk
            file_hash = hashlib.md5(file_content).hexdigest()
            metadata_list = []

            for idx in range(len(chunks)):
                meta = {
                    "filename": filename,
                    "file_type": file_type,
                    "file_hash": file_hash,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }

                if additional_metadata:
                    try:
                        import json
                        extra = json.loads(additional_metadata)
                        meta.update(extra)
                    except:
                        meta["raw_metadata"] = additional_metadata

                metadata_list.append(meta)

            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "metadata": metadata_list,
                "original_text": extracted_text
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    async def _extract_text(
        self,
        file_content: bytes,
        filename: str,
        file_type: str
    ) -> str:
        """Extract text from different file types"""
        try:
            # Run extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            if file_type == "application/pdf":
                text = await loop.run_in_executor(
                    self.executor,
                    self.document_processor.extract_from_pdf,
                    file_content
                )
            elif file_type.startswith("image/"):
                text = await loop.run_in_executor(
                    self.executor,
                    self.document_processor.extract_from_image,
                    file_content
                )
            elif file_type in ["text/plain", "text/markdown", "text/x-python"]:
                text = file_content.decode('utf-8', errors='ignore')
            elif file_type == "application/json":
                import json
                data = json.loads(file_content.decode('utf-8'))
                text = json.dumps(data, indent=2, ensure_ascii=False)
            else:
                # Try to decode as text
                text = file_content.decode('utf-8', errors='ignore')

            return text

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions)
        """
        try:
            return await self.embedder.embed_single(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (1536 dimensions each)
        """
        try:
            # OpenAI handles batching internally
            embeddings = await self.embedder.embed(texts)

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)