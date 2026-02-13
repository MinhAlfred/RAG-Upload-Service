"""
Main FastAPI application for document embedding service
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

from config.config import settings
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService
from model.schemas import (
    DocumentResponse,
    ExtractResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
embedding_service: Optional[EmbeddingService] = None
qdrant_service: Optional[QdrantService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global embedding_service, qdrant_service

    logger.info("Initializing services...")
    try:
        # Initialize services
        embedding_service = EmbeddingService()

        # Initialize Qdrant service (Cloud only)
        qdrant_service = QdrantService(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

        # Initialize collection
        await qdrant_service.init_collection(
            collection_name=settings.COLLECTION_NAME,
            vector_size=settings.EMBEDDING_DIMENSION
        )

        logger.info("Services initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        if qdrant_service:
            await qdrant_service.close()


app = FastAPI(
    title="Document Embedding Service",
    description="Service for embedding documents and storing in Qdrant for chatbot RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        qdrant_healthy = await qdrant_service.health_check()
        return HealthResponse(
            status="healthy" if qdrant_healthy else "degraded",
            qdrant_connected=qdrant_healthy
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", qdrant_connected=False)


@app.post("/extract", response_model=ExtractResponse)
async def extract_text(
        file: UploadFile = File(...),
):
    """
    Extract text from image/PDF/document (for chat user)
    Does NOT save to vector database - just returns extracted text
    
    Use case: Spring backend sends user's uploaded image to extract text for chat context
    """
    try:
        logger.info(f"Extracting text from: {file.filename} (type: {file.content_type})")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        # Read file contentl
        content = await file.read()

        # Validate file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
            )

        # Extract text only (no embedding, no storage)
        extracted_text = await embedding_service._extract_text(
            file_content=content,
            filename=file.filename,
            file_type=file.content_type
        )

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )

        logger.info(f"Extracted {len(extracted_text)} characters from {file.filename}")

        return ExtractResponse(
            text=extracted_text,
            filename=file.filename,
            file_type=file.content_type,
            char_count=len(extracted_text),
            status="success",
            message=f"Successfully extracted {len(extracted_text)} characters"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")


@app.post("/upload-document", response_model=DocumentResponse)
async def upload_document(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        metadata: Optional[str] = None
):
    """
    Upload and process a document for RAG knowledge base (Admin only)
    Extracts text → Generates embeddings → Stores in vector database
    
    Supports: PDF (scanned/text), images (code screenshots), text files
    
    Use case: Admin uploads documentation to build RAG knowledge base
    """
    try:
        logger.info(f"Processing file: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        # Read file content
        content = await file.read()

        # Process document based on type
        result = await embedding_service.process_document(
            file_content=content,
            filename=file.filename,
            file_type=file.content_type,
            additional_metadata=metadata
        )

        # Store in Qdrant
        doc_id = await qdrant_service.upsert_documents(
            collection_name=settings.COLLECTION_NAME,
            documents=result["chunks"],
            embeddings=result["embeddings"],
            metadata=result["metadata"]
        )

        logger.info(f"Document processed successfully: {doc_id}")

        return DocumentResponse(
            document_id=doc_id,
            filename=file.filename,
            chunks_count=len(result["chunks"]),
            status="success",
            message="Document processed and indexed successfully"
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-batch", response_model=List[DocumentResponse])
async def upload_batch(
        files: List[UploadFile] = File(...)
):
    """Upload multiple documents at once"""
    results = []

    for file in files:
        try:
            result = await upload_document(file=file)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(
                DocumentResponse(
                    document_id="",
                    filename=file.filename,
                    chunks_count=0,
                    status="failed",
                    message=str(e)
                )
            )

    return results


@app.post("/upload-textbook", response_model=DocumentResponse)
async def upload_textbook(
        file: UploadFile = File(...),
        book_name: str = Form(...),
        publisher: str = Form(...),
        background_tasks: BackgroundTasks = None,
        grade: Optional[str] = Form(None),
        product_name: Optional[str] = Form(None)
):
    """
    Upload and process a textbook document with enhanced metadata
    
    Args:
        file: The textbook file to upload
        book_name: Name of the book (required)
        publisher: Publisher name (required)
        grade: Grade level (optional)
        product_name: Optional custom name for the textbook (overrides generated name)
    
    Supports enhanced features for textbook files:
    - Tracks page numbers for each chunk
    - Includes subject, publisher, grade information
    - Allows custom product name override
    
    Use case: Upload textbooks for students with metadata information
    """
    try:
        logger.info(f"Processing textbook file: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        # Validate required fields
        if not book_name or not book_name.strip():
            raise HTTPException(status_code=400, detail="Book name is required")
        if not publisher or not publisher.strip():
            raise HTTPException(status_code=400, detail="Publisher is required")

        # Check if filename follows textbook naming convention (optional warning)
        if not file.filename.upper().startswith(('SGK_', 'SBT_', 'STK_')):
            logger.warning(f"Filename {file.filename} doesn't follow textbook convention")

        # Read file content
        content = await file.read()

        # Process textbook document with enhanced metadata
        result = await embedding_service.process_textbook_document(
            file_content=content,
            filename=file.filename,
            file_type=file.content_type,
            book_name=book_name,
            publisher=publisher,
            grade=grade,
            product_name=product_name
        )

        # Store in Qdrant
        doc_id = await qdrant_service.upsert_documents(
            collection_name=settings.COLLECTION_NAME,
            documents=result["chunks"],
            embeddings=result["embeddings"],
            metadata=result["metadata"]
        )

        logger.info(f"Textbook processed successfully: {doc_id}")

        return DocumentResponse(
            document_id=doc_id,
            filename=file.filename,
            chunks_count=len(result["chunks"]),
            status="success",
            message=f"Textbook processed successfully. Book: {result.get('book_metadata', {}).get('full_name', file.filename)}"
        )

    except Exception as e:
        logger.error(f"Error processing textbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoint removed - chatbot service will query Qdrant directly


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the vector store"""
    try:
        success = await qdrant_service.delete_document(
            collection_name=settings.COLLECTION_NAME,
            document_id=document_id
        )

        if success:
            return {"status": "success", "message": f"Document {document_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{document_id}/metadata")
async def get_document_metadata(document_id: str):
    """Get metadata for a specific document"""
    try:
        # Search for document chunks to get metadata
        points = await qdrant_service.search_documents(
            collection_name=settings.COLLECTION_NAME,
            query_vector=[0.0] * 1536,  # Dummy vector
            limit=1,
            filter_conditions={"must": [{"key": "document_id", "match": {"value": document_id}}]}
        )
        
        if not points:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Return metadata from first chunk (all chunks should have same document metadata)
        metadata = points[0].payload
        
        return {
            "document_id": document_id,
            "metadata": metadata,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get information about a collection"""
    try:
        info = await qdrant_service.get_collection_info(collection_name)
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))