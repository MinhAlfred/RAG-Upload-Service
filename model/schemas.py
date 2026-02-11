"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    chunks_count: int
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


class ExtractResponse(BaseModel):
    """Response model for text extraction (no embedding/storage)"""
    text: str = Field(..., description="Extracted text content")
    filename: str
    file_type: str
    char_count: int = Field(..., description="Number of characters extracted")
    status: str = "success"
    message: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    filter: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResult]
    count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    qdrant_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChunkMetadata(BaseModel):
    """Metadata for document chunks"""
    document_id: str
    chunk_index: int
    total_chunks: int
    filename: str
    file_type: str
    user_id: Optional[str] = None
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    source_page: Optional[int] = None
    language: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None