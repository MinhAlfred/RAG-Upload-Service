"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Union
import os


class Settings(BaseSettings):
    """Application settings"""

    # App settings
    APP_NAME: str = "Document Embedding Service"
    DEBUG: bool = False

    # Qdrant settings
    QDRANT_URL: Optional[str] = None  # For Qdrant Cloud: https://xxx.cloud.qdrant.io
    QDRANT_HOST: str = "localhost"  # For local/self-hosted
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None  # Required for Qdrant Cloud
    QDRANT_USE_CLOUD: bool = False  # Set to True when using Qdrant Cloud
    COLLECTION_NAME: str = "cs_chatbot_docs"
    EMBEDDING_DIMENSION: int = 1024  # For Cohere embed-multilingual-v3.0

    # Embedding settings
    OPENAI_API_KEY: str  # Required
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small dimension

    # Document processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FILE_TYPES: List[str] = [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "image/png",
        "image/jpeg",
        "image/jpg",
        "text/x-python",
        "application/json"
    ]

    # CORS
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    # Storage
    TEMP_UPLOAD_DIR: str = "/tmp/uploads"

    # Batch processing
    MAX_BATCH_SIZE: int = 10

    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create temp directory if not exists
os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)