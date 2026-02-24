"""
Qdrant vector database service
"""
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(
            self,
            url: str,
            api_key: str,
            timeout: int = 30
    ):
        if not url or not api_key:
            raise ValueError("Qdrant Cloud requires url and api_key")

        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=False,
            https=True
        )

        logger.info(f"Initialized Qdrant Cloud client: {url}")

    async def init_collection(
            self,
            collection_name: str,
            vector_size: int,
            distance: Distance = Distance.COSINE,
            recreate: bool = False
    ):
        """Initialize or create collection"""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(
                col.name == collection_name
                for col in collections.collections
            )

            if recreate and collection_exists:
                logger.info(f"Deleting existing collection: {collection_name}")
                await self.client.delete_collection(collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(f"Creating collection: {collection_name}")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                logger.info(f"Collection created: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")

            # Ensure payload index exists for duplicate-detection field.
            # Qdrant requires an index on a nested keyword field to filter reliably.
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.file_hash",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info("Payload index ensured: metadata.file_hash")
            except Exception as idx_err:
                # Index may already exist – that is fine, log and continue
                logger.debug(f"Payload index creation skipped (may already exist): {idx_err}")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    async def upsert_documents(
            self,
            collection_name: str,
            documents: List[str],
            embeddings: List[List[float]],
            metadata: List[Dict[str, Any]]
    ) -> str:
        """
        Insert or update documents with embeddings
        Returns the document_id of the first chunk
        """
        try:
            if not (len(documents) == len(embeddings) == len(metadata)):
                raise ValueError("Documents, embeddings, and metadata must have same length")

            # Generate document ID for this batch
            doc_id = str(uuid.uuid4())

            # Create points
            points = []
            for idx, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
                point_id = str(uuid.uuid4())

                # Create new payload structure with text and metadata separated
                payload = {
                    "text": doc,
                    "metadata": {
                        **meta,  # Include all existing metadata fields
                        "document_id": doc_id,
                        "chunk_index": idx,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

                point = PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
                points.append(point)

            # Upsert to Qdrant in batches to avoid timeout
            batch_size = 100  # Process 100 points at a time
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                await self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size} ({len(batch_points)} points)")

            logger.info(f"Successfully upserted {len(points)} points for document {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise

    async def search(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int = 5,
            score_threshold: float = 0.7,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            # Build filter if provided
            # Keys in filter_dict that refer to metadata fields must use "metadata.<key>" notation
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    # Automatically prefix bare keys with "metadata." so callers can pass
                    # e.g. {"document_id": "..."} without worrying about nesting
                    resolved_key = key if "." in key else f"metadata.{key}"
                    conditions.append(
                        FieldCondition(
                            key=resolved_key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)

            # Search
            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {})
                })

            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def find_document_by_hash(
            self,
            collection_name: str,
            file_hash: str
    ) -> Optional[str]:
        """
        Find an existing document by its file hash (MD5 of raw bytes).
        Returns the document_id string if a duplicate exists, None otherwise.
        This check should be done BEFORE expensive OCR/embedding work.
        """
        try:
            results, _ = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if results:
                doc_id = results[0].payload.get("metadata", {}).get("document_id")
                logger.info(f"Duplicate detected: hash={file_hash}, existing document_id={doc_id}")
                return doc_id
            logger.info(f"No duplicate found for hash={file_hash}")
            return None
        except Exception as e:
            # Log as ERROR so this is never silently ignored
            logger.error(f"[find_document_by_hash] Qdrant scroll failed for hash={file_hash}: {e}", exc_info=True)
            # Re-raise so the caller can decide — do NOT silently allow duplicates through
            raise

    async def delete_document(
            self,
            collection_name: str,
            document_id: str
    ) -> bool:
        """Delete all chunks of a document"""
        try:
            # Delete points with matching document_id (nested under metadata)
            await self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )

            logger.info(f"Deleted document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = await self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def close(self):
        """Close client connection"""
        try:
            await self.client.close()
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")