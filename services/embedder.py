"""
OpenAI embedding implementation - Simplified version
Using text-embedding-3-small (1536 dimensions)
"""
import logging
from typing import List
import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI embedding model wrapper - text-embedding-3-small"""

    def __init__(self, api_key: str):
        """
        Initialize OpenAI embedder

        Args:
            api_key: OpenAI API key
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
        self.dimension = 1536

        logger.info(f"Initialized OpenAI embedder: {self.model} ({self.dimension}D)")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (1536 dimensions each)
        """
        try:
            if not texts:
                return []

            # OpenAI supports large batches
            # But we batch for rate limiting safety
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )

                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

                # Small delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions)
        """
        embeddings = await self.embed([text])
        return embeddings[0]

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    async def close(self):
        """Close the client"""
        await self.client.close()