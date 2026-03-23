"""
core/embeddings.py

Responsible for:
1. Initializing the OpenAI embedding model
2. Providing a clean interface to embed text (queries and documents)
3. Handling embedding errors gracefully

Why OpenAI embeddings?
    - text-embedding-3-small is the industry standard: high quality, low cost
    - ~$0.00002 per 1K tokens — embedding a 100-page PDF costs fractions of a cent
    - 1536-dimensional vectors — rich enough to capture semantic nuance
    - Consistent: same text always produces the same vector (deterministic)

Design principle:
    This module is a thin, focused wrapper around LangChain's OpenAI embeddings.
    It owns one responsibility: give me an embedding model that works.
"""

from langchain_openai import OpenAIEmbeddings
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Wrapper around OpenAI's embedding model.

    Provides:
        - A ready-to-use LangChain embeddings object (for Pinecone integration)
        - A direct embed_text() method for embedding single strings (e.g., user queries)
        - A direct embed_documents() method for embedding batches (e.g., chunks)

    Usage:
        service = EmbeddingService()
        model = service.get_model()               # Pass to Pinecone
        vector = service.embed_query("What is X?") # Embed a question
    """

    def __init__(self):
        """
        Initialize the OpenAI embedding model.

        dimensions=EMBEDDING_DIMENSIONS ensures vectors are always 1536-dimensional.
        This must match the dimension configured in your Pinecone index,
        otherwise Pinecone will reject the vectors.
        """
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")

        self._model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            openai_api_key=OPENAI_API_KEY,
        )

        logger.info(
            f"Embedding model ready | "
            f"model={EMBEDDING_MODEL} | dimensions={EMBEDDING_DIMENSIONS}"
        )

    def get_model(self) -> OpenAIEmbeddings:
        """
        Return the raw LangChain OpenAIEmbeddings object.

        This is what you pass directly to Pinecone's LangChain integration
        so it can embed documents automatically during upsert and retrieval.

        Returns:
            OpenAIEmbeddings: Ready-to-use LangChain embedding model.

        Example:
            vector_store = PineconeVectorStore(
                index=index,
                embedding=embedding_service.get_model(),  # ← passed here
            )
        """
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single string (typically a user's question) into a vector.

        Used during retrieval — the user's question is embedded and then
        compared against all stored chunk vectors in Pinecone to find
        the closest matches.

        Args:
            text: The query string to embed.

        Returns:
            A list of 1536 floats representing the query vector.

        Raises:
            ValueError: If text is empty.
            Exception: Propagates OpenAI API errors with logging.

        Example:
            vector = service.embed_query("What are the key financial risks?")
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query text.")

        logger.info(f"Embedding query ({len(text)} chars)")

        try:
            vector = self._model.embed_query(text)
            logger.info(f"Query embedded successfully | vector_dim={len(vector)}")
            return vector

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings (typically document chunks) into vectors.

        Used during ingestion — all chunks from a PDF are embedded in batch
        before being upserted into Pinecone.

        Note: In practice, LangChain's Pinecone integration calls this
        automatically. This method is exposed for flexibility and testing.

        Args:
            texts: List of chunk strings to embed.

        Returns:
            List of vectors (each a list of 1536 floats).

        Raises:
            ValueError: If the list is empty.
            Exception: Propagates OpenAI API errors with logging.

        Example:
            vectors = service.embed_documents(["chunk one text", "chunk two text"])
        """
        if not texts:
            raise ValueError("Cannot embed empty list of documents.")

        logger.info(f"Embedding {len(texts)} document chunks...")

        try:
            vectors = self._model.embed_documents(texts)
            logger.info(
                f"Documents embedded successfully | "
                f"count={len(vectors)} | vector_dim={len(vectors[0])}"
            )
            return vectors

        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise