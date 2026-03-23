"""
core/vector_store.py

Responsible for:
1. Connecting to Pinecone
2. Creating the index if it doesn't exist
3. Upserting document chunk vectors into Pinecone
4. Exposing a LangChain-compatible vector store object for retrieval

Why Pinecone?
    - Fully managed — no infrastructure to maintain
    - Scales to billions of vectors
    - Sub-millisecond query latency
    - Native LangChain integration
    - Free tier is generous for development and small production apps

Design principle:
    VectorStoreService owns all Pinecone interactions.
    No other module touches Pinecone directly.
    This makes it easy to swap Pinecone for another DB (Weaviate, Qdrant)
    by only changing this file.
"""

import time
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_METRIC,
    EMBEDDING_DIMENSIONS,
)
from core.embeddings import EmbeddingService
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Manages the full lifecycle of the Pinecone vector store:
        - Index creation (first time setup)
        - Document upsert (ingestion pipeline)
        - Vector store retrieval (query pipeline)

    Usage:
        embedding_service = EmbeddingService()
        vs_service = VectorStoreService(embedding_service)
        vs_service.upsert(chunks)                  # During ingestion
        store = vs_service.get_vector_store()       # During retrieval
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize Pinecone client and ensure the index exists.

        Args:
            embedding_service: Initialized EmbeddingService instance.
                               Injected here (dependency injection pattern)
                               rather than created internally — makes testing easier.
        """
        self.embedding_service = embedding_service
        self._vector_store: Optional[PineconeVectorStore] = None

        logger.info("Connecting to Pinecone...")
        self._client = Pinecone(api_key=PINECONE_API_KEY)

        self._ensure_index_exists()
        logger.info("VectorStoreService ready")

    def _ensure_index_exists(self) -> None:
        """
        Create the Pinecone index if it doesn't already exist.

        This is idempotent — safe to call on every startup.
        If the index already exists, it does nothing.
        If it doesn't exist, it creates it with the correct dimensions
        and metric to match our embedding model.

        Pinecone Serverless is used (no pods to manage or pay for at rest).
        """
        existing_indexes = [i.name for i in self._client.list_indexes()]

        if PINECONE_INDEX_NAME in existing_indexes:
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists")
            return

        logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")

        self._client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,    # Must match embedding model output
            metric=PINECONE_METRIC,            # Cosine similarity — standard for text
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,          # e.g., "aws"
                region=PINECONE_REGION,        # e.g., "us-east-1"
            ),
        )

        # Pinecone index creation is async — wait until it's ready
        # before attempting to upsert or query
        self._wait_for_index_ready()

    def _wait_for_index_ready(self, timeout_seconds: int = 60) -> None:
        """
        Poll Pinecone until the index status is 'Ready'.
        New indexes take 10-30 seconds to initialize on Pinecone's servers.
        Without this wait, upsert calls immediately after creation will fail.

        Args:
            timeout_seconds: Max seconds to wait before giving up.

        Raises:
            TimeoutError: If index doesn't become ready within the timeout.
        """
        logger.info("Waiting for Pinecone index to become ready...")
        elapsed = 0
        poll_interval = 5  # Check every 5 seconds

        while elapsed < timeout_seconds:
            status = self._client.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready"):
                logger.info(f"Pinecone index ready after {elapsed}s")
                return

            logger.info(f"Index not ready yet, waiting {poll_interval}s...")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' did not become ready "
            f"within {timeout_seconds} seconds."
        )

    def upsert(self, chunks: List[Document]) -> None:
        """
        Embed document chunks and upsert them into Pinecone.

        'Upsert' = insert + update:
            - New chunks are inserted
            - Existing chunks with the same ID are updated
        This means re-uploading the same PDF won't create duplicates.

        LangChain's PineconeVectorStore.from_documents() handles:
            1. Calling OpenAI to embed each chunk
            2. Batching upserts efficiently (100 vectors per request)
            3. Attaching metadata to each vector

        Args:
            chunks: List of Document chunks from DocumentProcessor.

        Raises:
            ValueError: If chunks list is empty.
            Exception: Propagates Pinecone or OpenAI errors with logging.
        """
        if not chunks:
            raise ValueError("Cannot upsert empty chunk list.")

        logger.info(f"Upserting {len(chunks)} chunks into Pinecone...")

        try:
            # from_documents() embeds + upserts in one call
            # It also caches the vector store object for reuse
            self._vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embedding_service.get_model(),
                index_name=PINECONE_INDEX_NAME,
            )

            logger.info(
                f"Upsert complete | {len(chunks)} chunks stored in "
                f"index '{PINECONE_INDEX_NAME}'"
            )

        except Exception as e:
            logger.error(f"Failed to upsert chunks into Pinecone: {e}")
            raise

    def get_vector_store(self) -> PineconeVectorStore:
        """
        Return a LangChain-compatible PineconeVectorStore object.

        This object is passed to the retriever (core/retriever.py),
        which uses it to perform similarity searches at query time.

        If upsert() has already been called this session, returns the
        cached instance. Otherwise, connects to the existing Pinecone index.

        Returns:
            PineconeVectorStore: Ready for similarity search.

        Raises:
            RuntimeError: If the index doesn't exist yet (nothing ingested).
        """
        if self._vector_store:
            logger.info("Returning cached vector store instance")
            return self._vector_store

        logger.info(f"Connecting to existing Pinecone index '{PINECONE_INDEX_NAME}'...")

        try:
            self._vector_store = PineconeVectorStore(
                index=self._client.Index(PINECONE_INDEX_NAME),
                embedding=self.embedding_service.get_model(),
            )
            logger.info("Vector store connected successfully")
            return self._vector_store

        except Exception as e:
            logger.error(f"Failed to connect to Pinecone vector store: {e}")
            raise

    def get_index_stats(self) -> dict:
        """
        Return stats about the current Pinecone index.
        Useful for debugging and displaying in the UI
        (e.g., 'Index contains 243 vectors').

        Returns:
            dict with keys: total_vector_count, dimension, index_fullness, etc.
        """
        try:
            index = self._client.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to fetch index stats: {e}")
            return {}

    def delete_index(self) -> None:
        """
        Delete the Pinecone index entirely.
        Use this to reset the app and start fresh with new documents.

        WARNING: This is irreversible. All stored vectors will be lost.
        In production, you'd gate this behind an admin-only UI or CLI command.
        """
        logger.warning(f"Deleting Pinecone index '{PINECONE_INDEX_NAME}'...")

        try:
            self._client.delete_index(PINECONE_INDEX_NAME)
            self._vector_store = None
            logger.info("Index deleted successfully")

        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            raise