"""
core/retriever.py

Responsible for:
1. Wrapping the Pinecone vector store with retrieval logic
2. Performing similarity search against stored chunk vectors
3. Applying retrieval filters and scoring thresholds
4. Returning ranked, relevant Document chunks for a given query

This is the query-time counterpart to vector_store.py (which handles ingestion).

Design principle:
    RetrieverService owns ALL retrieval decisions:
        - How many chunks to return (top_k)
        - Minimum relevance score threshold
        - Metadata filtering (e.g., retrieve only from a specific document)
    The RAG chain (chains/rag_chain.py) should never make these decisions —
    it just asks for relevant chunks and trusts this module to return good ones.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from config.settings import RETRIEVER_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum cosine similarity score to accept a chunk as relevant.
# Cosine similarity ranges from 0.0 (completely different) to 1.0 (identical).
# 0.7 is a well-tested threshold — below this, chunks are usually noise.
SIMILARITY_THRESHOLD = 0.7


class RetrieverService:
    """
    Handles semantic retrieval from the Pinecone vector store.

    Two retrieval modes:
        1. Simple similarity search — fast, returns top_k chunks by score
        2. Score-filtered search — returns only chunks above a relevance threshold

    Usage:
        retriever = RetrieverService(vector_store)
        chunks = retriever.retrieve("What are the key financial risks?")
    """

    def __init__(self, vector_store: PineconeVectorStore, top_k: int = RETRIEVER_TOP_K):
        """
        Initialize the retriever with a connected vector store.

        Args:
            vector_store: Connected PineconeVectorStore from VectorStoreService.
            top_k: Number of chunks to retrieve per query. Defaults to config value.
                   Can be overridden per-instance for different use cases
                   (e.g., top_k=3 for focused answers, top_k=10 for broad summaries).
        """
        self._vector_store = vector_store
        self._top_k = top_k

        logger.info(
            f"RetrieverService initialized | "
            f"top_k={self._top_k} | threshold={SIMILARITY_THRESHOLD}"
        )

    def retrieve(
        self,
        query: str,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Primary retrieval method — returns top_k most relevant chunks for a query.

        Steps:
            1. Pinecone embeds the query (using the same model used during ingestion)
            2. Cosine similarity is computed against all stored vectors
            3. Top_k chunks with highest similarity are returned
            4. Chunks below the similarity threshold are filtered out

        Args:
            query: The user's question as a plain string.
            filter: Optional Pinecone metadata filter dict.
                    Example: {"source": "report.pdf"} to search only one document.
                    Useful when multiple PDFs are stored in the same index.

        Returns:
            List of Document chunks ranked by relevance (most relevant first).
            May return fewer than top_k if low-scoring chunks are filtered out.

        Raises:
            ValueError: If query is empty.
            Exception: Propagates Pinecone errors with logging.

        Example:
            chunks = retriever.retrieve("What is the revenue for Q3?")
            chunks = retriever.retrieve("Risks?", filter={"source": "annual_report.pdf"})
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Retrieving chunks for query: '{query[:80]}...' " if len(query) > 80 else f"Retrieving chunks for query: '{query}'")

        try:
            # similarity_search_with_score returns (Document, score) tuples
            # Score is cosine similarity: higher = more relevant
            results_with_scores = self._vector_store.similarity_search_with_score(
                query=query,
                k=self._top_k,
                filter=filter,
            )

            # Filter out chunks below the relevance threshold
            filtered = [
                (doc, score)
                for doc, score in results_with_scores
                if score >= SIMILARITY_THRESHOLD
            ]

            if not filtered:
                logger.warning(
                    f"No chunks met the similarity threshold ({SIMILARITY_THRESHOLD}) "
                    f"for query: '{query[:50]}'"
                )
                # Return top result anyway so the LLM can attempt an answer
                # rather than failing silently with an empty context
                if results_with_scores:
                    best_doc, best_score = results_with_scores[0]
                    logger.info(f"Falling back to best available chunk | score={best_score:.3f}")
                    filtered = [(best_doc, best_score)]

            # Attach retrieval score to each chunk's metadata for transparency
            chunks = []
            for doc, score in filtered:
                doc.metadata["retrieval_score"] = round(float(score), 4)
                chunks.append(doc)

            self._log_retrieval_summary(chunks, results_with_scores)
            return chunks

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query[:50]}': {e}")
            raise

    def get_langchain_retriever(self, filter: Optional[dict] = None):
        """
        Return a native LangChain retriever object.

        LangChain's LCEL (LangChain Expression Language) chains expect
        a retriever object, not raw chunks. This method wraps our vector
        store as a standard LangChain retriever for use in rag_chain.py.

        Args:
            filter: Optional metadata filter (same as retrieve() above).

        Returns:
            LangChain BaseRetriever — compatible with LCEL pipe operator (|).

        Example:
            retriever = retriever_service.get_langchain_retriever()
            chain = retriever | prompt | llm  # LCEL chain
        """
        logger.info(f"Creating LangChain retriever | top_k={self._top_k}")

        search_kwargs = {"k": self._top_k}
        if filter:
            search_kwargs["filter"] = filter

        return self._vector_store.as_retriever(
            search_type="similarity",       # Use cosine similarity (matches our Pinecone metric)
            search_kwargs=search_kwargs,
        )

    def _log_retrieval_summary(
        self,
        chunks: List[Document],
        all_results: List[tuple],
    ) -> None:
        """
        Log a clean summary of retrieval results for debugging.
        Shows which pages were retrieved and their relevance scores.

        Args:
            chunks: Final filtered chunks returned to the caller.
            all_results: All (doc, score) pairs before filtering.
        """
        logger.info(
            f"Retrieval complete | "
            f"returned={len(chunks)} chunks | "
            f"candidates={len(all_results)} | "
            f"filtered_out={len(all_results) - len(chunks)}"
        )

        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            logger.info(
                f"  Chunk {i + 1}: "
                f"page={meta.get('page', '?')} | "
                f"score={meta.get('retrieval_score', '?')} | "
                f"source={meta.get('source', '?')} | "
                f"preview='{chunk.page_content[:60].strip()}...'"
            )