"""
chains/rag_chain.py

Responsible for:
1. Building the RAG prompt with retrieved context
2. Initializing and managing the Claude LLM
3. Orchestrating the full RAG pipeline (retrieve → prompt → generate)
4. Returning structured answers with source citations

This is the orchestration layer — it coordinates all core modules
and produces the final answer delivered to the user.

Design principle:
    The chain owns the prompt template and LLM configuration.
    It delegates retrieval to RetrieverService and never touches
    Pinecone or embeddings directly. Clean separation of concerns.
"""

from typing import Optional
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from core.retriever import RetrieverService
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────
# RESPONSE DATA STRUCTURE
# ─────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Structured response returned by the RAG chain.

    Using a dataclass instead of a raw dict because:
    - Type safety: IDE autocomplete and type checking work correctly
    - Explicit contract: Callers know exactly what fields to expect
    - Extensible: Easy to add fields (confidence score, latency, etc.) later

    Fields:
        answer:   Claude's generated answer as a string
        sources:  List of Document chunks used to generate the answer
        query:    The original user question (for display/logging)
        model:    The LLM model used (for transparency in UI)
    """
    answer: str
    sources: list[Document]
    query: str
    model: str = LLM_MODEL


# ─────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────

# This is the most important string in the entire application.
# It determines how Claude uses the retrieved context to answer questions.
#
# Key prompt engineering decisions:
#   1. Explicit instruction to use ONLY the provided context
#      → Prevents hallucination from Claude's training data
#   2. Explicit instruction to say "I don't know" when context is insufficient
#      → Prevents confident wrong answers (the worst RAG failure mode)
#   3. Citation instruction — reference page numbers in the answer
#      → Builds user trust and enables fact-checking
#   4. Concise but complete — don't pad, don't truncate
#      → Avoids verbose LLM responses that waste tokens and user time

RAG_PROMPT_TEMPLATE = """You are an expert assistant that answers questions strictly based on the provided document context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below
- If the context does not contain enough information to answer the question, say: "I don't have enough information in the provided document to answer this question."
- Do NOT use your general knowledge or training data to fill gaps
- Cite the page number(s) where you found the information (e.g., "According to page 3...")
- Be concise but complete — answer fully without unnecessary padding
- If the question asks for a list, use bullet points for clarity

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# ─────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────

class RAGChain:
    """
    Orchestrates the full RAG pipeline:
        User question
            → RetrieverService fetches relevant chunks
            → Chunks formatted into prompt context
            → Claude generates a grounded answer
            → RAGResponse returned with sources attached

    Usage:
        chain = RAGChain(retriever_service)
        response = chain.run("What are the key risks mentioned in the report?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(self, retriever_service: RetrieverService):
        """
        Initialize the RAG chain with a retriever and Claude LLM.

        Args:
            retriever_service: Initialized RetrieverService instance.
                               Injected (not created internally) for testability.
        """
        self._retriever = retriever_service
        self._llm = self._init_llm()
        self._prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self._output_parser = StrOutputParser()

        logger.info(
            f"RAGChain initialized | "
            f"model={LLM_MODEL} | temperature={LLM_TEMPERATURE}"
        )

    def _init_llm(self) -> ChatAnthropic:
        """
        Initialize the Claude LLM via LangChain's Anthropic integration.

        temperature=0.0: Deterministic, factual answers.
                         For RAG, creativity is the enemy — you want Claude
                         to stick to the retrieved context, not improvise.

        max_tokens=LLM_MAX_TOKENS: Caps response length to control cost.
                                    1024 tokens ≈ ~750 words — enough for
                                    detailed answers without runaway responses.

        Returns:
            Initialized ChatAnthropic LLM ready for inference.
        """
        logger.info(f"Initializing Claude LLM | model={LLM_MODEL}")

        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            anthropic_api_key=ANTHROPIC_API_KEY,
        )

    def _format_context(self, chunks: list[Document]) -> str:
        """
        Format retrieved chunks into a single context string for the prompt.

        Each chunk is formatted with its page number clearly labeled.
        This structured format helps Claude:
            1. Identify which page each piece of information comes from
            2. Cite sources accurately in its answer
            3. Distinguish between multiple chunks if they conflict

        Args:
            chunks: Retrieved Document chunks from RetrieverService.

        Returns:
            Formatted context string injected into the prompt template.

        Example output:
            [Page 3 | Score: 0.92]
            Revenue grew 23% year-over-year driven by...

            [Page 7 | Score: 0.87]
            Key risks include market volatility and...
        """
        formatted_chunks = []

        for chunk in chunks:
            page = chunk.metadata.get("page", "Unknown")
            score = chunk.metadata.get("retrieval_score", "N/A")
            source = chunk.metadata.get("source", "Unknown")
            text = chunk.page_content.strip()

            formatted_chunks.append(
                f"[Page {page} | Relevance: {score} | Source: {source}]\n{text}"
            )

        return "\n\n---\n\n".join(formatted_chunks)

    def run(
        self,
        query: str,
        source_filter: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Pipeline:
            1. Retrieve top_k relevant chunks from Pinecone
            2. Format chunks into structured context
            3. Inject context + query into prompt template
            4. Send to Claude for answer generation
            5. Return RAGResponse with answer + source chunks

        Args:
            query: The user's question as a plain string.
            source_filter: Optional Pinecone metadata filter.
                           e.g., {"source": "report.pdf"} to restrict
                           retrieval to a specific document.

        Returns:
            RAGResponse dataclass with answer, sources, query, model.

        Raises:
            ValueError: If query is empty.
            Exception: Propagates retrieval or LLM errors with logging.

        Example:
            response = chain.run("What is the company's revenue in 2023?")
            print(response.answer)
            for doc in response.sources:
                print(f"Source: page {doc.metadata['page']}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"RAG pipeline starting | query='{query[:80]}'")

        # ── Step 1: Retrieve relevant chunks ────────────────────────
        chunks = self._retriever.retrieve(query, filter=source_filter)
        logger.info(f"Retrieved {len(chunks)} chunks for context")

        # ── Step 2: Format context for prompt ───────────────────────
        context = self._format_context(chunks)

        # ── Step 3: Build and invoke the chain ──────────────────────
        # LangChain LCEL (pipe) syntax:
        #   RunnablePassthrough() passes the input dict unchanged to the prompt
        #   prompt formats the context + question into the full prompt string
        #   llm sends the prompt to Claude and returns a message object
        #   output_parser extracts the string content from the message

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | self._prompt
            | self._llm
            | self._output_parser
        )

        logger.info("Invoking Claude LLM...")

        try:
            answer = chain.invoke(query)
            logger.info(
                f"RAG pipeline complete | "
                f"answer_length={len(answer)} chars | "
                f"sources={len(chunks)} chunks"
            )

            return RAGResponse(
                answer=answer,
                sources=chunks,
                query=query,
                model=LLM_MODEL,
            )

        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            raise

    def get_source_summary(self, response: RAGResponse) -> list[dict]:
        """
        Extract a clean source summary from a RAGResponse.
        Used by the Streamlit UI to display source citations
        in a structured, readable format below the answer.

        Args:
            response: A completed RAGResponse from run().

        Returns:
            List of dicts, each with: page, score, source, preview.

        Example return value:
            [
                {
                    "page": 3,
                    "score": 0.92,
                    "source": "annual_report.pdf",
                    "preview": "Revenue grew 23% year-over-year..."
                },
                ...
            ]
        """
        sources = []
        for doc in response.sources:
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "score": doc.metadata.get("retrieval_score", "?"),
                "source": doc.metadata.get("source", "?"),
                "preview": doc.page_content[:150].strip() + "...",
            })
        return sources