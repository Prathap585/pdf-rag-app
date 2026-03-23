"""
app.py

Streamlit UI entry point for the PDF RAG Assistant.

Responsibilities:
    - Render the chat interface and sidebar
    - Handle PDF upload and trigger ingestion pipeline
    - Accept user questions and display Claude's answers
    - Show source citations for every answer
    - Manage session state across Streamlit reruns

Design principle:
    This file is purely a UI layer. All business logic lives in core/ and chains/.
    app.py only instantiates services, calls their public methods, and renders results.

Run with:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from config.settings import validate_settings, APP_TITLE, APP_DESCRIPTION
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingService
from core.vector_store import VectorStoreService
from core.retriever import RetrieverService
from chains.rag_chain import RAGChain
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────
# STARTUP VALIDATION
# ─────────────────────────────────────────

try:
    validate_settings()
except ValueError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.stop()  # Halt the app — nothing works without valid API keys


# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists values across reruns within the same session.
# We initialize services here once and reuse them — avoiding re-initialization
# on every rerun (which would reconnect to Pinecone, reload models, etc.)

def init_session_state():
    """Initialize all session state variables on first load."""

    if "services_initialized" not in st.session_state:
        st.session_state.services_initialized = False

    if "chat_history" not in st.session_state:
        # Each item: {"role": "user"|"assistant", "content": str, "sources": list}
        st.session_state.chat_history = []

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    if "current_pdf_name" not in st.session_state:
        st.session_state.current_pdf_name = None

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "vector_store_service" not in st.session_state:
        st.session_state.vector_store_service = None


def initialize_services() -> bool:
    """
    Initialize all backend services and cache them in session state.
    Called once per session — subsequent reruns reuse the cached instances.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    if st.session_state.services_initialized:
        return True

    try:
        with st.spinner("🔧 Initializing services..."):
            logger.info("Initializing all services...")

            # Build the service graph (bottom-up dependency order)
            embedding_service = EmbeddingService()
            vs_service = VectorStoreService(embedding_service)
            retriever_service = RetrieverService(vs_service.get_vector_store())
            rag_chain = RAGChain(retriever_service)

            # Cache in session state for reuse across reruns
            st.session_state.embedding_service = embedding_service
            st.session_state.vector_store_service = vs_service
            st.session_state.retriever_service = retriever_service
            st.session_state.rag_chain = rag_chain
            st.session_state.services_initialized = True

            logger.info("All services initialized successfully")
            return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        st.error(f"Failed to initialize services: {e}")
        return False


# ─────────────────────────────────────────
# PDF INGESTION
# ─────────────────────────────────────────

def process_pdf(uploaded_file) -> bool:
    """
    Run the full ingestion pipeline for an uploaded PDF.
    Updates session state on success.

    Pipeline: save to disk → load → clean → chunk → embed → upsert to Pinecone

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        True if ingestion succeeded, False otherwise.
    """
    try:
        processor = DocumentProcessor()

        with st.spinner(f"📖 Reading '{uploaded_file.name}'..."):
            save_path = processor.save_upload(uploaded_file)
            logger.info(f"PDF saved to: {save_path}")

        with st.spinner("✂️ Chunking document..."):
            chunks = processor.process(save_path)
            logger.info(f"Created {len(chunks)} chunks")

        with st.spinner(f"🧠 Embedding {len(chunks)} chunks & uploading to Pinecone..."):
            st.session_state.vector_store_service.upsert(chunks)
            logger.info("Chunks upserted to Pinecone")

        # Reinitialize retriever + chain with freshly upserted vector store
        # (so the chain uses the latest data)
        retriever_service = RetrieverService(
            st.session_state.vector_store_service.get_vector_store()
        )
        st.session_state.rag_chain = RAGChain(retriever_service)

        # Update session state
        st.session_state.pdf_processed = True
        st.session_state.current_pdf_name = uploaded_file.name
        st.session_state.chat_history = []  # Clear history for new document

        logger.info(f"PDF ingestion complete: {uploaded_file.name}")
        return True

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        st.error(f"❌ Failed to process PDF: {e}")
        return False


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with upload controls and index stats."""

    with st.sidebar:
        st.title("📄 PDF RAG Assistant")
        st.caption(APP_DESCRIPTION)
        st.divider()

        # ── PDF Upload ───────────────────────────────────────────────
        st.subheader("📤 Upload Document")

        uploaded_file = st.file_uploader(
            label="Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF — reports, papers, manuals, contracts.",
        )

        if uploaded_file:
            # Only re-process if a NEW file is uploaded
            if uploaded_file.name != st.session_state.current_pdf_name:
                if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                    success = process_pdf(uploaded_file)
                    if success:
                        st.success(f"✅ '{uploaded_file.name}' is ready!")
                        st.rerun()
            else:
                st.success(f"✅ '{uploaded_file.name}' is loaded and ready.")

        st.divider()

        # ── Index Stats ──────────────────────────────────────────────
        st.subheader("📊 Index Stats")

        if st.session_state.get("vector_store_service"):
            if st.button("🔄 Refresh Stats", use_container_width=True):
                stats = st.session_state.vector_store_service.get_index_stats()
                total = stats.get("total_vector_count", 0)
                st.metric("Vectors in Pinecone", f"{total:,}")

        st.divider()

        # ── Reset ────────────────────────────────────────────────────
        st.subheader("⚠️ Danger Zone")

        if st.button("🗑️ Reset Index", type="secondary", use_container_width=True):
            if st.session_state.get("vector_store_service"):
                with st.spinner("Deleting Pinecone index..."):
                    st.session_state.vector_store_service.delete_index()
                st.session_state.pdf_processed = False
                st.session_state.current_pdf_name = None
                st.session_state.chat_history = []
                st.session_state.services_initialized = False
                st.warning("Index deleted. Refresh the page to reinitialize.")
                logger.warning("Pinecone index deleted by user")


# ─────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────

def render_chat():
    """Render the main chat interface — history display + input box."""

    st.title(APP_TITLE)

    # ── Guard: require PDF to be processed first ─────────────────────
    if not st.session_state.pdf_processed:
        st.info("👈 Upload and process a PDF from the sidebar to get started.")

        # Show example questions to set user expectations
        st.subheader("💡 Example questions you can ask:")
        examples = [
            "What are the main topics covered in this document?",
            "Summarize the key findings.",
            "What risks are mentioned?",
            "What recommendations does the author make?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")
        return

    # ── Chat History ─────────────────────────────────────────────────
    # Render all previous messages in order
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show source citations below assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    # ── Chat Input ───────────────────────────────────────────────────
    if query := st.chat_input(
        placeholder=f"Ask anything about '{st.session_state.current_pdf_name}'..."
    ):
        handle_query(query)


def handle_query(query: str):
    """
    Process a user query through the RAG pipeline and display the response.

    Args:
        query: The user's question string from the chat input.
    """
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Append to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "sources": [],
    })

    # Run RAG pipeline and display response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document & generating answer..."):
            try:
                response = st.session_state.rag_chain.run(query)
                source_summary = st.session_state.rag_chain.get_source_summary(response)

                # Display the answer
                st.markdown(response.answer)

                # Display source citations
                render_sources(source_summary)

                # Append assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": source_summary,
                })

                logger.info(
                    f"Query answered | "
                    f"query='{query[:50]}' | "
                    f"answer_length={len(response.answer)} | "
                    f"sources={len(source_summary)}"
                )

            except Exception as e:
                error_msg = f"❌ Something went wrong: {e}"
                st.error(error_msg)
                logger.error(f"Query failed: {e}")


def render_sources(sources: list[dict]):
    """
    Render source citations in a clean expandable section.
    Shows page number, relevance score, and a text preview for each chunk.

    Args:
        sources: List of source dicts from RAGChain.get_source_summary().
    """
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} chunks used)", expanded=False):
        for i, source in enumerate(sources, 1):
            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                st.metric(f"Source {i}", f"Page {source['page']}")
            with col2:
                score = source.get("score", "N/A")
                score_display = f"{float(score):.0%}" if isinstance(score, float) else score
                st.metric("Relevance", score_display)
            with col3:
                st.caption(f"**{source['source']}**")
                st.caption(source["preview"])

            if i < len(sources):
                st.divider()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    """
    Application entry point.
    Order matters:
        1. Initialize session state variables
        2. Initialize backend services (once per session)
        3. Render sidebar (upload + controls)
        4. Render main chat interface
    """
    init_session_state()

    if not initialize_services():
        st.stop()

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()