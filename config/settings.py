"""
config/settings.py

Central configuration for the PDF RAG application.
Supports both:
    - Local development: reads from .env file via python-dotenv
    - Streamlit Cloud: reads from st.secrets via Streamlit's secrets manager

No other module should read secrets directly — always import from here.
"""

import os
from dotenv import load_dotenv

# Load .env file for local development
# On Streamlit Cloud this is a no-op (no .env file exists — that's fine)
load_dotenv()


def _get_secret(key: str) -> str:
    """
    Get a secret value — works in both local and Streamlit Cloud environments.

    Priority order:
        1. Streamlit secrets (st.secrets) — used on Streamlit Cloud
        2. Environment variables (os.getenv) — used locally via .env

    Args:
        key: The secret key name (e.g., "ANTHROPIC_API_KEY")

    Returns:
        The secret value as a string, or empty string if not found.
    """
    # Try Streamlit secrets first (Streamlit Cloud environment)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # Streamlit not available or secrets not configured — fall through
        pass

    # Fall back to environment variables (local .env file)
    return os.getenv(key, "")


# ─────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────

ANTHROPIC_API_KEY: str = _get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str = _get_secret("OPENAI_API_KEY")
PINECONE_API_KEY: str = _get_secret("PINECONE_API_KEY")


# ─────────────────────────────────────────
# LLM SETTINGS (Claude)
# ─────────────────────────────────────────

LLM_MODEL: str = "claude-sonnet-4-20250514"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024


# ─────────────────────────────────────────
# EMBEDDING SETTINGS (OpenAI)
# ─────────────────────────────────────────

EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536


# ─────────────────────────────────────────
# PINECONE SETTINGS
# ─────────────────────────────────────────

PINECONE_INDEX_NAME: str = "pdf-rag-index"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = _get_secret("PINECONE_ENV") or "us-east-1"
PINECONE_METRIC: str = "cosine"


# ─────────────────────────────────────────
# DOCUMENT PROCESSING SETTINGS
# ─────────────────────────────────────────

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


# ─────────────────────────────────────────
# RETRIEVAL SETTINGS
# ─────────────────────────────────────────

RETRIEVER_TOP_K: int = 5


# ─────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────

UPLOAD_DIR: str = "data/uploads"
APP_TITLE: str = "📄 PDF RAG Assistant"
APP_DESCRIPTION: str = "Upload a PDF and ask questions about it."


# ─────────────────────────────────────────
# VALIDATION — fail fast if keys are missing
# ─────────────────────────────────────────

def validate_settings() -> None:
    """
    Call this at app startup.
    Raises ValueError immediately if any required API key is missing,
    rather than failing silently mid-request.
    """
    missing = []

    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please check your .env file (local) or Streamlit secrets (cloud)."
        )