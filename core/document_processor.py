"""
core/document_processor.py

Responsible for:
1. Loading PDF files from disk
2. Cleaning extracted text
3. Splitting documents into chunks for embedding

This is the first stage of the RAG ingestion pipeline.
Output of this module feeds directly into core/embeddings.py

Design principle:
    One class (DocumentProcessor) owns the full ingestion responsibility.
    It is stateless — you can call it with any PDF at any time.
"""

import os
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Handles the full document ingestion pipeline:
        PDF file → cleaned text → LangChain Document chunks

    Usage:
        processor = DocumentProcessor()
        chunks = processor.process(pdf_path)
    """

    def __init__(self):
        """
        Initialize the text splitter with settings from config.

        RecursiveCharacterTextSplitter is the industry standard for RAG.
        It tries to split on natural boundaries in this order:
            1. Paragraphs ("\\n\\n")
            2. Lines ("\\n")
            3. Sentences (". ")
            4. Words (" ")
            5. Characters ("")
        This preserves semantic meaning as much as possible.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splitting
        )
        logger.info(
            f"DocumentProcessor initialized | "
            f"chunk_size={CHUNK_SIZE} | chunk_overlap={CHUNK_OVERLAP}"
        )

    def load(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return a list of LangChain Document objects.
        Each Document represents one page of the PDF, with metadata attached.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of LangChain Document objects (one per page).

        Raises:
            FileNotFoundError: If the PDF does not exist at the given path.
            ValueError: If the file is not a PDF.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"PDF not found at path: {file_path}")

        if path.suffix.lower() != ".pdf":
            logger.error(f"Invalid file type: {path.suffix}")
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        logger.info(f"Loading PDF: {path.name}")

        # PyMuPDFLoader is faster and more accurate than PyPDFLoader
        # It preserves layout better and handles complex PDFs (tables, columns)
        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from '{path.name}'")
        return documents

    def clean(self, documents: List[Document]) -> List[Document]:
        """
        Clean raw extracted text from PDF pages.
        PDFs often contain noisy artifacts from extraction:
            - Multiple consecutive blank lines
            - Excessive whitespace
            - Page headers/footers repeated on every page
            - Hyphenated word breaks from column layouts

        Args:
            documents: Raw LangChain Documents from the loader.

        Returns:
            Documents with cleaned page_content.
        """
        logger.info(f"Cleaning {len(documents)} pages...")

        cleaned = []
        for doc in documents:
            text = doc.page_content

            # Fix hyphenated line breaks (e.g., "infor-\nmation" → "information")
            text = re.sub(r"-\n", "", text)

            # Collapse multiple newlines into a maximum of two (paragraph break)
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Collapse multiple spaces into one
            text = re.sub(r" {2,}", " ", text)

            # Strip leading/trailing whitespace per page
            text = text.strip()

            # Skip pages that are effectively empty after cleaning
            # (e.g., blank pages, image-only pages)
            if len(text) < 50:
                logger.warning(
                    f"Skipping near-empty page {doc.metadata.get('page', '?')} "
                    f"({len(text)} chars after cleaning)"
                )
                continue

            cleaned.append(Document(page_content=text, metadata=doc.metadata))

        logger.info(f"Cleaned documents: {len(cleaned)} pages retained")
        return cleaned

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split cleaned documents into smaller overlapping chunks.
        This is the core of the ingestion pipeline.

        Each chunk becomes one vector in Pinecone.
        Metadata (source file, page number) is preserved on every chunk
        so we can cite sources when answering.

        Args:
            documents: Cleaned LangChain Documents.

        Returns:
            List of chunk Documents ready for embedding.
        """
        logger.info(f"Splitting {len(documents)} pages into chunks...")

        chunks = self.text_splitter.split_documents(documents)

        # Enrich metadata on every chunk for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)

        logger.info(
            f"Split complete: {len(chunks)} chunks created from {len(documents)} pages"
        )
        return chunks

    def process(self, file_path: str) -> List[Document]:
        """
        Full ingestion pipeline: load → clean → split.
        This is the primary public method — call this from outside.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of chunk Documents ready to be embedded and stored in Pinecone.

        Example:
            processor = DocumentProcessor()
            chunks = processor.process("data/uploads/report.pdf")
        """
        logger.info(f"Starting document processing pipeline for: {file_path}")

        documents = self.load(file_path)
        cleaned = self.clean(documents)
        chunks = self.split(cleaned)

        logger.info(
            f"Pipeline complete | file='{Path(file_path).name}' | "
            f"pages={len(documents)} | chunks={len(chunks)}"
        )
        return chunks

    def save_upload(self, uploaded_file) -> str:
        """
        Save a Streamlit uploaded file to the uploads directory on disk.
        Streamlit gives us a file-like object (not a path), so we
        need to write it to disk before PyMuPDFLoader can read it.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Full path to the saved file on disk.
        """
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Uploaded file saved to: {save_path}")
        return save_path