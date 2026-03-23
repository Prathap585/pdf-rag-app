# debug_pdf.py — run this in your project root
from core.document_processor import DocumentProcessor

processor = DocumentProcessor()
docs = processor.load("data/uploads/borrowersigned.pdf")  # ← put your PDF name here

print(f"Pages loaded: {len(docs)}")
for i, doc in enumerate(docs[:3]):  # show first 3 pages
    print(f"\n--- Page {i+1} ({len(doc.page_content)} chars) ---")
    print(doc.page_content[:300])