# 📄 PDF RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and ask questions about it in natural language. Built with Claude (Anthropic), OpenAI Embeddings, Pinecone, LangChain, and Streamlit.

---

## 🧠 What Is RAG?

Large Language Models (LLMs) like Claude are powerful but have two key limitations:

- They are trained on public data up to a cutoff date — they know nothing about **your private documents**
- They have a **context window limit** — you can't simply paste a 200-page PDF into a chat

**RAG solves both problems** by:

1. **Splitting** your document into small, overlapping chunks
2. **Embedding** each chunk into a vector (a list of numbers capturing its meaning)
3. **Storing** those vectors in a database (Pinecone)
4. When you ask a question, **retrieving** only the most relevant chunks
5. **Injecting** those chunks into Claude's prompt so it can answer accurately

The result: Claude answers questions grounded strictly in your document, with source citations.

---

## ✨ Features

- 📤 **PDF Upload** — drag and drop any PDF via the browser UI
- ✂️ **Smart Chunking** — overlapping chunks preserve context at boundaries
- 🧠 **Semantic Search** — finds relevant content by meaning, not just keywords
- 💬 **Conversational Q&A** — persistent chat history within a session
- 📚 **Source Citations** — every answer shows which pages it came from and the relevance score
- 📊 **Index Stats** — see how many vectors are stored in Pinecone
- 🗑️ **Reset** — delete the index and start fresh with a new document
- 📝 **Persistent Logging** — all events logged to console and `logs/rag_app.log`

---

## 🏗️ Architecture

```
pdf-rag-app/
│
├── app.py                        # Streamlit UI entry point
├── requirements.txt              # Python dependencies
├── .env                          # API keys (never commit this)
├── .gitignore
│
├── config/
│   └── settings.py               # Centralized config & environment loading
│
├── core/
│   ├── document_processor.py     # PDF loading, cleaning, chunking
│   ├── embeddings.py             # OpenAI embedding model wrapper
│   ├── vector_store.py           # Pinecone index management & upsert
│   └── retriever.py              # Semantic similarity search
│
├── chains/
│   └── rag_chain.py              # RAG pipeline orchestration + Claude LLM
│
├── utils/
│   └── logger.py                 # Centralized structured logging
│
├── data/
│   └── uploads/                  # Temporary storage for uploaded PDFs
│
└── logs/
    └── rag_app.log               # Persistent application log
```

### Data Flow

```
INGESTION (one-time per PDF)
─────────────────────────────────────────
PDF File
  → document_processor.py   (load, clean, chunk)
  → embeddings.py            (OpenAI text-embedding-3-small)
  → vector_store.py          (upsert vectors to Pinecone)

QUERY (every user question)
─────────────────────────────────────────
User Question
  → retriever.py             (semantic search → top 5 chunks)
  → rag_chain.py             (build prompt → Claude Sonnet)
  → app.py                   (display answer + source citations)
```

---

## 🛠️ Tech Stack

| Component      | Technology                      | Purpose                    |
| -------------- | ------------------------------- | -------------------------- |
| **LLM**        | Claude Sonnet (Anthropic)       | Answer generation          |
| **Embeddings** | text-embedding-3-small (OpenAI) | Text → vector conversion   |
| **Vector DB**  | Pinecone (Serverless)           | Semantic similarity search |
| **Framework**  | LangChain                       | RAG orchestration          |
| **UI**         | Streamlit                       | Web interface              |
| **PDF Loader** | PyMuPDF                         | Fast, accurate PDF parsing |

---

## ⚙️ Prerequisites

- Python 3.10 or higher
- API keys for Anthropic, OpenAI, and Pinecone (see setup below)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-rag-app.git
cd pdf-rag-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on Mac/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your API Keys

You need three API keys:

**Anthropic (Claude)**

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up / log in → **API Keys** → **Create Key**
3. New accounts receive $5 free credits

**OpenAI (Embeddings)**

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / log in → **API Keys** → **Create new secret key**
3. Used only for embeddings — very low cost (~$0.0001 per 1K tokens)

**Pinecone (Vector Database)**

1. Go to [pinecone.io](https://www.pinecone.io) and create a free account
2. From the dashboard → **API Keys** → copy the default key
3. Note your cloud region (e.g., `us-east-1`)

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=us-east-1
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 6. Run the App

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload a PDF** using the sidebar on the left
2. Click **Process PDF** — this chunks, embeds, and stores the document in Pinecone
3. Once processing is complete, type your question in the chat box
4. Claude will answer based strictly on your document
5. Expand **Sources** below each answer to see which pages were used
6. Use **Reset Index** in the sidebar to clear all data and start fresh

---

## 💰 Cost Estimate

For typical usage (uploading and querying a 50-page PDF):

| Service   | Action                            | Estimated Cost |
| --------- | --------------------------------- | -------------- |
| OpenAI    | Embedding a 50-page PDF           | ~$0.002        |
| Anthropic | 20 questions to Claude Sonnet     | ~$0.05         |
| Pinecone  | Free tier (1 index, 100K vectors) | $0.00          |
| **Total** |                                   | **< $0.10**    |

---

## 🔧 Configuration

All tunable parameters live in `config/settings.py`:

| Parameter         | Default                    | Description                             |
| ----------------- | -------------------------- | --------------------------------------- |
| `LLM_MODEL`       | `claude-sonnet-4-20250514` | Claude model to use                     |
| `LLM_TEMPERATURE` | `0.0`                      | 0 = deterministic, factual answers      |
| `LLM_MAX_TOKENS`  | `1024`                     | Max length of Claude's response         |
| `EMBEDDING_MODEL` | `text-embedding-3-small`   | OpenAI embedding model                  |
| `CHUNK_SIZE`      | `1000`                     | Characters per chunk                    |
| `CHUNK_OVERLAP`   | `200`                      | Overlapping characters between chunks   |
| `RETRIEVER_TOP_K` | `5`                        | Number of chunks retrieved per question |

---

## 📁 Key Design Decisions

**Separation of Concerns** — Each file has a single responsibility. The UI never touches Pinecone directly; the chain never reads files. Swapping Pinecone for another vector DB requires changing only `vector_store.py`.

**Fail Fast** — `validate_settings()` checks all API keys at startup and crashes immediately with a clear message if any are missing — rather than failing mid-request with a cryptic error.

**Similarity Threshold** — Chunks below a cosine similarity of 0.7 are filtered out to prevent irrelevant context from degrading Claude's answers.

**Dependency Injection** — Services are passed into each other rather than created internally. This makes testing easier and keeps modules decoupled.

**Session State Caching** — Streamlit reruns the full script on every interaction. Services are initialized once and cached in `st.session_state` to avoid reconnecting to Pinecone on every keypress.

---

## 🐛 Troubleshooting

**App shows "Configuration Error"**
→ Check your `.env` file exists and all three API keys are set correctly.

**"File not found" when processing PDF**
→ Make sure the `data/uploads/` directory exists (it is created automatically on first run).

**Pinecone index takes a long time to initialize**
→ First-time index creation on Pinecone can take 30–60 seconds. The app waits automatically.

**Answers seem off-topic or hallucinated**
→ The retrieved chunks may not be relevant enough. Try rephrasing your question to be more specific. Check `logs/rag_app.log` for retrieval scores.

**Out of memory errors on large PDFs**
→ Reduce `CHUNK_SIZE` in `config/settings.py` to process fewer characters per chunk.

---

## 🔮 Possible Enhancements

- **Multi-document support** — store multiple PDFs in one index with metadata filtering
- **Streaming responses** — stream Claude's answer token by token for better UX
- **Conversation memory** — pass chat history into the prompt for follow-up questions
- **Reranking** — add a cross-encoder reranker after retrieval for higher precision
- **Authentication** — add login to support multiple users with isolated indexes
- **Docker** — containerize for one-command deployment

---

## 📄 License

MIT License — free to use, modify, and distribute.
