# Enterprise Knowledge Base Q&A System (RAG Implementation)

A production-ready **Retrieval-Augmented Generation (RAG)** application that enables employees to query proprietary enterprise documents through a natural language interface, receiving accurate, grounded, and citation-backed answers — eliminating hallucinations and the limitations of keyword-based enterprise search.

**🌐 Live App:** [https://enterprise-rag-system.streamlit.app](https://enterprise-rag-system.streamlit.app)

---

## Table of Contents

- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Tech Stack](#tech-stack)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Project Structure](#project-structure)
- [Key Engineering Decisions](#key-engineering-decisions)
- [UI & Design System](#ui--design-system)
- [Local Setup & Installation](#local-setup--installation)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

---

## Business Problem

Traditional enterprise search systems are constrained by rigid keyword matching — they return irrelevant results when users fail to use the exact terminology indexed in the system. On the other side, general-purpose Large Language Models (LLMs), while capable of fluent reasoning, are trained on public internet data and have no knowledge of an organization's internal, proprietary, or confidential documents. When queried about such data, they resort to **hallucination** — generating plausible-sounding but factually incorrect responses.

This application solves both problems simultaneously by combining **semantic vector retrieval** with **controlled LLM generation**, anchoring every response exclusively to private enterprise data.

---

## Solution Overview

This system implements a full **RAG (Retrieval-Augmented Generation)** pipeline consisting of two primary phases:

**Ingestion Phase:** Proprietary PDF documents are uploaded, parsed, semantically chunked, and transformed into high-dimensional vector embeddings that are stored in a local FAISS vector database.

**Query Phase:** A user's natural language question is embedded in real-time and compared against the FAISS index via similarity search. The most relevant document segments are retrieved and injected into a strict LLM prompt that prohibits the model from using any knowledge outside the provided context — guaranteeing grounded, hallucination-free responses.

---

## Tech Stack

| Layer | Technology | Detail |
|---|---|---|
| **Language** | Python 3.x | Core application language |
| **Frontend** | Streamlit | Interactive web UI with session state management |
| **LLM (Generation)** | `gemini-3.1-flash-lite-preview` | Low-latency generative model via Google GenAI |
| **Embedding Model** | `gemini-embedding-001` | Produces 3072-dimensional dense vector embeddings |
| **Vector Database** | FAISS (`IndexFlatL2`) | Facebook AI Similarity Search — in-memory L2 index |
| **PDF Parsing** | PyPDF2 | Stream-based text extraction from uploaded PDFs |
| **Vector Math** | NumPy | Float32 array construction for FAISS compatibility |
| **Styling** | Custom CSS + TOML | Enterprise-themed UI with chat bubble differentiation |
| **Deployment** | Streamlit Community Cloud | Zero-infrastructure cloud deployment |

---

## System Architecture & Workflow

The application operates through a modular, multi-stage pipeline that governs the full lifecycle of enterprise data — from raw document upload to final answer generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                          │
│                                                                 │
│  PDF Upload → PyPDF2 Extraction → Recursive Word Chunking       │
│       → Batch Embedding (gemini-embedding-001, 3072-dim)        │
│               → FAISS IndexFlatL2 Storage                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          QUERY PHASE                            │
│                                                                 │
│  User Question → Real-time Query Embedding                      │
│       → FAISS Similarity Search (Top-K = 10)                    │
│           → Context Injection into Strict RAG Prompt            │
│               → gemini-3.1-flash-lite-preview Generation        │
│                   → Grounded Answer + Citation Expander         │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: PDF Ingestion & Text Extraction

Uploaded PDF files are processed using `PyPDF2.PdfReader`. The system iterates through all pages, extracts raw text, and sanitizes it using UTF-8 encoding with `errors='ignore'` to handle malformed or non-standard characters gracefully. Pages that yield no extractable text are skipped automatically.

### Stage 2: Recursive Word-Based Chunking

Extracted text is split into semantically coherent segments using a word-based recursive chunking strategy:

- **Chunk Size:** 800 words per segment
- **Overlap:** 150 words between adjacent chunks

The 150-word overlap is a deliberate architectural choice. It ensures that information spanning a chunk boundary — such as a conclusion that references a premise from the previous segment — is captured in at least one chunk, preserving semantic continuity during retrieval.

### Stage 3: Batch-Optimized Embedding Generation

To minimize API latency and reduce the number of round-trip network calls, the system groups chunks into **batches of up to 100** and sends each batch in a single API request to the `gemini-embedding-001` model. Each chunk is transformed into a **3072-dimensional float32 vector**. The resulting embeddings are returned as a flat list and paired with their corresponding text chunks.

### Stage 4: FAISS Vector Indexing

The embeddings are stacked into a NumPy `float32` array and added to a **FAISS `IndexFlatL2`** index initialized with dimension 3072. This index performs exact **Euclidean distance (L2)** comparisons, guaranteeing that the nearest neighbours returned are the mathematically most similar vectors — ensuring retrieval precision without approximation error. The text chunks are stored in a parallel in-memory list, indexed identically to their embedding counterparts in FAISS.

### Stage 5: Real-Time Query Retrieval

When a user submits a question, it is embedded on-the-fly using the same `gemini-embedding-001` model. The resulting 3072-dimensional query vector is passed to `index.search()`, which returns the **Top-10 nearest neighbours** by L2 distance. Invalid indices (returned as `-1` by FAISS when the index has fewer entries than `top_k`) are filtered out before the chunks are assembled into a context string.

### Stage 6: Grounded Answer Generation (The RAG Loop)

The retrieved chunks are joined with separator markers (`---`) and injected into a structured prompt sent to `gemini-3.1-flash-lite-preview`. The prompt enforces five strict rules on the model:

1. Use **only** the provided context — no external knowledge.
2. Do **not** hallucinate or fabricate information.
3. Structure responses with paragraphs and bullet points for readability.
4. Extract **maximum relevant detail** from the context to fully address the query.
5. Explicitly state when the context does not contain sufficient information.

The answer is rendered in the Streamlit chat interface alongside a collapsible **"View Retrieved Context Sources"** expander showing each retrieved chunk numbered by relevance rank, providing complete auditability of the retrieval process.

---

## Project Structure

```
Enterprise-RAG-QA-System/
│
├── .streamlit/
│   └── config.toml           # Streamlit theme (primary green #0A5C36) and server config
│
├── backend/
│   └── rag_pipeline.py       # EnterpriseRAG class: extraction, chunking, embedding, FAISS, generation
│
├── static/
│   └── style.css             # Custom CSS: chat bubbles, sidebar, typography overrides
│
├── app.py                    # Streamlit entry point: UI, session state, caching, error handling
│
└── requirements.txt          # Production dependencies
```

### File Responsibilities

**`backend/rag_pipeline.py`** — The core intelligence layer. Contains the `EnterpriseRAG` class with four methods:
- `get_embeddings()` — Handles extraction, chunking, and batch embedding. Returns a `(chunks, embeddings)` tuple.
- `add_to_index()` — Adds validated embeddings to the FAISS index and extends the parallel chunk store.
- `retrieve_context()` — Embeds a query, searches FAISS, and returns the top-k matching text chunks.
- `generate_answer()` — Assembles the strict RAG prompt and calls the Gemini generation model.

**`app.py`** — The presentation and orchestration layer. Manages Streamlit session state, the `st.cache_data` caching decorator, file deduplication logic, sidebar UI, the main chat loop, and all error handling.

**`static/style.css`** — Overrides Streamlit's default component styles for chat messages, sidebar, and expanders to apply the enterprise visual identity.

**`.streamlit/config.toml`** — Declares the Streamlit theme (light base, primary green `#0A5C36`) and enables static file serving for CSS injection.

---

## Key Engineering Decisions

### 1. Content-Hash-Based Caching (`st.cache_data`)

The most expensive operation in this system is the embedding API call. The `get_cached_embeddings()` function is decorated with `@st.cache_data`, and critically, it caches on **file bytes content** — not the filename. This means:

- Re-uploading a file with the same name but different content correctly triggers a fresh API call.
- Uploading the same file under a different filename is still served from cache.
- The cache persists across Streamlit reruns within the same deployment, drastically reducing API costs in multi-user or repeated-upload scenarios.

A temporary `EnterpriseRAG` instance is used inside the cached function to isolate the embedding step from the global FAISS index, ensuring the cache returns only pure data (`chunks`, `embeddings`) without side effects.

### 2. FAISS Index & Session State Persistence

The `EnterpriseRAG` engine is instantiated once and stored in `st.session_state.rag_engine`. This architectural choice ensures that the FAISS index, which is held entirely in memory, is not destroyed and rebuilt on every Streamlit UI interaction or rerun. The chat history and document ingestion flag are similarly persisted in session state, enabling multi-turn conversational continuity without re-processing.

### 3. Duplicate Document Guard

A `st.session_state.processed_files` set tracks filenames of all successfully ingested documents. Before processing any uploaded file, the application checks this set and skips files that have already been embedded and added to the FAISS index. This prevents **vector duplication** — a scenario where the same document is indexed twice, which would cause it to appear twice in retrieval results and artificially inflate its influence on generated answers.

### 4. Graceful Rate-Limit Handling (429 Errors)

All API-calling code paths — both the ingestion pipeline and the generation step — are wrapped in `try/except` blocks that specifically detect `429` status codes. Rather than surfacing a raw exception, the UI presents a clear, user-friendly message distinguishing quota exhaustion from unexpected system failures, which is essential for a deployed, multi-user application.

### 5. `IndexFlatL2` for Exact Retrieval Precision

FAISS offers both exact and approximate nearest-neighbour search indices. This system deliberately uses `IndexFlatL2` — an **exact** L2 search — rather than an approximate index such as `IndexIVFFlat`. For an enterprise Q&A system where retrieval accuracy is paramount (a missed or wrong chunk directly degrades answer quality), exact search is the correct tradeoff at the document scales typical of enterprise knowledge bases.

---

## UI & Design System

The interface is built with an **Enterprise-First** aesthetic using a two-layer styling system: Streamlit's native TOML theming for global configuration, and a custom CSS file for component-level overrides.

### Theme Configuration (`.streamlit/config.toml`)

| Property | Value | Purpose |
|---|---|---|
| `base` | `light` | Light mode foundation |
| `primaryColor` | `#0A5C36` | Enterprise green — buttons, sliders, active elements |
| `backgroundColor` | `#FFFFFF` | Main content area |
| `secondaryBackgroundColor` | `#F4F6F8` | Sidebar and secondary surfaces |
| `textColor` | `#1E293B` | High-contrast dark slate for readability |
| `font` | `sans serif` | Clean, professional typeface |
| `enableStaticServing` | `true` | Enables CSS injection from the `static/` directory |

### Chat Message Differentiation (`static/style.css`)

Visual distinction between conversational roles is enforced through left-border colour coding on a shared card treatment:

- **User messages:** `border-left: 4px solid #0A5C36` (enterprise green) on a `#F8FAFC` background
- **Assistant messages:** `border-left: 4px solid #1E88E5` (professional blue) on white

Both message types share a unified card treatment — `border-radius: 10px`, subtle `box-shadow`, and a `1px solid #E2E8F0` border — maintaining visual hierarchy without distraction.

### System Diagnostics Panel

A dedicated sidebar section provides real-time observability into the RAG engine's state:
- **Total Vectors in FAISS:** Live count sourced directly from `index.ntotal`, indicating the current depth and breadth of the active knowledge base.
- **Active LLM:** Displays the exact model identifier powering response generation.

---

## Local Setup & Installation

### Prerequisites

- Python 3.8 or higher
- A valid **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/))

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/Avik-Das-567/Enterprise-RAG-QA-System.git
cd Enterprise-RAG-QA-System
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure your API key:**

Create the Streamlit secrets file:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add your key to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

> ⚠️ **Important:** Never commit `secrets.toml` to version control. Add it to your `.gitignore`.

**5. Run the application:**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Usage

1. In the **sidebar**, upload one or more proprietary PDF documents.
2. Click **"Process & Ingest Documents"** to extract, chunk, embed, and index the content into FAISS.
3. Once the success message confirms the number of vectors added, use the **chat input** to ask natural language questions.
4. Expand **"View Retrieved Context Sources"** beneath any answer to inspect the exact document segments used to generate it.

---

## Configuration

All core RAG parameters are centralized in `rag_pipeline.py` and can be tuned without touching application logic:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `rag_pipeline.py` | `800` | Number of words per text chunk |
| `overlap` | `rag_pipeline.py` | `150` | Word overlap between adjacent chunks |
| `batch_size` | `rag_pipeline.py` | `100` | Max chunks per embedding API call |
| `dimension` | `rag_pipeline.py` | `3072` | Embedding vector dimensionality |
| `top_k` | `rag_pipeline.py` | `10` | Number of chunks retrieved per query |
| `embedding_model` | `rag_pipeline.py` | `gemini-embedding-001` | Embedding model identifier |
| `generation_model` | `rag_pipeline.py` | `gemini-3.1-flash-lite-preview` | Generation model identifier |
| `primaryColor` | `config.toml` | `#0A5C36` | Streamlit UI primary accent colour |

---

## Dependencies

```
streamlit       # Web application framework and UI components
google-genai    # Google Generative AI SDK (Gemini embedding + generation)
faiss-cpu       # Facebook AI Similarity Search — CPU-optimised vector index
numpy           # Numerical arrays for FAISS-compatible float32 vector operations
PyPDF2          # PDF parsing and page-level text extraction
```
