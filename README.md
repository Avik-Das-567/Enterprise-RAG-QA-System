# Enterprise Knowledge Base Q&A System (RAG)

## Project Overview
This repository implements an industry-level **Retrieval-Augmented Generation (RAG)** system designed to address the inherent limitations of traditional enterprise search and standard Large Language Model (LLM) deployments. Conventional keyword-based search systems often fail to capture semantic intent, requiring exact matches that overlook relevant information. Furthermore, LLMs frequently "hallucinate" when queried about proprietary or non-public internal data that was not included in their original training sets.

By integrating a high-performance vector retrieval engine with generative AI, this application provides a secure, natural language interface for employees to query proprietary documents, receiving accurate and citation-backed responses grounded solely in private enterprise data.

---

## Technical Tech Stack
* **Language Engine:** Python 3.14
* **Generative LLM:** `gemini-3.1-flash-lite-preview` (for low-latency, exhaustive reasoning).
* **Embedding Model:** `gemini-embedding-001` (producing high-density 3072-dimensional vectors).
* **Vector Database:** **FAISS** (Facebook AI Similarity Search) utilizing an `IndexFlatL2` implementation for precise Euclidean distance matching.
* **Frontend:** **Streamlit** with custom CSS and theme-level TOML injection.
* **Processing Libraries:** `PyPDF2` for stream-based text extraction and `NumPy` for optimized vector array manipulation.

---

## Comprehensive System Workflow

The application operates through a modular multi-stage pipeline that manages the lifecycle of enterprise data from initial ingestion to final response generation.

### 1. Data Ingestion & Pre-processing
The workflow begins with the secure upload of proprietary PDF documents. The system employs `PyPDF2` to iterate through document pages, extracting raw text while filtering out ignored encoding errors to ensure data integrity.

### 2. Recursive Semantic Chunking
To preserve context and adhere to model window constraints, extracted text is processed through a recursive chunking strategy. The system splits text into segments of **800 words** with a **150-word overlap**. This overlap is critical; it ensures that semantic relationships and contextual continuity are preserved across chunk boundaries, preventing the loss of information during the retrieval phase.

### 3. Batch-Optimized Embedding & Indexing
To combat API overhead and latency, the system implements a **Batch Processing** strategy. It sends up to **100 chunks in a single API call** to the `gemini-embedding-001` model. The resulting 3072-dimensional embeddings are then added to a local **FAISS L2 index**. This indexing method allows for sub-millisecond similarity searches, ensuring the application scales effectively even as the document repository grows.

### 4. Semantic Discovery (Retrieval)
When a user submits a natural language question, the system converts the query into a vector embedding in real-time. It then performs a similarity search against the FAISS index to identify the **Top-k (10)** most mathematically relevant document segments. These chunks form the "ground truth" context for the LLM.

### 5. Grounded Generation (The RAG Loop)
The retrieved segments are injected into a highly specific "Expert Enterprise Assistant" prompt template. The LLM is governed by strict system instructions: it must generate answers **ONLY** using the provided context and must explicitly state when information is missing, effectively eliminating hallucinations. Finally, the response is rendered in the UI alongside a "View Retrieved Context Sources" expander for full auditability.

---

## Key Engineering Decisions

### API Optimization & Caching
The system utilizes `st.cache_data` to hash file bytes rather than filenames. This ensures that if the same file is uploaded across different sessions or by different users, the application skips the expensive embedding API call and retrieves the vectors directly from the cache, significantly reducing operational costs.

### Session State Persistence
The FAISS vector database, chat history, and ingestion status are maintained within the **Streamlit Session State**. This architectural choice allows for continuous, multi-turn conversations and persistent access to the knowledge base throughout a session without requiring the user to re-process documents after every interaction.

### Performance Monitoring & UI Diagnostics
For production transparency, a dedicated "System Diagnostics" panel provides real-time visibility into the RAG engine:
* **Total Vectors:** Displays the current size of the FAISS index, indicating the depth of the active knowledge base.
* **Active LLM:** Confirms the specific model version powering the response generation.
* **Rate-Limit Resilience:** The code includes 429 error detection to gracefully handle API quota exhaustion.

---

## Project Structure

The project follows a clean separation of concerns, isolating the core RAG logic from the frontend presentation layer:

```
Enterprise-RAG-QA-System/
├── .streamlit/
│       └── config.toml       # Enforces corporate theme (Green #0A5C36) and server settings
├── backend/
│       └── rag_pipeline.py   # Core logic: Extraction, Recursive Chunking, FAISS, and RAG Loop
├── static/
│       └── style.css         # UI Logic: Custom chat bubbles, sidebar styling, and typography
├── requirements.txt
│
└── app.py                    # Streamlit Entry Point: State management, UI rendering, and caching
```

---

## Design Philosophy
The user interface is designed with an "Enterprise-First" aesthetic. Through a custom `style.css` and a `config.toml` file, the application enforces a professional "Light" theme with specific visual cues:
* **Message Distinction:** User messages are grounded with a green border (`#0A5C36`), while AI responses are highlighted in blue (`#1E88E5`) to ensure a clear conversational flow.
* **Information Hierarchy:** Headers and message bubbles are styled for maximum readability, ensuring that citation data remains accessible but non-intrusive.
