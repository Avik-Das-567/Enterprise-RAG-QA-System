import streamlit as st
import os
import io
from backend.rag_pipeline import EnterpriseRAG

# Page Configuration & Static Assets

st.set_page_config(page_title="Enterprise KB System", page_icon="🏢", layout="wide")

# Inject Custom CSS from static folder
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

load_css("static/style.css")

# Initialization & Session State

# Fetch API Key securely from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize RAG Engine in session state so vector DB persists across reruns
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = EnterpriseRAG(api_key=api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_ingested" not in st.session_state:
    st.session_state.document_ingested = False

# We track processed file names so we don't accidentally duplicate vectors in FAISS
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Streamlit Caching for API Optimization

@st.cache_data(show_spinner=False)
def get_cached_embeddings(file_name: str, file_bytes: bytes, _api_key: str):
    """
    Caches the expensive API embedding process. Streamlit uses the file_bytes 
    to hash the content. If the same file is uploaded again, it skips the API call!
    """
    file_obj = io.BytesIO(file_bytes)
    # Temporary engine just to run the extraction without touching the global FAISS index
    temp_engine = EnterpriseRAG(api_key=_api_key)
    return temp_engine.get_embeddings(file_obj)

# UI: Sidebar (Knowledge Base Management)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942789.png", width=60)
    st.title("Knowledge Base Setup")
    st.markdown("Upload proprietary enterprise documents to build your local Vector DB (FAISS).")
    
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Ingest Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents and generating embeddings..."):
                total_chunks_added = 0
                
                for file in uploaded_files:
                    # Prevent re-adding the exact same file to the FAISS index
                    if file.name in st.session_state.processed_files:
                        continue
                        
                    try:
                        # Fetch chunks & embeddings (Uses Streamlit Cache!)
                        chunks, embeddings = get_cached_embeddings(file.name, file.getvalue(), api_key)
                        
                        # Add to FAISS Vector DB
                        chunks_added = st.session_state.rag_engine.add_to_index(chunks, embeddings)
                        total_chunks_added += chunks_added
                        
                        st.session_state.processed_files.add(file.name)
                        
                    except Exception as e:
                        # Graceful Error Handling for Document Uploads
                        if "429" in str(e):
                            st.error(f"⚠️ Rate limit reached while processing {file.name}. The daily free-tier limit may be exhausted.")
                        else:
                            st.error(f"An unexpected error occurred processing {file.name}: {e}")
                
                if total_chunks_added > 0:
                    st.session_state.document_ingested = True
                    st.success(f"✅ Knowledge Base updated! Added {total_chunks_added} new vectors.")
                elif len(st.session_state.processed_files) > 0:
                    st.info("Documents are already processed and active in the Knowledge Base.")
                else:
                    st.warning("Could not extract text from the provided PDFs.")
        else:
            st.error("Please upload at least one PDF first.")

    st.divider()
    st.markdown("### System Diagnostics")
    st.metric(label="Total Vectors in FAISS", value=st.session_state.rag_engine.index.ntotal)
    st.metric(label="Active LLM", value="Gemini 3.1 Flash-Lite")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# UI: Main Chat Interface

st.title("🏢 Enterprise Knowledge Base Q&A")
st.markdown("Ask natural language questions about your uploaded proprietary documents. Powered by **Gemini 3.1 Flash-Lite** and **FAISS**.")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("🔍 View Retrieved Context Sources"):
                for idx, chunk in enumerate(message["context"]):
                    st.markdown(f"**Source {idx+1}:**\n{chunk}")

# Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.document_ingested:
        st.warning("⚠️ Please upload and process a document in the sidebar before asking questions.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching Knowledge Base & Generating Answer..."):
                try:
                    # Retrieve & Generate
                    retrieved_chunks = st.session_state.rag_engine.retrieve_context(prompt)
                    answer = st.session_state.rag_engine.generate_answer(prompt, retrieved_chunks)
                    
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Retrieved Context Sources"):
                        if retrieved_chunks:
                            for idx, chunk in enumerate(retrieved_chunks):
                                st.markdown(f"**Source {idx+1}:**\n{chunk}")
                        else:
                            st.write("No relevant context found in the uploaded documents.")
                            
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "context": retrieved_chunks
                    })
                    
                except Exception as e:
                    # Graceful Error Handling for Chat Generation
                    if "429" in str(e):
                        st.error("⚠️ The daily free-tier API limit for this portfolio project has been reached. Please try again tomorrow!")
                    else:
                        st.error(f"An unexpected error occurred while generating the answer: {e}")