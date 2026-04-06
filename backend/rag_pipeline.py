import os
import faiss
import numpy as np
import PyPDF2
from google import genai

class EnterpriseRAG:
    def __init__(self, api_key: str):
        """Initialize the RAG system with Google GenAI client and FAISS index."""
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = 'gemini-embedding-001'
        self.generation_model = 'gemini-3.1-flash-lite-preview'
        self.dimension = 3072
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    def get_embeddings(self, uploaded_file) -> tuple:
        """Extracts text, chunks it, and calls the Gemini API to get embeddings (Uses Batching)."""
        # Extract Text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                clean_text = extracted.encode('utf-8', errors='ignore').decode('utf-8')
                text += clean_text + "\n"

        if not text.strip():
            return [], []

        # Chunk Text
        chunk_size = 800
        overlap = 150
        words = text.split()
        
        new_chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text:
                new_chunks.append(chunk_text)

        # Embed using Batching
        embeddings = []
        if new_chunks:
            batch_size = 100  # Send up to 100 chunks in a single API request
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                
                res = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch
                )
                
                for embedding_item in res.embeddings:
                    embeddings.append(embedding_item.values)
                    
        return new_chunks, embeddings

    def add_to_index(self, new_chunks: list, embeddings: list) -> int:
        """Adds the retrieved chunks and embeddings to the local FAISS index."""
        if not new_chunks or not embeddings:
            return 0
            
        # Convert to numpy array and add to FAISS
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.chunks.extend(new_chunks)
        
        return len(new_chunks)

    def retrieve_context(self, query: str, top_k: int = 10) -> list:
        """Embeds the query and retrieves the most relevant chunks from FAISS."""
        if self.index.ntotal == 0:
            return []

        # Embed query
        res = self.client.models.embed_content(
            model=self.embedding_model,
            contents=query
        )
        query_embedding = np.array([res.embeddings[0].values]).astype('float32')

        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve text chunks
        retrieved_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks) and idx != -1:
                retrieved_chunks.append(self.chunks[idx])
                
        return retrieved_chunks

    def generate_answer(self, query: str, context_chunks: list) -> str:
        """Generates an answer using Gemini 3.1 Flash-Lite based on retrieved context."""
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
        You are an expert Enterprise Assistant. Your task is to provide a highly detailed, comprehensive, and exhaustive answer to the user's question.
        
        RULES:
        1. Use ONLY the provided internal document context below. 
        2. Do not hallucinate or use outside knowledge.
        3. Break down your answer into paragraphs and use bullet points where applicable.
        4. Extract as much relevant detail from the context as possible to fully satisfy the user's query.
        5. If the answer cannot be found in the context, politely state that you do not have enough information.

        CONTEXT:
        {context_str}

        USER QUESTION:
        {query}
        """

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt
        )
        
        return response.text