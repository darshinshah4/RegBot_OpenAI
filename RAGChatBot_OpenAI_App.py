#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import os
import pdfplumber
import openai
import requests
import faiss
import numpy as np
from collections import deque
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


# OpenAI API Key (Set in Streamlit Secrets or ENV variable)
openai.api_key = os.environ.get("OPENAI_API_KEY")


# UI Setup
st.set_page_config(page_title="RegBot - PDF Q&A", layout="wide")
st.title("RegBot - Regulatory Document Chatbot")

# File Upload & Processing
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload your regulatory PDFs", accept_multiple_files=True, type=["pdf"])

chat_history = deque(maxlen=5)  # Stores last 5 queries & responses
vector_store = None  # FAISS vector store

def load_pdfs_with_tables(files):
    """Extracts text & tables from PDFs separately."""
    documents = []
    
    for file in files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text))
                
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                    table_doc = Document(page_content=f"[TABLE DATA]\n{table_text}")
                    documents.append(table_doc)

    return documents

def chunk_documents_with_tables(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into text and table chunks."""
    text_chunks = []
    table_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for doc in documents:
        if "[TABLE DATA]" in doc.page_content:
            table_chunks.append(doc)  # Keep table intact
        else:
            text_chunks.extend(text_splitter.split_documents([doc]))

    return text_chunks + table_chunks

def create_faiss_index(doc_chunks, save_path="faiss_index"):
    """Creates FAISS vector index using OpenAI embeddings."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = embedding_model.embed_documents([doc.page_content for doc in doc_chunks])

    # Convert to FAISS format
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Store documents alongside vectors
    faiss.write_index(index, save_path)
    return index, doc_chunks

if uploaded_files:
    st.sidebar.write("Processing PDFs...")
    documents = load_pdfs_with_tables(uploaded_files)
    chunked_docs = chunk_documents_with_tables(documents)
    vector_store, stored_docs = create_faiss_index(chunked_docs)
    st.sidebar.success(f"{len(uploaded_files)} PDFs processed!")

# Chat Functionality
def search_faiss(query, index_path="faiss_index", top_k=3):
    """Retrieves relevant document chunks from FAISS using OpenAI embeddings."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_embedding = np.array([embedding_model.embed_query(query)])  # Ensure numpy array

    index = faiss.read_index(index_path)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_docs = []

    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(stored_docs) and distances[0][i] < 1.0:  # Prevent out-of-bounds access
            retrieved_docs.append(stored_docs[idx])

    return retrieved_docs

def generate_openai_response(query, retrieved_chunks):
    """Generates a response using GPT-4o-mini, but ONLY from extracted PDFs."""
    
    if not retrieved_chunks:  # No relevant documents found
        return "I don't know. The answer is not in the uploaded PDFs."

    # Extract text from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    
    # Prevent very short or irrelevant contexts
    if len(context.strip()) < 50:  
        return "I don't know. The answer is not in the uploaded PDFs."

    # Stronger system prompt to restrict hallucinations
    prompt = f"""You are an AI assistant for answering regulatory reporting queries. 
You MUST ONLY use the provided PDF context to answer. 
If the answer is not explicitly stated in the PDFs, reply with: "I don't know. The answer is not in the uploaded PDFs."

### Context from PDFs:
{context}

### Question:
{query}

### Answer (only from PDFs):"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a regulatory expert assistant. Only answer using the provided document context."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    # Ensure bot does not provide hallucinated answers
    if "I don't know" in answer or len(answer) < 10:
        return "I don't know. The answer is not in the uploaded PDFs."
    
    # Store response in chat history
    chat_history.append((query, answer))  
    return answer

# -------------------------------
# Chat Interface
# -------------------------------
st.subheader("Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the PDFs...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_chunks = search_faiss(user_input) if vector_store else []

    if retrieved_chunks:  # Ensure valid retrieved documents
        response = generate_openai_response(user_input, retrieved_chunks)
    else:
        response = "I don't know. The answer is not in the uploaded PDFs."

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

