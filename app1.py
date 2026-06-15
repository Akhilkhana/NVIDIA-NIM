# ─────────────────────────────────────────────────────────────────
#  RAG (Retrieval-Augmented Generation) App using NVIDIA NIM + LangChain
#  - Loads PDF documents from the ./us_census folder
#  - Converts them into vector embeddings using NVIDIA Embeddings
#  - Stores embeddings in a FAISS vector store for fast similarity search
#  - Answers user questions using Llama 3.1 70B via NVIDIA NIM API
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import os
import time

# NVIDIA AI endpoints for embeddings and chat model
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# Splits large documents into smaller chunks for better embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chain that stuffs retrieved document chunks into the LLM prompt
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Template to structure the prompt sent to the LLM
from langchain_core.prompts import ChatPromptTemplate

# Chain that combines retriever + document chain into a full RAG pipeline
from langchain_classic.chains import create_retrieval_chain

# FAISS: fast in-memory vector store for similarity search
from langchain_community.vectorstores import FAISS

# Loads all PDF files from a directory
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables (NVIDIA_API_KEY) from the .env file
from dotenv import load_dotenv
load_dotenv()

# Set the NVIDIA API key so the SDK can authenticate requests
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")


def vector_embedding():
    """
    Builds the FAISS vector store from PDF documents.
    Only runs once per session — result is cached in st.session_state.
    """
    if "vectors" not in st.session_state:

        # Step 1: Initialize NVIDIA embedding model to convert text → vectors
        st.session_state.embeddings = NVIDIAEmbeddings()

        # Step 2: Load all PDF files from the ./us_census directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()

        # Step 3: Split documents into chunks of 700 characters with 50-char overlap
        # Overlap ensures context isn't lost at chunk boundaries
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=50
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30]  # Use first 30 pages to keep it fast
        )

        # Step 4: Convert chunks into embeddings and store them in FAISS
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


# ── App UI ────────────────────────────────────────────────────────

st.title("Nvidia NIM Demo")

# Initialize the LLM — Llama 3.1 70B hosted on NVIDIA NIM
# Using meta/llama-3.1-70b-instruct (correct model name with version and hyphens)
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Prompt template: instructs the LLM to answer only from the provided context
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}

"""
)

# Text input for the user's question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to trigger document loading and embedding creation
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# When user submits a question, run the full RAG pipeline
if prompt1:
    # Guard: vector store must be built before answering
    if "vectors" not in st.session_state:
        st.error("Please click 'Documents Embedding' first to build the vector store.")
        st.stop()

    # Step 1: Create a chain that passes retrieved docs into the LLM prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Step 2: Set up the retriever to find relevant chunks from FAISS
    retriever = st.session_state.vectors.as_retriever()

    # Step 3: Combine retriever + document chain into a full RAG pipeline
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Step 4: Run the chain and measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)

    # Display the LLM's answer
    st.write(response['answer'])

    # Show the source document chunks used to generate the answer
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
