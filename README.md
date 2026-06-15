# 🚀 NVIDIA NIM — RAG Demo with LangChain & Streamlit

A Retrieval-Augmented Generation (RAG) application powered by **NVIDIA NIM** inference endpoints, **LangChain**, and **Streamlit**. Ask questions about US Census PDF reports and get accurate, context-grounded answers from **Meta Llama 3.1 70B**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Sample Questions](#-sample-questions)
- [Configuration](#-configuration)
- [Contributing](#-contributing)

---

## 🧠 Overview

This project demonstrates how to build a **production-ready RAG pipeline** using NVIDIA's NIM (NVIDIA Inference Microservices) platform. Instead of relying on a generic LLM response, the app:

1. Loads real PDF documents (US Census Bureau reports)
2. Splits them into chunks and converts them into vector embeddings
3. Stores embeddings in a **FAISS** in-memory vector store
4. At query time, retrieves the most relevant chunks and passes them to the LLM
5. Returns precise, document-grounded answers

This approach dramatically reduces hallucinations and grounds every answer in real source material.

---

## ⚙️ How It Works

The pipeline works in these stages:

1. **Document Loading** — PyPDFDirectoryLoader reads all PDF files from the us_census folder.
2. **Text Splitting** — RecursiveCharacterTextSplitter breaks documents into 700-character chunks with a 50-character overlap to preserve context at boundaries.
3. **Embedding** — NVIDIAEmbeddings converts each chunk into a dense vector representation.
4. **Vector Store** — FAISS indexes all vectors for fast cosine-similarity search.
5. **Retrieval** — At query time, the retriever finds the top-K most relevant chunks.
6. **Generation** — Retrieved chunks are stuffed into a ChatPromptTemplate and sent to ChatNVIDIA (meta/llama-3.1-70b-instruct) for answer generation.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Meta Llama 3.1 70B via NVIDIA NIM |
| **Embeddings** | NVIDIA NIM Embeddings API |
| **Vector Store** | FAISS (in-memory) |
| **Orchestration** | LangChain |
| **UI** | Streamlit |
| **Document Source** | US Census Bureau PDFs |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
NVIDIA-NIM/
├── app1.py              # Main Streamlit application & RAG pipeline
├── us_census/           # Source PDF documents
│   ├── acsbr-015.pdf    # ACS Brief: Health Insurance Coverage
│   ├── acsbr-016.pdf    # ACS Brief: Poverty Statistics
│   ├── acsbr-017.pdf    # ACS Brief: Income & Earnings
│   └── p70-178.pdf      # Current Population Report
├── .env                 # Environment variables (not committed)
└── README.md            # This file
```

---

## ✅ Prerequisites

- Python **3.10+**
- An **NVIDIA NIM API key** — get one free at [build.nvidia.com](https://build.nvidia.com)
- pip or a virtual environment manager

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akhilkhana/NVIDIA-NIM.git
cd NVIDIA-NIM
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-nvidia-ai-endpoints langchain-community langchain-text-splitters langchain-classic faiss-cpu pypdf python-dotenv
```

### 4. Set up your API key

Create a .env file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_nim_api_key_here
```

**Never commit your .env file.** Add it to .gitignore.

---

## ▶️ Usage

```bash
streamlit run app1.py
```

Open your browser at **http://localhost:8501**, then:

1. Click **"Documents Embedding"** to load the PDFs and build the vector store (takes ~10-30 seconds the first time).
2. Wait for the **"Vector Store DB Is Ready"** confirmation message.
3. Type your question in the text box and press **Enter**.
4. The app will retrieve relevant passages and display a grounded answer.

---

## 💬 Sample Questions

Try asking things like:

- "What percentage of people were uninsured in 2021?"
- "How did poverty rates change between 2019 and 2021?"
- "What is the median household income reported in the census data?"
- "Which states had the highest health insurance coverage rates?"

---

## 🔧 Configuration

You can tune the following parameters in app1.py:

| Parameter | Default | Description |
|---|---|---|
| chunk_size | 700 | Characters per document chunk |
| chunk_overlap | 50 | Overlap between consecutive chunks |
| docs[:30] | 30 | Number of PDF pages to embed |
| model | meta/llama-3.1-70b-instruct | NVIDIA NIM model ID |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/my-feature)
3. Commit your changes (git commit -m 'Add my feature')
4. Push to the branch (git push origin feature/my-feature)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the MIT License.

---

Built with ❤️ using **NVIDIA NIM** · **LangChain** · **Streamlit**
