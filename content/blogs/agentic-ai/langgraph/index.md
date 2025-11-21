---
title: "Building an Agentic Hybrid RAG System: Complete Guide with LangGraph and Gemini"
date: 2025-11-18
draft: false
author: "Saeed Mehrang"
tags: ["LangGraph", "RAG", "Agentic AI", "Gemini", "LangChain", "Hybrid Search", "Docker"]
categories: ["Agentic AI Workflows", "Tutorials", "Machine Learning"]
description: "A complete beginner-friendly guide to building a self-correcting Agentic RAG system using Google's Gemini LLM, LangGraph for orchestration, and Hybrid Search combining FAISS vector search with BM25 keyword retrieval."
summary: "Learn how to build an intelligent RAG application from scratch using **Gemini**, **LangGraph**, and **Hybrid Search** (FAISS + BM25). This comprehensive guide covers repository structure, indexing pipelines, state management, and Docker deployment—everything you need to create your own agentic RAG system."
cover:
    # image: "gemini-hybrid-rag-architecture.png"
    alt: "LangGraph Agentic RAG Architecture with Gemini"
    caption: "Building intelligent, self-correcting RAG systems with Gemini and Hybrid Search"
    relative: true
showToc: true
TocOpen: true
weight: 1
---

## 1. Introduction: Why Build an Agentic RAG System?

Traditional RAG (Retrieval-Augmented Generation) systems have a critical limitation: they blindly trust whatever documents are retrieved. But what if the retrieval fails? What if the retrieved context is irrelevant to the user's question?

This is where **Agentic RAG** comes in. By adding intelligent agents that can evaluate, decide, and self-correct, we create a system that:

- **Self-corrects**: Rewrites queries when initial retrieval fails
- **Routes intelligently**: Falls back to web search when internal knowledge is insufficient
- **Grades relevance**: Uses LLM judgment to filter irrelevant documents before generation

In this comprehensive guide, you'll learn how to build such a system from scratch using:

1. **Gemini** (Google's powerful LLM) for generation, grading, and reasoning
2. **LangGraph** for orchestrating the agentic workflow as a state machine
3. **Hybrid Search** combining FAISS (semantic/vector search) with BM25 (keyword search) for superior retrieval accuracy

This guide is designed for **beginners** with basic Python knowledge. By the end, you'll have a complete, working repository that you can extend with your own documents and use cases.

---

## 2. Complete Repository Structure

Let's start with the full project structure. Each directory has a specific purpose, and we'll explore every component in detail:

```text
agentic-gemini-rag/
├── src/
│   ├── config/
│   │   └── settings.py          # Environment variables and configuration
│   ├── components/
│   │   ├── models.py            # Gemini LLM and Embeddings initialization
│   │   ├── retrieval.py         # Hybrid retriever (FAISS + BM25)
│   │   └── tools.py             # Web search tool
│   ├── ingestion/
│   │   ├── loader.py            # Document loading (PDF, TXT, MD)
│   │   ├── chunking.py          # Text splitting strategies
│   │   └── indexer.py           # Build and save FAISS + BM25 indices
│   ├── nodes/
│   │   ├── retrieve.py          # Retrieval node
│   │   ├── grade.py             # Grading and query rewriting nodes
│   │   └── generate.py          # Answer generation node
│   ├── state.py                 # LangGraph State schema
│   ├── prompts.py               # Prompt templates for grading/rewriting
│   ├── graph.py                 # LangGraph workflow definition
│   └── main.py                  # CLI entry point
├── scripts/
│   └── index_documents.py       # CLI script to index your documents
├── tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_retrieval.py
│   │   └── test_nodes.py
│   └── integration/
│       └── test_full_flow.py
├── data/                        # Place your documents here (PDF, TXT, MD)
├── indices/                     # Saved FAISS and BM25 indices (generated)
├── notebooks/                   # Jupyter notebooks for experimentation
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

### Directory Breakdown

- **`src/config/`**: Centralized configuration management using environment variables
- **`src/components/`**: Reusable components (models, retrievers, tools)
- **`src/ingestion/`**: **NEW** - Complete pipeline for loading, chunking, and indexing documents
- **`src/nodes/`**: Individual LangGraph nodes (each node is one step in the workflow)
- **`src/state.py`**: **NEW** - Defines the state schema that flows through the graph
- **`src/prompts.py`**: **NEW** - Centralized prompt templates for consistency
- **`scripts/`**: **NEW** - Utility scripts for indexing and setup
- **`indices/`**: **NEW** - Persistent storage for FAISS and BM25 indices

---

## 3. Core Components Explained

### 3.1. Configuration Management (`src/config/settings.py`)

First, let's centralize all configuration in one place. This module loads environment variables and provides default values:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

# Retrieval Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# Hybrid Search Weights (vector_weight, bm25_weight)
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))

# Agentic Workflow Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Paths
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDICES_DIR = os.getenv("INDICES_DIR", "./indices")
FAISS_INDEX_PATH = os.path.join(INDICES_DIR, "faiss_index")
BM25_INDEX_PATH = os.path.join(INDICES_DIR, "bm25_index.pkl")
```

**What's happening here:**
- We use `python-dotenv` to load variables from a `.env` file
- All tunable parameters are configurable (chunk sizes, retrieval counts, model names)
- Hybrid search weights allow you to balance vector vs keyword search
- `MAX_RETRIES` prevents infinite loops in the query rewriting cycle

### 3.2. Gemini Integration (`src/components/models.py`)

This module initializes the LLM and embeddings using Google's official LangChain integration:

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config import settings

def get_llm():
    """Initialize Gemini LLM for generation, grading, and routing."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.TEMPERATURE,
        convert_system_message_to_human=True  # Gemini compatibility
    )

def get_embeddings():
    """Initialize Gemini embeddings for vector search."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY
    )
```

**Key points:**
- **`gemini-2.0-flash-exp`**: Fast, efficient model suitable for educational purposes
- **`text-embedding-004`**: Google's latest embedding model with 768 dimensions
- **`temperature=0`**: Deterministic outputs for grading and routing (you want consistency, not creativity)
- **Single API key**: Only `GEMINI_API_KEY` is required (get it from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 3.3. Document Ingestion Pipeline

Before we can retrieve documents, we need to load, chunk, and index them. This is a critical step that beginners often overlook.

#### 3.3.1. Document Loading (`src/ingestion/loader.py`)

This module loads various document formats from the `data/` directory:

```python
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from src.config import settings

def load_documents(data_dir: str = None):
    """Load all documents from the data directory."""
    if data_dir is None:
        data_dir = settings.DATA_DIR

    documents = []
    data_path = Path(data_dir)

    # Load PDFs
    for pdf_file in data_path.glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    # Load text files
    for txt_file in data_path.glob("**/*.txt"):
        loader = TextLoader(str(txt_file))
        documents.extend(loader.load())

    # Load markdown files
    for md_file in data_path.glob("**/*.md"):
        loader = UnstructuredMarkdownLoader(str(md_file))
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} documents from {data_dir}")
    return documents
```

**What this does:**
- Recursively scans the `data/` directory for PDF, TXT, and MD files
- Uses LangChain's document loaders to extract text
- Returns a list of `Document` objects with `.page_content` and `.metadata`

#### 3.3.2. Text Chunking (`src/ingestion/chunking.py`)

Large documents must be split into smaller chunks for effective retrieval:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import settings

def chunk_documents(documents):
    """Split documents into chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks
```

**Why chunking matters:**
- **Chunk size (1000 chars)**: Small enough to be precise, large enough to be meaningful
- **Overlap (200 chars)**: Ensures context isn't lost at chunk boundaries
- **RecursiveCharacterTextSplitter**: Splits on paragraphs first, then sentences, then words

#### 3.3.3. Indexing Pipeline (`src/ingestion/indexer.py`)

Now we create both FAISS and BM25 indices:

```python
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from src.components.models import get_embeddings
from src.config import settings

def build_and_save_indices(documents):
    """Build FAISS and BM25 indices and save them to disk."""

    # Create indices directory
    Path(settings.INDICES_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Build FAISS vector index
    print("Building FAISS vector index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(settings.FAISS_INDEX_PATH)
    print(f"FAISS index saved to {settings.FAISS_INDEX_PATH}")

    # 2. Build BM25 keyword index
    print("Building BM25 keyword index...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = settings.RETRIEVAL_TOP_K

    # Save BM25 index using pickle
    with open(settings.BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 index saved to {settings.BM25_INDEX_PATH}")

    return vectorstore, bm25_retriever

def load_indices():
    """Load pre-built FAISS and BM25 indices from disk."""
    embeddings = get_embeddings()

    # Load FAISS
    vectorstore = FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Load BM25
    with open(settings.BM25_INDEX_PATH, 'rb') as f:
        bm25_retriever = pickle.load(f)

    return vectorstore, bm25_retriever
```

**Critical points:**
- **FAISS persistence**: Uses `.save_local()` to save vectors to disk
- **BM25 persistence**: Uses `pickle` since BM25Retriever doesn't have built-in serialization
- **Two functions**: One for building (run once), one for loading (run every query)

### 3.4. Hybrid Search Retriever (`src/components/retrieval.py`)

Now we combine FAISS and BM25 into a unified hybrid retriever:

```python
from langchain.retrievers import EnsembleRetriever
from src.ingestion.indexer import load_indices
from src.config import settings

def get_hybrid_retriever():
    """Create a hybrid retriever combining FAISS (vector) and BM25 (keyword)."""

    # Load both indices
    vectorstore, bm25_retriever = load_indices()

    # Create vector retriever from FAISS
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVAL_TOP_K}
    )

    # Combine using EnsembleRetriever with configurable weights
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[settings.VECTOR_WEIGHT, settings.BM25_WEIGHT]
    )

    return hybrid_retriever
```

**How Hybrid Search works:**
1. **Vector retriever**: Finds semantically similar documents using cosine similarity
2. **BM25 retriever**: Finds documents with matching keywords using term frequency
3. **EnsembleRetriever**: Combines results using **Reciprocal Rank Fusion (RRF)**
   - RRF formula: `score = sum(1 / (k + rank_i))` for each retriever
   - Documents ranked highly by both retrievers get the best scores
4. **Weights**: You can tune `[0.5, 0.5]` to favor vector or keyword search

---

## 4. The Agentic Workflow: LangGraph State Machine

Now we get to the heart of the system: the agentic workflow orchestrated by LangGraph.

### 4.1. State Schema (`src/state.py`)

First, we define the state that flows through the graph:

```python
from typing import List, TypedDict

class GraphState(TypedDict):
    """
    State schema for the LangGraph workflow.

    Attributes:
        question: Original user question
        generation: Final generated answer
        documents: Retrieved documents (as strings)
        retries: Number of query rewrites attempted
    """
    question: str
    generation: str
    documents: List[str]
    retries: int
```

**Why this matters:**
- **Typed state**: Each node reads and updates this state
- **Retry counter**: Prevents infinite loops (max 2 rewrites)
- **Documents as strings**: We serialize document content for simplicity

### 4.2. Prompt Templates (`src/prompts.py`)

Centralized prompts ensure consistency:

```python
GRADING_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

Retrieved Document:
{document}

User Question:
{question}

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant.

Provide the score with no preamble or explanation. Just answer 'yes' or 'no'."""

REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version optimized for retrieval.

Look at the original question and formulate an improved question that will retrieve better documents from the knowledge base.

Original Question:
{question}

Provide only the improved question with no preamble or explanation."""

GENERATION_PROMPT = """You are an assistant for question-answering tasks.

Use the following retrieved context to answer the question. If you don't know the answer based on the context, say that you don't know.

Question: {question}

Context:
{context}

Answer:"""
```

**Design choices:**
- **Binary grading**: Simple "yes/no" makes parsing easier
- **No preamble**: Reduces token usage and ensures consistent outputs
- **Grounded generation**: Explicitly tells the LLM to use only the provided context

### 4.3. LangGraph Nodes

Each node is a function that takes the state and returns updated state.

#### 4.3.1. Retrieve Node (`src/nodes/retrieve.py`)

```python
from src.components.retrieval import get_hybrid_retriever
from src.state import GraphState

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents using hybrid search."""
    print(f"---RETRIEVE (attempt {state['retries'] + 1})---")
    question = state["question"]

    retriever = get_hybrid_retriever()
    documents = retriever.get_relevant_documents(question)

    # Convert to strings
    state["documents"] = [doc.page_content for doc in documents]
    print(f"Retrieved {len(documents)} documents")

    return state
```

#### 4.3.2. Grade Documents Node (`src/nodes/grade.py`)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.components.models import get_llm
from src.prompts import GRADING_PROMPT
from src.state import GraphState

def grade_documents(state: GraphState) -> GraphState:
    """Grade retrieved documents for relevance."""
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(GRADING_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # Grade each document
    filtered_docs = []
    for doc in documents:
        score = chain.invoke({"document": doc, "question": question}).strip().lower()
        if score == "yes":
            print("  ✓ Document is relevant")
            filtered_docs.append(doc)
        else:
            print("  ✗ Document is not relevant")

    state["documents"] = filtered_docs
    print(f"Filtered to {len(filtered_docs)} relevant documents")

    return state

def decide_to_generate(state: GraphState) -> str:
    """Routing function: generate answer or rewrite query?"""
    if not state["documents"]:
        print("---DECISION: REWRITE QUERY---")
        return "rewrite"
    else:
        print("---DECISION: GENERATE ANSWER---")
        return "generate"
```

**What's happening:**
- Each document is graded individually with the LLM
- Binary "yes/no" responses filter irrelevant documents
- **Routing decision**: If no documents passed grading, rewrite the query

#### 4.3.3. Rewrite Query Node (`src/nodes/grade.py`)

```python
from src.config import settings

def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the query to improve retrieval."""
    print("---REWRITE QUERY---")
    question = state["question"]
    retries = state["retries"]

    # Check max retries
    if retries >= settings.MAX_RETRIES:
        print(f"Max retries ({settings.MAX_RETRIES}) reached. Stopping.")
        state["generation"] = "I don't have enough information to answer this question."
        return state

    # Rewrite using LLM
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    better_question = chain.invoke({"question": question})
    print(f"Rewritten: {better_question}")

    state["question"] = better_question
    state["retries"] = retries + 1

    return state
```

**Key insight:**
- The rewritten question **replaces** the original in state
- Retry counter increments to prevent loops
- After max retries, we give up gracefully

#### 4.3.4. Generate Answer Node (`src/nodes/generate.py`)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.components.models import get_llm
from src.prompts import GENERATION_PROMPT
from src.state import GraphState

def generate(state: GraphState) -> GraphState:
    """Generate final answer using retrieved documents."""
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # Combine documents into context
    context = "\n\n".join(documents)

    # Generate answer
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context})
    state["generation"] = answer

    return state
```

### 4.4. Building the Graph (`src/graph.py`)

Now we connect all nodes into a state machine:

```python
from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes.retrieve import retrieve
from src.nodes.grade import grade_documents, decide_to_generate, rewrite_query
from src.nodes.generate import generate

def create_graph():
    """Build the LangGraph workflow."""

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    # Build graph flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")  # Loop back to retrieval
    workflow.add_edge("generate", END)

    return workflow.compile()
```

**Graph flow visualization:**

```
┌─────────┐
│ START   │
└────┬────┘
     │
     v
┌──────────┐
│ retrieve │ ←──────────┐
└────┬─────┘            │
     │                  │
     v                  │
┌────────────────┐      │
│ grade_documents│      │
└────┬───────────┘      │
     │                  │
     v                  │
  [Decision]            │
     │                  │
   ┌─┴─┐                │
   │   │                │
 yes   no               │
   │   │                │
   │   v                │
   │ ┌──────────────┐   │
   │ │rewrite_query │───┘
   │ └──────────────┘
   │
   v
┌──────────┐
│ generate │
└────┬─────┘
     │
     v
   [END]
```

**Critical concepts:**
1. **Entry point**: Always starts with retrieval
2. **Conditional edge**: Routes to generate or rewrite based on grading results
3. **Loop**: Rewrite → Retrieve creates self-correction cycle
4. **Termination**: Either reaches generate or hits max retries

---

## 5. Running the System

### 5.1. Indexing Script (`scripts/index_documents.py`)

This CLI script builds the indices (run this once before querying):

```python
#!/usr/bin/env python3
"""
Index documents from the data/ directory.
Run this script once to build FAISS and BM25 indices.
"""
from src.ingestion.loader import load_documents
from src.ingestion.chunking import chunk_documents
from src.ingestion.indexer import build_and_save_indices

def main():
    print("=== Starting Document Indexing ===\n")

    # Step 1: Load documents
    documents = load_documents()

    if not documents:
        print("No documents found in data/ directory!")
        return

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Build and save indices
    build_and_save_indices(chunks)

    print("\n=== Indexing Complete ===")
    print("You can now run queries using: python -m src.main")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/index_documents.py
```

### 5.2. Main CLI Entry Point (`src/main.py`)

This is the interactive query interface:

```python
#!/usr/bin/env python3
"""
Main entry point for the Agentic RAG system.
Run queries against your indexed documents.
"""
from src.graph import create_graph
from src.state import GraphState

def main():
    print("=== Agentic RAG with Gemini & Hybrid Search ===\n")

    # Build the graph
    app = create_graph()

    # Interactive query loop
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        # Initialize state
        initial_state: GraphState = {
            "question": question,
            "generation": "",
            "documents": [],
            "retries": 0,
        }

        # Run the graph
        print("\n" + "="*60)
        final_state = app.invoke(initial_state)

        # Display answer
        print("\n" + "="*60)
        print("ANSWER:")
        print(final_state["generation"])
        print("="*60)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python -m src.main
```

---

## 6. Docker Deployment

### 6.1. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data indices

CMD ["python", "-m", "src.main"]
```

### 6.2. docker-compose.yml

```yaml
version: '3.8'

services:
  agentic-rag:
    build: .
    container_name: agentic-gemini-rag
    volumes:
      - ./data:/app/data          # Mount your documents
      - ./indices:/app/indices    # Persist indices
      - ./.env:/app/.env          # Environment variables
    stdin_open: true
    tty: true
    environment:
      - PYTHONUNBUFFERED=1
```

### 6.3. .env.example

```bash
# Gemini API Key (get from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_api_key_here

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

# Retrieval Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=4

# Hybrid Search Weights (must sum to 1.0)
VECTOR_WEIGHT=0.5
BM25_WEIGHT=0.5

# Agentic Workflow
MAX_RETRIES=2
TEMPERATURE=0

# Paths (usually don't need to change)
DATA_DIR=./data
INDICES_DIR=./indices
```

### 6.4. Complete Setup Instructions

**Step 1: Clone and setup**
```bash
git clone <your-repo>
cd agentic-gemini-rag
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Step 2: Add your documents**
```bash
# Place PDF, TXT, or MD files in the data/ directory
cp /path/to/your/documents/*.pdf data/
```

**Step 3: Build and run with Docker**
```bash
# Build the Docker image
docker-compose build

# Run indexing (one-time setup)
docker-compose run agentic-rag python scripts/index_documents.py

# Start interactive query session
docker-compose run agentic-rag
```

**Alternative: Run locally without Docker**
```bash
# Install dependencies
pip install -r requirements.txt

# Index documents
python scripts/index_documents.py

# Run queries
python -m src.main
```

---

## 7. Complete requirements.txt

```txt
# Core LangChain
langchain==0.1.20
langchain-core==0.1.52
langchain-community==0.0.38

# Google Gemini
langchain-google-genai==1.0.1

# Vector Store
faiss-cpu==1.7.4

# Document Loaders
pypdf==3.17.4
unstructured==0.12.5

# Utilities
python-dotenv==1.0.1

# LangGraph
langgraph==0.0.55

# BM25
rank-bm25==0.2.2
```

---

## 8. Testing Your System

For educational purposes, we'll focus on basic manual testing rather than comprehensive unit tests.

### 8.1. Test Your Setup

**Test 1: Verify Gemini API**
```python
# test_gemini.py
from src.components.models import get_llm, get_embeddings

# Test LLM
llm = get_llm()
response = llm.invoke("Say hello!")
print(f"LLM Response: {response.content}")

# Test embeddings
embeddings = get_embeddings()
vector = embeddings.embed_query("test query")
print(f"Embedding dimension: {len(vector)}")
```

**Test 2: Verify Indexing**
```bash
# After running indexing script, check if files exist
ls -lh indices/
# Should see: faiss_index/ and bm25_index.pkl
```

**Test 3: Test Hybrid Retrieval**
```python
# test_retrieval.py
from src.components.retrieval import get_hybrid_retriever

retriever = get_hybrid_retriever()
docs = retriever.get_relevant_documents("your test query")
print(f"Retrieved {len(docs)} documents")
for i, doc in enumerate(docs):
    print(f"\nDoc {i+1}: {doc.page_content[:200]}...")
```

### 8.2. Example Query Session

Here's what a typical session looks like:

```
=== Agentic RAG with Gemini & Hybrid Search ===

Ask a question (or 'quit' to exit): What is LangGraph?

============================================================
---RETRIEVE (attempt 1)---
Retrieved 4 documents
---GRADE DOCUMENTS---
  ✓ Document is relevant
  ✓ Document is relevant
  ✗ Document is not relevant
  ✓ Document is relevant
Filtered to 3 relevant documents
---DECISION: GENERATE ANSWER---
---GENERATE ANSWER---

============================================================
ANSWER:
LangGraph is a framework for building stateful, multi-agent workflows
using language models. It provides a directed graph structure where
nodes represent computational steps and edges define the flow of control...
============================================================

Ask a question (or 'quit' to exit): quit
Goodbye!
```

---

## 9. Extending the System

Now that you have a working foundation, here are ways to extend it:

### 9.1. Add Web Search Fallback

Currently unused, you can add a web search node:

```python
# src/components/tools.py
from langchain_community.tools import DuckDuckGoSearchRun

def get_web_search_tool():
    return DuckDuckGoSearchRun()

# src/nodes/web_search.py
from src.components.tools import get_web_search_tool
from src.state import GraphState

def web_search(state: GraphState) -> GraphState:
    """Perform web search when internal docs insufficient."""
    print("---WEB SEARCH---")
    question = state["question"]

    search = get_web_search_tool()
    results = search.run(question)

    state["documents"] = [results]
    return state
```

Then modify the routing logic to use web search after max retries.

### 9.2. Add Conversation Memory

Track conversation history for multi-turn dialogues:

```python
# Extend GraphState
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retries: int
    chat_history: List[tuple]  # NEW
```

### 9.3. Improve Chunking Strategy

Experiment with semantic chunking instead of fixed-size:

```python
from langchain_experimental.text_splitter import SemanticChunker

def semantic_chunk_documents(documents):
    embeddings = get_embeddings()
    splitter = SemanticChunker(embeddings)
    return splitter.split_documents(documents)
```

### 9.4. Add Document Metadata Filtering

Enhance retrieval with metadata filters (e.g., date, source, author):

```python
vector_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "technical_docs"}  # Filter by metadata
    }
)
```

---

## 10. Key Takeaways and Architecture Summary

### Why This Architecture Works

1. **Hybrid Search**: Combines semantic understanding (FAISS) with keyword precision (BM25) for robust retrieval
2. **Self-Correction**: Query rewriting creates a feedback loop that improves results
3. **Grading**: LLM-based relevance filtering prevents hallucinations from irrelevant context
4. **State Machine**: LangGraph provides clear, debuggable workflow logic
5. **Persistence**: Saving indices enables fast subsequent queries without re-indexing

### Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└───────────────────┬─────────────────────────────────────┘
                    │
                    v
┌─────────────────────────────────────────────────────────┐
│              LangGraph State Machine                    │
│  ┌──────────┐   ┌───────┐   ┌─────────┐   ┌─────────┐ │
│  │ Retrieve │──▶│ Grade │──▶│ Rewrite │──▶│Generate │ │
│  └────┬─────┘   └───┬───┘   └────┬────┘   └────┬────┘ │
│       │             │             │              │      │
└───────┼─────────────┼─────────────┼──────────────┼──────┘
        │             │             │              │
        v             v             v              v
┌────────────┐  ┌─────────┐  ┌─────────┐    ┌─────────┐
│Hybrid      │  │ Gemini  │  │ Gemini  │    │ Gemini  │
│Retriever   │  │ LLM     │  │ LLM     │    │ LLM     │
│(FAISS+BM25)│  │(Grading)│  │(Rewrite)│    │(Generate)│
└────────────┘  └─────────┘  └─────────┘    └─────────┘
```

### When to Use This Architecture

**Ideal for:**
- Internal knowledge bases (company docs, research papers)
- Educational content Q&A
- Technical documentation chatbots
- Personal knowledge management

**Not ideal for:**
- Real-time data (use web search or API integration)
- Extremely large corpora (consider Qdrant/Weaviate instead of FAISS)
- Multi-modal documents (would need vision models)

---

## 11. Troubleshooting Common Issues

### Issue 1: "No module named 'src'"
**Solution:** Run from project root: `python -m src.main` (not `python src/main.py`)

### Issue 2: FAISS index load fails
**Solution:** Ensure you indexed with the same embedding model you're loading with

### Issue 3: BM25 pickle error
**Solution:** Rebuild indices - pickle files aren't compatible across Python versions

### Issue 4: Gemini API quota exceeded
**Solution:** Add rate limiting or use caching for repeated queries

### Issue 5: All documents graded as irrelevant
**Solution:**
- Check chunk quality (may be too small/large)
- Adjust grading prompt to be less strict
- Verify your documents actually contain relevant information

---

## 12. Additional Resources

- **LangGraph Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Gemini API**: [https://ai.google.dev/](https://ai.google.dev/)
- **Hybrid Search Explained**: [Hybrid Search RAG in 10 Minutes (Video)](https://www.youtube.com/watch?v=Hn0UK2l1Nkw)
- **FAISS Documentation**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **BM25 Algorithm**: [https://en.wikipedia.org/wiki/Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

---

## Conclusion

You now have a complete blueprint for building an **Agentic RAG system** with:

✅ **Hybrid Search** (FAISS + BM25) for superior retrieval
✅ **Self-correcting workflow** with query rewriting
✅ **LLM-based grading** to filter irrelevant documents
✅ **Gemini integration** for fast, accurate generation
✅ **Docker deployment** for easy setup
✅ **Persistent indices** for efficient querying

The repository structure is modular and extensible—perfect for learning and experimentation. Clone it, add your documents, and start building intelligent RAG applications!

**Next steps:**
1. Set up the repository following Section 6.4
2. Index your own documents
3. Experiment with different chunk sizes and retrieval parameters
4. Extend with web search, memory, or metadata filtering

Happy building! 🚀
