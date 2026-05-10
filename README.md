# Cancer Research Agentic RAG Chatbot

A production-style **Agentic Retrieval-Augmented Generation (RAG) chatbot** built to answer questions from cancer-related PDF documents using grounded retrieval, hybrid search, PII masking, caching, observability, LLM-as-Judge evaluation, and retrieval metrics.

This project is designed as more than a basic PDF chatbot. It demonstrates an end-to-end GenAI architecture that retrieves relevant information from uploaded cancer research and Medicare documents, generates document-grounded answers, displays source citations, and provides transparent debugging information.

---

## Project Overview

The goal of this project is to help users ask natural language questions from cancer-related PDF documents such as:

- Cancer facts and statistics
- Medicare cancer screening coverage
- Cancer treatment costs
- Cancer informatics
- Screening rates
- Risk factors
- Clinical and biomedical data sources

Instead of allowing the LLM to answer from memory, the system first retrieves relevant document chunks and then generates an answer only from the retrieved context.

---

## Problem Statement

Cancer research and healthcare policy documents are often:

- Long and difficult to search manually
- Filled with technical terminology
- Spread across multiple PDFs
- Hard to interpret quickly
- Risky for hallucination if answered only by a general LLM

This chatbot solves that problem by combining:

- Document ingestion
- Semantic search
- Keyword search
- Query rewriting
- Source-grounded generation
- Evaluation
- Observability

---

## Key Features

- PDF ingestion and chunking
- FAISS vector store
- OpenAI embeddings
- Hybrid retrieval using FAISS + BM25
- SLM-based intent classification
- SLM-based query rewriting
- Answer generation using LLM
- PII masking for sensitive user input
- Rewrite cache and answer cache
- LLM-as-Judge evaluation
- Retrieval evaluation metrics
- Token usage and estimated cost tracking
- Source citations with page numbers
- Streamlit frontend
- Debug dashboard
- Live RAG pipeline visualization
- Similarity scores and retrieval source tracking
- Safe fallback for unsupported questions

---

## What is Agentic RAG in This Project?

This project follows an **Agentic RAG-style architecture** because it does more than simple retrieval and answering.

It includes multiple decision-making and orchestration steps:

- Intent classification
- Query rewriting
- Cache decision
- Hybrid retrieval
- Reranking
- Answer generation
- LLM-based evaluation
- Safe fallback behavior

> Note: This project currently implements agentic behavior through modular pipeline logic. Full LangGraph orchestration is planned as future work.

---
## Project Setup

Follow these steps to run the Cancer Research Agentic RAG Chatbot locally.

### 1. Clone the Repository

```bash
git clone https://github.com/KashirajuSateesh/cancer-agentic-rag-chatbot.git
cd cancer-agentic-rag-chatbot
```

### 2. Create and Activate Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the project root folder and add your OpenAI API key.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> Do not commit the `.env` file to GitHub.

### 5. Add PDF Files

Place your PDF documents inside:

```text
data/pdfs/
```

Example:

```text
data/pdfs/cancer-facts-and-figures-2021.pdf
data/pdfs/acscan-medicare-chartbook.pdf
data/pdfs/Informatics at the frontier of cancer research.pdf
```

### 6. Run the Ingestion Pipeline

This step loads PDFs, chunks the text, creates embeddings, and stores them in FAISS.

```bash
python src/ingest.py
```

After successful ingestion, the FAISS vectorstore will be created here:

```text
vectorstore/faiss_index/
```

### 7. Run the Streamlit Chatbot

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

### 8. Run Retrieval Evaluation Metrics

To calculate Precision@K, Recall@K, Hit Rate, MRR, and NDCG:

```bash
python src/evaluate_retrieval.py
```

This uses:

```text
data/eval_questions.json
```

---

## Quick Start: Windows

```bash
git clone https://github.com/KashirajuSateesh/cancer-agentic-rag-chatbot.git
cd cancer-agentic-rag-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Create .env file manually and add:
# OPENAI_API_KEY=your_openai_api_key_here

# Add PDFs inside data/pdfs/

python src/ingest.py
streamlit run app.py
```
---

## Project Architecture

```mermaid
flowchart TD
    A[User Question] --> B[PII Masking]
    B --> C[SLM Intent Classification]

    C -->|Greeting or Meta Question| D[General Chatbot Response]
    C -->|Document Question| E[Answer Cache Check]

    E -->|Cache Hit| F[Return Cached Answer]
    E -->|Cache Miss| G[SLM Query Rewriting]

    G --> H[Hybrid Retrieval]
    H --> H1[FAISS Semantic Search]
    H --> H2[BM25 Keyword Search]

    H1 --> I[Merge and Deduplicate Results]
    H2 --> I

    I --> J[Reranking]
    J --> K[Context Building]

    K --> L[LLM Answer Generation]
    L --> M[LLM-as-Judge Evaluation]

    M --> N[Final Answer with Sources]
    N --> O[Observability Logs and Debug Dashboard]

