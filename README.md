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
