# 🏥 MediQuery — Agentic Clinical RAG System

> **The Arch: RAG and Agentic AI Hackathon** · IIT Kharagpur · Healthcare Track
> Built with LangGraph · Ollama (Llama 3) · FAISS · Streamlit · Python

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-purple?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

# What it does

MediQuery is an **Agentic Retrieval-Augmented Generation (RAG)** system designed for clinical question answering.

Instead of using a simple retriever + LLM pipeline, MediQuery uses a **LangGraph workflow** that:

* Plans how to answer a question
* Retrieves relevant medical information
* Grades retrieved chunks
* Rewrites poor queries automatically
* Generates grounded responses
* Displays a full reasoning trace

**Ask a clinical question and receive an evidence-based answer with transparent retrieval steps.**

---

# Features

| Feature             | Description                                             |
| ------------------- | ------------------------------------------------------- |
| 🧠 Agentic Planning | Routes queries through retrieval or direct-answer paths |
| 🔍 FAISS Search     | Semantic search over medical PDF documents              |
| ✅ Relevance Grading | Filters irrelevant retrieved chunks                     |
| 🔄 Query Rewriting  | Automatically improves failed searches                  |
| 📄 Source Tracking  | Preserves source document metadata                      |
| 👁️ Reasoning Trace | Displays graph execution steps                          |
| 🔒 Local Inference  | Runs completely on Ollama                               |

---

# Architecture

```text
User Query
    │
    ▼
┌─────────┐
│ Planner │
└────┬────┘
     │
     ▼
┌───────────┐
│ Retriever │
└─────┬─────┘
      │
      ▼
┌────────┐
│ Grader │
└───┬────┘
    │
    ▼
┌───────────┐
│ Generator │
└─────┬─────┘
      │
      ▼
Answer + Sources + Trace
```

---

# Tech Stack

| Layer           | Technology            |
| --------------- | --------------------- |
| Agent Framework | LangGraph             |
| LLM             | Ollama (Llama 3)      |
| Embeddings      | Sentence Transformers |
| Vector Store    | FAISS                 |
| Document Loader | PyPDFLoader           |
| UI              | Streamlit             |
| Language        | Python 3.10+          |

---

# Project Structure

```text
mediquery/
│
├── app.py
├── agent.py
├── retriever.py
├── ingestor.py
├── config.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── architecture.png
│   ├── demo_screenshot.png
│   └── reasoning_trace.png
│
└── data/
    ├── raw/
    └── vectorstore/
```

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/Vishvaleon/mediquery.git
cd mediquery
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install Ollama

Download:

https://ollama.com/download

Pull the model:

```bash
ollama pull llama3
```

---

# Add Medical PDFs

Place PDF files inside:

```text
data/raw/
```

Example sources:

* WHO Clinical Guidelines
* WHO Essential Medicines List
* CDC Reports
* NIH Medical References

---

# Build Vector Store

```bash
python ingestor.py
```

Expected output:

```text
✓ Ingested PDFs
✓ Created chunks
✓ Saved vector store
```

---

# Run the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Example Queries

```text
What are the symptoms of pneumonia?

What is the first-line treatment for tuberculosis?

What lifestyle changes help manage hypertension?

What are the side effects of metformin?
```

---

# Evaluation

Run:

```bash
python evaluate.py
```

This tests retrieval and answer quality on predefined clinical questions.

---

# Team

| Name           | Role                     |
| -------------- | ------------------------ |
| Vishva James   | Lead / RAG Architecture  |
| Karishma Sri K | LangGraph Agent Workflow |
| Sujitha V      | Data Pipeline & UI       |

Institution:

**SRM Madurai College of Engineering and Technology**

---

# Links

GitHub Repository:

https://github.com/Vishvaleon/mediquery

Live Demo:

https://showing-fade-mowing.ngrok-free.dev/

---

# License

MIT License
