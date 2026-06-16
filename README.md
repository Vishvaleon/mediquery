# MediQuery — Agentic Clinical RAG System

**The Arch: RAG and Agentic AI Hackathon · IIT Kharagpur · Healthcare Track**
Built with LangGraph, Ollama (Llama 3), FAISS, Streamlit, and Python

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-purple?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-orange?style=flat-square)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?style=flat-square)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

MediQuery is an agentic retrieval-augmented generation (RAG) system for clinical question answering. Rather than performing a single retrieve-and-generate pass, it runs a six-node LangGraph agent that plans the query strategy, retrieves and grades evidence, rewrites failed searches, generates a grounded answer, and self-scores its own confidence — producing a cited, transparency-first response to every clinical question it receives.

All inference runs locally through Ollama, so no patient or query data leaves the user's machine.

---

## Key Features

| Feature | Description |
|---|---|
| Agentic planning | Routes each query to retrieval or direct response based on clinical intent |
| FAISS vector search | Semantic search across 36 WHO, NIH, and clinical reference documents |
| Relevance grading | Filters retrieved chunks so only contextually relevant content reaches the generator |
| Query rewriting | Automatically reformulates failed queries using medical terminology (up to two attempts) |
| Hallucination guard | Scores every answer from 1–10 with an accompanying clinical reasoning explanation |
| Source citations | Every answer references the originating document and page number |
| Reasoning trace | Full step-by-step agent trace exposed in the UI for auditability |
| Local-first design | Runs entirely on Ollama (Llama 3) with no external API calls |

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐
│ Planner │ ── decides: retrieve or answer directly
└────┬────┘
     │
  [retrieve]                    [direct]
     │                              │
     ▼                              │
┌───────────┐                       │
│ Retriever │ ── FAISS top-8 chunks │
└─────┬─────┘                       │
      │                             │
      ▼                             │
┌────────┐   0 chunks pass?         │
│ Grader │ ──────────────────► Rewriter (max 2×)
└───┬────┘                          │
    │ graded chunks                 │
    ▼                               │
┌───────────┐ ◄─────────────────────┘
│ Generator │ ── synthesizes answer using verified clinical terms
└─────┬─────┘
      │
      ▼
┌────────────┐
│ Confidence │ ── scores answer 1–10, flags hallucination risk
└─────┬──────┘
      │
      ▼
   Answer + Sources + Reasoning Trace + Confidence Score
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.2.14 |
| LLM | Ollama Llama 3 (8B, local) |
| Embeddings | pritamdeka/S-PubMedBert-MS-MARCO (biomedical) |
| Vector store | FAISS (CPU) |
| Document loader | LangChain PyPDFLoader |
| Interface | Streamlit 1.37 |
| Language | Python 3.10+ |

---

## Project Structure

```
mediquery/
├── app.py              # Streamlit UI (answer + reasoning trace panel)
├── agent.py            # LangGraph six-node agentic loop
├── retriever.py        # FAISS vector store loader and semantic search
├── ingestor.py         # PDF loader, chunker, embedding pipeline
├── config.py           # Model names, paths, chunking configuration
├── evaluate.py         # Eight-query evaluation suite with category breakdown
├── stress_test.py      # Twenty-query adversarial stress test (four failure modes)
├── debug_check.py      # Vector store keyword coverage diagnostic
├── requirements.txt
├── README.md
├── assets/
│   ├── architecture.png
│   ├── demo_screenshot.png
│   ├── reasoning_trace.png
│   ├── confidence_bar.png
│   └── stress_test_results.txt
└── data/
    ├── raw/             # Source PDFs (gitignored)
    │   └── sources.md   # PDF source URLs for reproducibility
    └── vectorstore/     # FAISS index (gitignored, rebuilt locally)
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/download) installed locally
- 8 GB RAM minimum (16 GB recommended for Llama 3)

### 1. Clone the repository

```bash
git clone https://github.com/Vishvaleon/mediquery.git
cd mediquery
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama model

```bash
ollama pull llama3
```

This downloads approximately 4.7 GB. Starting this early is recommended, as it runs in the background.

### 5. Add medical source documents

Download the clinical PDFs listed in `data/raw/sources.md` and place them in `data/raw/`. The evaluation suite was built using:

- WHO Essential Medicines List
- WHO Malaria Treatment Guidelines 2022
- WHO Tuberculosis Treatment Guidelines
- WHO Cardiovascular Disease Guidelines
- WHO Diabetes Action Plan
- NIH Metformin Drug Monograph
- Ibuprofen Prescribing Information

### 6. Build the vector store

```bash
python ingestor.py
```

### 7. Verify keyword coverage (optional)

```bash
python debug_check.py
```

All eight core clinical keywords should pass before running the application.

### 8. Launch the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Evaluation Results

### Standard Evaluation (8 queries)

```
Overall Accuracy   : 7/8  (87.5%)
Avg Response Time  : 15.2s per query

Category Breakdown:
  Symptoms     2/2  (100%)
  Dosage       1/1  (100%)
  Treatment    2/2  (100%)
  Definition   1/1  (100%)
  Medication   1/2  (50%)
```

```bash
python evaluate.py
```

### Adversarial Stress Test (20 queries, 4 categories)

```
Overall Pass Rate : 15/20  (75%)

Category Breakdown:
  Hallucination Bait      3/5  (60%)
  Keyword Fidelity        5/5  (100%)
  Out-of-Scope             5/5  (100%)
  Adversarial Phrasing     2/5  (40%)
```

```bash
python stress_test.py
```

---

## Key Design Decisions

**LangGraph over LangChain's AgentExecutor.** LangGraph exposes the agent's state machine explicitly — every node, edge, and conditional branch is visible and debuggable. AgentExecutor operates as a black box, which is a poor fit for a healthcare application where auditability is essential.

**S-PubMedBert-MS-MARCO over general-purpose embeddings.** General embedding models do not reliably capture that "myocardial infarction" and "chest pain" are clinically related. A biomedical-tuned embedding model substantially improves retrieval precision for clinical queries.

**A dedicated grading node.** Top-K similarity search can surface chunks that are topically adjacent but contextually wrong — for example, a "malaria treatment" query returning content about mosquito nets. The grader filters these out before they reach the generator.

**A confidence-scoring node.** Hallucination is the primary risk in clinical AI systems. The confidence node requires the model to self-evaluate every answer; responses scoring 5 or below are flagged so users know to verify with a clinician.

---

## Example Queries

```
What is the first-line treatment for tuberculosis?
What are the symptoms of pneumonia?
What is the maximum daily dose of ibuprofen for adults?
What medications treat uncomplicated malaria?
What are the gastrointestinal side effects of metformin?
What lifestyle changes help manage hypertension?
```

---

## Team

| Name | Role | Profile |
|---|---|---|
| Vishva James | Lead — RAG core, architecture | [GitHub](https://github.com/Vishvaleon) · [LinkedIn](https://linkedin.com/in/vishva17) |
| Karishma Sri K | Agentic loop, LangGraph | [GitHub](https://github.com/Karishmasri07) |
| Sujitha V | Data pipeline, UI, deployment | [GitHub](https://github.com/Sujithavenkatraj) |

**Institution:** SRM Madurai College of Engineering and Technology, Sivagangai

---

## Links

- **Live demo:** [MediQuery](https://showing-fade-mowing.ngrok-free.dev/)
- **Repository:** [github.com/Vishvaleon/mediquery](https://github.com/Vishvaleon/mediquery)

---

## License

Released under the [MIT License](LICENSE).

---

<div align="center">
Built for <strong>The Arch: RAG and Agentic AI Hackathon</strong> · IIT Kharagpur · 2025
</div>
