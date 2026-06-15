# 🏥 MediQuery — Agentic Clinical RAG System

> **The Arch: RAG and Agentic AI Hackathon** · IIT Kharagpur · Healthcare Track  
> Built with LangGraph · Ollama (llama3) · FAISS · Streamlit · Python

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-purple?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-orange?style=flat-square)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?style=flat-square)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What it does

MediQuery is a **production-ready Agentic RAG system** for clinical question answering. It goes beyond basic retrieval-augmented generation by implementing a **6-node LangGraph agent** that plans, retrieves, grades, rewrites, generates, and self-scores every answer.

**Ask a clinical question. Get a cited, confidence-scored answer with full reasoning transparency.**

| Feature | Description |
|---|---|
| 🧠 Agentic planning | Routes queries to retrieval or direct answer based on clinical intent |
| 🔍 FAISS vector search | Semantic search over 36 WHO / NIH / clinical PDF documents |
| ✅ Relevance grading | Filters retrieved chunks — only contextually relevant content reaches the generator |
| 🔄 Query rewriting | Auto-rewrites failed queries using medical terminology (max 2 attempts) |
| 🛡️ Hallucination guard | Self-scores every answer 1–10 with a clinical reasoning explanation |
| 📄 Source citations | Every answer cites source document filename and page number |
| 👁️ Reasoning trace | Full agent step-by-step trace visible in the UI |
| 🔒 100% local & private | Runs on Ollama (llama3) — no data leaves your machine |

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐
│ Planner │ ──── decides: retrieve or direct answer
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
│ Generator │ ── synthesizes answer with verbatim clinical terms
└─────┬─────┘
      │
      ▼
┌────────────┐
│ Confidence │ ── scores answer 1–10, flags hallucination risk
└─────┬──────┘
      │
      ▼
   Answer + Sources + Trace + Confidence Score
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.2.14 |
| LLM | Ollama llama3 (8B, local) |
| Embeddings | pritamdeka/S-PubMedBert-MS-MARCO (biomedical) |
| Vector store | FAISS (CPU) |
| Document loader | LangChain PyPDFLoader |
| UI | Streamlit 1.37 |
| Language | Python 3.10+ |

---

## Project structure

```
mediquery/
├── app.py              ← Streamlit UI (two-column: answer + reasoning trace)
├── agent.py            ← LangGraph 6-node agentic loop
├── retriever.py        ← FAISS vector store loader + semantic search
├── ingestor.py         ← PDF loader, chunker, embedding pipeline
├── config.py           ← Model names, paths, chunk settings
├── evaluate.py         ← 8-query evaluation suite with category breakdown
├── stress_test.py      ← 20-query adversarial stress test (5 failure modes)
├── debug_check.py      ← Vectorstore keyword coverage diagnostic
├── requirements.txt
├── README.md
├── assets/
│   ├── architecture.png
│   ├── demo_screenshot.png
│   └── stress_test_results.txt
└── data/
    ├── raw/            ← Drop medical PDFs here (gitignored)
    │   └── sources.md  ← PDF source URLs for reproducibility
    └── vectorstore/    ← FAISS index (gitignored, rebuild locally)
```

---

## Setup & run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- 8 GB RAM minimum (16 GB recommended for llama3)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/mediquery.git
cd mediquery
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
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

> This downloads ~4.7 GB. Start this early — it runs in the background.

### 5. Add medical PDFs

Download clinical PDFs from the sources listed in `data/raw/sources.md` and place them in `data/raw/`.

Recommended minimum (already used in evaluation):
- WHO Essential Medicines List
- WHO Malaria Treatment Guidelines 2022
- WHO Tuberculosis Treatment Guidelines
- WHO Cardiovascular Disease guidelines
- WHO Diabetes Action Plan
- NIH metformin drug monograph
- Ibuprofen prescribing information

### 6. Build the vectorstore

```bash
python ingestor.py
```

### 7. (Optional) Verify keyword coverage

```bash
python debug_check.py
```

All 8 core clinical keywords should show ✅ before running the app.

### 8. Launch the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Evaluation results

### Standard evaluation (8 queries)

```
Overall Accuracy  : 7/8  (87.5%)
Avg Response Time : 15.2s per query

Category Breakdown:
  Symptoms     ██  2/2  (100%)
  Dosage       █   1/1  (100%)
  Treatment    ██  2/2  (100%)
  Definition   █   1/1  (100%)
  Medication   █░  1/2   (50%)
```

Run it yourself:
```bash
python evaluate.py
```

### Adversarial stress test (20 queries, 5 categories)

```
Overall Pass Rate : 15/20  (75%)

Category Breakdown:
  Hallucination bait     ███░  3/4  (75%)
  Out-of-scope           ████  4/4 (100%)
  Specificity stress     ████  4/4 (100%)
  Ambiguous query        ███░  3/4  (75%)
  Multi-hop reasoning    █░░░  1/4  (25%)
```

Run it yourself:
```bash
python stress_test.py
```

---

## Key design decisions

**Why LangGraph over LangChain AgentExecutor?**  
LangGraph gives explicit control over the agent's state machine — every node, edge, and conditional branch is visible and debuggable. AgentExecutor is a black box. For a healthcare system where auditability matters, the graph-based approach is the right call.

**Why S-PubMedBert-MS-MARCO over all-MiniLM?**  
General-purpose embedding models don't understand that "myocardial infarction" and "chest pain" are semantically related. The PubMedBert model was fine-tuned on biomedical literature and dramatically improves retrieval precision for clinical queries.

**Why a grader node?**  
Raw FAISS retrieval returns the top-K most similar chunks — but similar doesn't mean relevant. A query about "malaria treatment" can retrieve chunks about mosquito nets (similar topic, wrong content). The grader filters these out before they pollute the generator's context.

**Why a confidence scorer?**  
Hallucination is the #1 risk in clinical AI. The confidence node forces the LLM to self-evaluate every answer and flag uncertainty explicitly. Low-confidence answers (score ≤5) signal to the user that they should verify with a clinician.

---

## Example queries

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
| Vishva James | Lead / RAG core / Architecture | [GitHub](https://github.com/Vishvaleon) · [LinkedIn](https://linkedin.com/in/vishva17) |
| Member 2 | Agentic loop / LangGraph | GitHub |
| Member 3 | Data pipeline / UI / Deployment | GitHub |

**Institution:** SRM Madurai College of Engineering and Technology, Sivagangai

---

## Live demo

🔗 **[Live Demo](YOUR_DEMO_LINK_HERE)**  
📁 **[GitHub Repository](https://github.com/YOUR_USERNAME/mediquery)**

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built for <strong>The Arch: RAG and Agentic AI Hackathon</strong> · IIT Kharagpur · 2025
</div>