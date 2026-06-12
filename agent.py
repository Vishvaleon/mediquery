from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from retriever import load_retriever
from config import MODEL_NAME

import json


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: List[Document]
    graded_docs: List[Document]
    answer: str
    sources: List[str]
    rewrite_count: int
    route: str


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0
)


# ------------------------------------------------------------------
# Trace Logger
# ------------------------------------------------------------------

trace_log: List[str] = []


def log(msg: str):
    trace_log.append(msg)
    print(msg)


# ------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------

PLANNER_PROMPT = PromptTemplate.from_template("""
You are a router for a medical RAG system.

Rules:

- Return "retrieve" for treatment, diagnosis, symptoms, medications,
  dosage, guidelines, protocols, recommendations, diseases,
  laboratory tests, clinical procedures, or any question that may
  require document evidence.

- Return "direct" ONLY for simple medical concepts and definitions.

Question:
{query}

Respond with ONLY:
retrieve
or
direct
""")


def planner_node(state: AgentState) -> AgentState:

    log("[Planner] Deciding retrieval strategy...")

    response = llm.invoke(
        PLANNER_PROMPT.format(
            query=state["query"]
        )
    )

    route = (
        "retrieve"
        if "retrieve" in response.lower()
        else "direct"
    )

    log(f"[Planner] Route → {route}")

    return {
        **state,
        "route": route,
        "rewritten_query": state["query"]
    }


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------

def retriever_node(state: AgentState) -> AgentState:

    query = (
        state.get("rewritten_query")
        or state["query"]
    )

    log(
        f"[Retriever] Searching vectorstore for: '{query}'"
    )

    retriever = load_retriever()

    docs = retriever.invoke(query)

    log(
        f"[Retriever] Found {len(docs)} chunks"
    )

    return {
        **state,
        "retrieved_docs": docs
    }


# ------------------------------------------------------------------
# Grader
# ------------------------------------------------------------------

GRADER_PROMPT = PromptTemplate.from_template("""
You are a relevance grader.

Question:
{query}

Document:
{doc}

Respond ONLY with valid JSON.

{{"relevant": true}}

or

{{"relevant": false}}
""")


def grader_node(state: AgentState) -> AgentState:

    log(
        "[Grader] Scoring retrieved chunks..."
    )

    graded = []

    for doc in state["retrieved_docs"]:

        try:

            # Sanitize chunk text before sending to LLM
            safe_content = (
                doc.page_content[:600]
                .replace('"', "'")
                .replace('\n', ' ')
            )

            response = llm.invoke(
                GRADER_PROMPT.format(
                    query=state["query"],
                    doc=safe_content
                )
            )

            raw = response.strip()

            # Extract just the JSON object, ignore surrounding text
            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start != -1 and end > start:

                json_str = raw[start:end]

                # Remove any control characters that break JSON
                json_str = ''.join(
                    c for c in json_str
                    if ord(c) >= 32
                )

                parsed = json.loads(json_str)

                if parsed.get("relevant"):
                    graded.append(doc)

        except Exception as e:

            log(
                f"[Grader] Parse error: {e} — skipping chunk"
            )

    log(
        f"[Grader] {len(graded)}/{len(state['retrieved_docs'])} chunks passed"
    )

    return {
        **state,
        "graded_docs": graded
    }


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

GENERATOR_PROMPT = PromptTemplate.from_template("""
You are MediQuery, a clinical AI assistant. Answer the question using ONLY the context below.
Be specific — use exact medical terms, drug names, and clinical keywords from the context.
Do not paraphrase drug names or symptoms. If the context mentions 'rifampicin', say 'rifampicin'.
Cite which chunk supports each point.
If the context is insufficient, say so clearly.

Question: {query}

Context:
{context}

Answer:""")


DIRECT_PROMPT = PromptTemplate.from_template("""
You are MediQuery.

Answer the question briefly.

Question:
{query}

Answer:
""")


def generator_node(state: AgentState) -> AgentState:

    log(
        "[Generator] Synthesizing answer..."
    )

    docs = state.get(
        "graded_docs",
        []
    )

    route = state.get(
        "route",
        "retrieve"
    )

    if route == "direct" or not docs:

        log(
            "[Generator] Using direct answer (no graded docs or direct route)"
        )

        answer = llm.invoke(
            DIRECT_PROMPT.format(
                query=state["query"]
            )
        )

        sources = [
            "General medical knowledge"
        ]

    else:

        log(
            f"[Generator] Building answer from {len(docs)} graded doc(s)"
        )

        context = "\n\n".join(
            [
                f"[Chunk {i + 1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )

        answer = llm.invoke(
            GENERATOR_PROMPT.format(
                query=state["query"],
                context=context
            )
        )

        sources = [
            doc.page_content[:200] + "..."
            for doc in docs
        ]

    log(
        "[Generator] Answer ready."
    )

    return {
        **state,
        "answer": answer,
        "sources": sources
    }


# ------------------------------------------------------------------
# Rewriter
# ------------------------------------------------------------------

REWRITE_PROMPT = PromptTemplate.from_template("""
You are a medical search query optimizer.
Rewrite the query below into ONE short search phrase (5-8 words max).
Use specific medical terminology. Output ONLY the rewritten query, nothing else.
No explanation, no numbering, no alternatives.

Original query: {query}

Rewritten query:""")


def rewrite_node(state: AgentState) -> AgentState:

    count = (
        state.get("rewrite_count", 0)
        + 1
    )

    log(
        f"[Rewriter] Attempt {count}"
    )

    rewritten = llm.invoke(
        REWRITE_PROMPT.format(
            query=state["query"]
        )
    )

    log(
        f"[Rewriter] Rewritten query: '{rewritten.strip()}'"
    )

    return {
        **state,
        "rewritten_query": rewritten.strip(),
        "rewrite_count": count
    }


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------

def route_after_planner(
    state: AgentState
) -> Literal["retriever", "generator"]:

    if state["route"] == "retrieve":
        return "retriever"

    return "generator"


def route_after_grader(
    state: AgentState
) -> Literal["generator", "rewrite"]:

    if (
        not state["graded_docs"]
        and state.get("rewrite_count", 0) < 2
    ):
        return "rewrite"

    return "generator"


# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

def build_graph():

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "retriever",
        retriever_node
    )

    graph.add_node(
        "grader",
        grader_node
    )

    graph.add_node(
        "generator",
        generator_node
    )

    graph.add_node(
        "rewrite",
        rewrite_node
    )

    graph.set_entry_point(
        "planner"
    )

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "generator": "generator"
        }
    )

    graph.add_edge(
        "retriever",
        "grader"
    )

    graph.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "generator": "generator",
            "rewrite": "rewrite"
        }
    )

    graph.add_edge(
        "rewrite",
        "retriever"
    )

    graph.add_edge(
        "generator",
        END
    )

    return graph.compile()


# ------------------------------------------------------------------
# Public Interface
# ------------------------------------------------------------------

_graph = None


def run_agent(query: str) -> dict:

    global _graph, trace_log

    trace_log = []

    if _graph is None:
        _graph = build_graph()

    initial_state = {
        "query": query,
        "rewritten_query": query,
        "retrieved_docs": [],
        "graded_docs": [],
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "route": "retrieve"
    }

    final_state = _graph.invoke(
        initial_state
    )

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "trace": trace_log,
        "rewrite_count": final_state["rewrite_count"]
    }


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":

    queries = [
        "What are the treatment guidelines for hypertension?",
        "What is the recommended dosage of metformin for Type 2 diabetes?",
        "What is homeostasis?"
    ]

    for q in queries:

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(q)

        result = run_agent(q)

        print("\nANSWER:")
        print(result["answer"])

        print("\nTRACE:")
        for step in result["trace"]:
            print(" ", step)

        print(f"\nREWRITE COUNT: {result['rewrite_count']}")
        print("=" * 60)