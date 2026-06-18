from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from retriever import load_retriever

import json
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Gemini LLM Setup
# ------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

llm_chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


def llm_invoke(prompt: str) -> str:
    """Helper to invoke Gemini and return plain text."""
    response = llm_chat.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


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
    confidence_score: int
    confidence_reason: str


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

    response = llm_invoke(PLANNER_PROMPT.format(query=state["query"]))

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

    log(f"[Retriever] Searching vectorstore for: '{query}'")

    retriever = load_retriever()
    docs = retriever.invoke(query)

    log(f"[Retriever] Found {len(docs)} chunks")

    return {
        **state,
        "retrieved_docs": docs
    }


# ------------------------------------------------------------------
# Grader
# ------------------------------------------------------------------

GRADER_PROMPT = PromptTemplate.from_template("""
You are a relevance filter for a medical RAG system.
Your job is to KEEP chunks that are even partially related to the question topic.
Only REJECT chunks that are completely unrelated (e.g. question is about malaria, chunk is about surgery equipment).

Be LIBERAL — when in doubt, keep the chunk.

Question: {query}
Document chunk: {doc}

Respond ONLY with valid JSON, no explanation:
{{"relevant": true}} or {{"relevant": false}}

JSON:""")


def grader_node(state: AgentState) -> AgentState:

    log("[Grader] Scoring retrieved chunks...")

    graded = []

    for doc in state["retrieved_docs"]:

        try:

            safe_content = (
                doc.page_content[:600]
                .replace('"', "'")
                .replace('\n', ' ')
            )

            response = llm_invoke(
                GRADER_PROMPT.format(
                    query=state["query"],
                    doc=safe_content
                )
            )

            raw = response.strip()

            start = raw.find("{")
            end   = raw.rfind("}") + 1

            if start != -1 and end > start:

                json_str = raw[start:end]
                json_str = ''.join(
                    c for c in json_str
                    if ord(c) >= 32
                )

                parsed = json.loads(json_str)

                if parsed.get("relevant"):
                    graded.append(doc)

        except Exception as e:
            log(f"[Grader] Parse error: {e} — keeping chunk by default")
            graded.append(doc)

    if not graded:
        log("[Grader] All chunks rejected — falling back to full retrieved set")
        graded = state["retrieved_docs"]

    log(f"[Grader] {len(graded)}/{len(state['retrieved_docs'])} chunks passed")

    return {
        **state,
        "graded_docs": graded
    }


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

GENERATOR_PROMPT = PromptTemplate.from_template("""
You are MediQuery, a clinical AI assistant. Answer the question using ONLY the context below.
IMPORTANT: Preserve exact drug names, dosages, and clinical terms verbatim from the context.
Do not paraphrase drug names — if the context says 'rifampicin', your answer must say 'rifampicin'.
Be specific. List all relevant drugs, doses, and terms mentioned in the context.
If the context is insufficient, say so clearly — do not hallucinate.

Question: {query}

Context:
{context}

Answer:""")


DIRECT_PROMPT = PromptTemplate.from_template("""
You are MediQuery.

Answer the question briefly using correct clinical terminology.
State clearly if you are uncertain about any detail.

Question:
{query}

Answer:
""")


def generator_node(state: AgentState) -> AgentState:

    log("[Generator] Synthesizing answer...")

    docs  = state.get("graded_docs", [])
    route = state.get("route", "retrieve")

    if route == "direct" or not docs:

        log("[Generator] Using direct answer (no graded docs or direct route)")

        answer = llm_invoke(DIRECT_PROMPT.format(query=state["query"]))
        sources = ["General medical knowledge"]

    else:

        log(f"[Generator] Building answer from {len(docs)} graded doc(s)")

        context = "\n\n".join(
            [
                f"[Chunk {i + 1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )

        answer = llm_invoke(
            GENERATOR_PROMPT.format(
                query=state["query"],
                context=context
            )
        )

        sources = []
        for doc in docs:
            filename = doc.metadata.get("source", "unknown")
            page     = doc.metadata.get("page", "?")
            snippet  = doc.page_content[:300]
            sources.append(
                f"{snippet}\n\nSource: {filename} | Page {page}"
            )

    log("[Generator] Answer ready.")

    return {
        **state,
        "answer":  answer,
        "sources": sources
    }


# ------------------------------------------------------------------
# Confidence Scorer
# ------------------------------------------------------------------

CONFIDENCE_PROMPT = PromptTemplate.from_template("""
You are a medical AI quality checker.
Given a question and an answer, rate the answer's confidence on a scale of 1-10.
Consider: Is the answer specific? Does it use clinical terms? Could it be hallucinated?

Question: {query}

Answer: {answer}

Respond ONLY with a JSON object like this:
{{"score": 8, "reason": "Answer uses specific drug names from context"}}

JSON:""")


def confidence_node(state: AgentState) -> AgentState:

    log("[Confidence] Scoring answer reliability...")

    score  = 5
    reason = "Could not score"

    try:

        response = llm_invoke(
            CONFIDENCE_PROMPT.format(
                query=state["query"],
                answer=state["answer"][:500]
            )
        )

        raw = response.strip()

        start = raw.find("{")
        end   = raw.rfind("}") + 1

        if start != -1 and end > start:

            json_str = raw[start:end]
            json_str = ''.join(
                c for c in json_str
                if ord(c) >= 32
            )

            parsed = json.loads(json_str)
            score  = max(1, min(10, int(parsed.get("score", 5))))
            reason = str(parsed.get("reason", ""))

    except Exception as e:
        log(f"[Confidence] Scoring failed: {e}")

    log(f"[Confidence] Score: {score}/10 — {reason}")

    return {
        **state,
        "confidence_score":  score,
        "confidence_reason": reason
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

    count = state.get("rewrite_count", 0) + 1

    log(f"[Rewriter] Attempt {count}")

    rewritten = llm_invoke(REWRITE_PROMPT.format(query=state["query"]))

    # take first line only — prevents multi-line bullet responses
    rewritten = rewritten.strip().split('\n')[0].strip()

    log(f"[Rewriter] Rewritten query: '{rewritten}'")

    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_count":   count
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

    graph = StateGraph(AgentState)

    graph.add_node("planner",    planner_node)
    graph.add_node("retriever",  retriever_node)
    graph.add_node("grader",     grader_node)
    graph.add_node("generator",  generator_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("rewrite",    rewrite_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "generator": "generator"
        }
    )

    graph.add_edge("retriever", "grader")

    graph.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "generator": "generator",
            "rewrite":   "rewrite"
        }
    )

    graph.add_edge("rewrite",    "retriever")
    graph.add_edge("generator",  "confidence")
    graph.add_edge("confidence", END)

    return graph.compile()


# ------------------------------------------------------------------
# Public Interface
# ------------------------------------------------------------------

_graph = None


def run_agent(query: str) -> dict:

    global _graph, trace_log

    # reset trace for every new query
    trace_log = []

    if _graph is None:
        _graph = build_graph()

    initial_state = {
        "query":             query,
        "rewritten_query":   query,
        "retrieved_docs":    [],
        "graded_docs":       [],
        "answer":            "",
        "sources":           [],
        "rewrite_count":     0,
        "route":             "retrieve",
        "confidence_score":  5,
        "confidence_reason": ""
    }

    final_state = _graph.invoke(initial_state)

    return {
        "answer":            final_state["answer"],
        "sources":           final_state["sources"],
        "trace":             trace_log.copy(),
        "rewrite_count":     final_state["rewrite_count"],
        "confidence_score":  final_state.get("confidence_score", 5),
        "confidence_reason": final_state.get("confidence_reason", ""),
        "query":             query,
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

        print("\nSOURCES:")
        for i, src in enumerate(result["sources"]):
            print(f"  [{i+1}] {src[:120]}...")

        print("\nTRACE:")
        for step in result["trace"]:
            print(" ", step)

        print(f"\nREWRITE COUNT  : {result['rewrite_count']}")
        print(f"CONFIDENCE     : {result['confidence_score']}/10")
        print(f"REASON         : {result['confidence_reason']}")
        print("=" * 60)
