"""
app.py — MediQuery Streamlit UI
================================
Features:
  - Two-column layout: answer + reasoning trace side by side
  - Agent reasoning trace with step-by-step node logs
  - Source citations with expandable chunks and source filename
  - Rewrite counter badge when query was rewritten
  - Response time display
  - Query history in sidebar
  - Evaluation metrics panel in sidebar
"""

import streamlit as st
import time
from agent import run_agent

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .mq-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .mq-title {
        font-size: 26px;
        font-weight: 600;
        margin: 0;
        color: inherit;
    }
    .mq-caption {
        font-size: 13px;
        color: #6b7280;
        margin: 0;
    }

    /* Answer card — transparent bg so it works on dark + light themes */
    .answer-card {
        background: rgba(134, 239, 172, 0.08);
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        line-height: 1.7;
        color: inherit;
    }

    /* Trace step */
    .trace-step {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 7px 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 13px;
    }
    .trace-step:last-child { border-bottom: none; }
    .trace-node {
        font-weight: 600;
        min-width: 90px;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 6px;
        text-align: center;
    }
    .node-planner   { background: #ede9fe; color: #5b21b6; }
    .node-retriever { background: #dbeafe; color: #1d4ed8; }
    .node-grader    { background: #fef9c3; color: #92400e; }
    .node-generator { background: #dcfce7; color: #166534; }
    .node-rewriter  { background: #fee2e2; color: #991b1b; }
    .node-default   { background: #f3f4f6; color: #374151; }

    /* Source citation */
    .source-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 4px;
        font-weight: 500;
    }

    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(249, 250, 251, 0.1);
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: inherit;
        margin-right: 6px;
    }

    /* Rewrite warning */
    .rewrite-badge {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        margin-bottom: 10px;
    }

    /* History item */
    .history-item {
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 13px;
        cursor: pointer;
        border: 1px solid #e5e7eb;
        margin-bottom: 6px;
        background: white;
    }
    .history-item:hover { background: #f9fafb; }

    /* Accuracy bar */
    .acc-bar-wrap {
        background: #f3f4f6;
        border-radius: 4px;
        height: 8px;
        margin: 4px 0 12px;
        overflow: hidden;
    }
    .acc-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: #22c55e;
    }

    /* Trace container — dark-theme safe border */
    .trace-container {
        border: 1px solid rgba(229, 231, 235, 0.3);
        border-radius: 10px;
        padding: 12px;
        background: rgba(250, 250, 250, 0.05);
    }

    /* Hide streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "current" not in st.session_state:
    st.session_state.current = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_rewrites" not in st.session_state:
    st.session_state.total_rewrites = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏥 MediQuery")
    st.caption("Agentic RAG · LangGraph · Ollama")
    st.divider()

    # Session metrics
    st.markdown("**Session metrics**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Queries", st.session_state.total_queries)
    with col_b:
        st.metric("Rewrites", st.session_state.total_rewrites)

    if st.session_state.history:
        avg_time = (
            sum(h["time"] for h in st.session_state.history)
            / len(st.session_state.history)
        )
        st.metric("Avg response time", f"{avg_time:.1f}s")

    st.divider()

    # Query history
    if st.session_state.history:
        st.markdown("**Query history**")
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            label = (
                h["query"][:48] + "..."
                if len(h["query"]) > 48
                else h["query"]
            )
            if st.button(
                f"↩ {label}",
                key=f"hist_{i}",
                use_container_width=True
            ):
                st.session_state.current = h

    st.divider()

    # About the agent
    with st.expander("How the agent works"):
        st.markdown("""
**Graph nodes:**
1. **Planner** — routes to retrieval or direct answer
2. **Retriever** — fetches top-5 FAISS chunks
3. **Grader** — filters chunks for relevance
4. **Generator** — synthesizes answer from graded chunks
5. **Rewriter** — rewrites query if no relevant chunks found (max 2×)

Built with LangGraph + Ollama (llama3) + FAISS
        """)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="mq-header">
  <span style="font-size:36px">🏥</span>
  <div>
    <p class="mq-title">MediQuery</p>
    <p class="mq-caption">Clinical AI Assistant · Agentic RAG · LangGraph + Ollama · Local &amp; Private</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Query input ─────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Clinical question",
        placeholder="e.g. What are the treatment guidelines for hypertension?",
        label_visibility="collapsed"
    )
with col_btn:
    ask = st.button("Ask", type="primary", use_container_width=True)

# Example queries
st.caption(
    "Try: &nbsp;"
    + " &nbsp;·&nbsp; ".join([
        "Symptoms of pneumonia",
        "First-line TB treatment",
        "Ibuprofen dosage",
        "What is hypertension?",
        "Diabetes treatment guidelines"
    ])
)

st.divider()

# ─── Run agent ────────────────────────────────────────────────────────────────

result = None

if ask and query.strip():

    with st.spinner("Agent is thinking..."):
        t0      = time.time()
        result  = run_agent(query.strip())
        elapsed = round(time.time() - t0, 2)

    result["time"]  = elapsed
    result["query"] = query.strip()

    st.session_state.history.append(result)
    st.session_state.current        = result
    st.session_state.total_queries  += 1
    st.session_state.total_rewrites += result.get("rewrite_count", 0)

elif st.session_state.current:
    result = st.session_state.current

# ─── Results layout ───────────────────────────────────────────────────────────

if result:

    col_left, col_right = st.columns([3, 2])

    # ── Left: answer + sources ──────────────────────────────────────────────
    with col_left:

        rewrites  = result.get("rewrite_count", 0)
        elapsed   = result.get("time", 0)
        n_sources = len(result.get("sources", []))

        # Metrics row
        st.markdown(
            f'<span class="metric-pill">⏱ {elapsed}s</span>'
            f'<span class="metric-pill">📄 {n_sources} source(s)</span>'
            f'<span class="metric-pill">🔄 {rewrites} rewrite(s)</span>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Rewrite notice
        if rewrites > 0:
            st.markdown(
                f'<div class="rewrite-badge">'
                f'⚠️ Query was rewritten {rewrites} time(s) to improve retrieval quality.'
                f'</div>',
                unsafe_allow_html=True
            )

        # Answer
        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-card">{result["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Sources
        sources = result.get("sources", [])
        is_general = (
            not sources
            or sources == ["General medical knowledge"]
            or sources == ["General medical knowledge (no documents retrieved)"]
        )

        if not is_general:
            st.markdown("#### Source citations")
            for i, src in enumerate(sources):

                src_text = src if isinstance(src, str) else str(src)
                filename = "Document chunk"

                if "Source:" in src_text:
                    parts    = src_text.split("Source:")
                    filename = parts[-1].strip()[:40]
                    src_text = parts[0].strip()

                with st.expander(f"📄 Source {i + 1} — {filename}"):
                    st.markdown(
                        f'<span class="source-badge">Chunk {i + 1}</span>',
                        unsafe_allow_html=True
                    )
                    st.write(
                        src_text[:800]
                        + ("..." if len(src_text) > 800 else "")
                    )
        else:
            st.info(
                "ℹ️ This answer was generated from general medical knowledge "
                "(no document retrieval needed)."
            )

    # ── Right: reasoning trace ───────────────────────────────────────────────
    with col_right:

        st.markdown("#### Agent reasoning trace")

        trace = result.get("trace", [])

        if trace:

            node_map = {
                "[Planner]":   ("node-planner",   "Planner"),
                "[Retriever]": ("node-retriever",  "Retriever"),
                "[Grader]":    ("node-grader",     "Grader"),
                "[Generator]": ("node-generator",  "Generator"),
                "[Rewriter]":  ("node-rewriter",   "Rewriter"),
            }

            trace_html = '<div class="trace-container">'

            for step in trace:

                css_class = "node-default"
                label     = "Agent"
                msg       = step

                for key, (cls, lbl) in node_map.items():
                    if key in step:
                        css_class = cls
                        label     = lbl
                        msg       = step.replace(key, "").strip()
                        break

                trace_html += (
                    f'<div class="trace-step">'
                    f'<span class="trace-node {css_class}">{label}</span>'
                    f'<span style="color:inherit;line-height:1.5">{msg}</span>'
                    f'</div>'
                )

            trace_html += "</div>"

            st.markdown(trace_html, unsafe_allow_html=True)

            # Route + node count summary
            planner_decision = next(
                (s for s in trace if "Route →" in s), None
            )
            if planner_decision:
                route = (
                    "retrieve"
                    if "retrieve" in planner_decision
                    else "direct"
                )
                st.caption(
                    f"Route taken: **{route}** | "
                    f"Nodes visited: **{len(trace)}**"
                )

        else:
            st.info("No trace available for this result.")

        # Final rewritten query (if any)
        rewrites = result.get("rewrite_count", 0)
        if rewrites > 0:
            rewrite_steps = [
                s for s in trace
                if "[Rewriter]" in s and "Rewritten" in s
            ]
            if rewrite_steps:
                st.markdown("**Final rewritten query:**")
                final_rewrite = (
                    rewrite_steps[-1]
                    .replace("[Rewriter]", "")
                    .replace("Rewritten query:", "")
                    .strip()
                )
                st.code(final_rewrite, language=None)

else:

    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#9ca3af">
      <div style="font-size:48px;margin-bottom:1rem">🏥</div>
      <div style="font-size:18px;font-weight:500;color:#6b7280;margin-bottom:8px">
        Ask a clinical question
      </div>
      <div style="font-size:14px">
        MediQuery retrieves from medical documents and shows every reasoning step
      </div>
    </div>
    """, unsafe_allow_html=True)