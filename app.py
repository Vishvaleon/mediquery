import streamlit as st
from agent import run_agent


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------
# Header
# --------------------------------

st.title("🏥 MediQuery")

st.caption(
    "Agentic Clinical RAG — powered by LangGraph + Ollama"
)

st.divider()


# --------------------------------
# Layout
# --------------------------------

col1, col2 = st.columns([2, 1])


# --------------------------------
# Left Column — Query + Results
# --------------------------------

with col1:

    query = st.text_input(
        "Clinical question:",
        placeholder="e.g. What are treatment guidelines for hypertension?"
    )

    ask_clicked = st.button(
        "Ask MediQuery",
        use_container_width=True
    )

    if ask_clicked and query:

        with st.spinner("Agent is thinking..."):
            result = run_agent(query)

        st.session_state["result"] = result

    if "result" in st.session_state and st.session_state["result"]:

        result = st.session_state["result"]

        # ---- Answer ----

        st.markdown("### 💬 Answer")
        st.write(result["answer"])

        # ---- Rewrite Badge ----

        if result["rewrite_count"] > 0:
            st.info(
                f"🔄 Query was rewritten **{result['rewrite_count']}** time(s) "
                "to improve retrieval."
            )

        # ---- Sources ----

        st.markdown("### 📄 Sources Used")

        if result["sources"]:

            for i, src in enumerate(result["sources"]):

                with st.expander(f"Source {i + 1}"):
                    st.write(src)

        else:

            st.caption("No document sources — answer drawn from general medical knowledge.")


# --------------------------------
# Right Column — Reasoning Trace
# --------------------------------

with col2:

    st.markdown("### 🧠 Agent Reasoning Trace")

    if "result" in st.session_state and st.session_state["result"]:

        result = st.session_state["result"]

        for step in result["trace"]:

            # Color-code by node label

            if "[Planner]" in step:
                st.success(step)

            elif "[Retriever]" in step:
                st.info(step)

            elif "[Grader]" in step:
                st.warning(step)

            elif "[Rewriter]" in step:
                st.error(step)

            elif "[Generator]" in step:
                st.success(step)

            else:
                st.code(step, language=None)

    else:

        st.caption(
            "Run a query to see the agent's step-by-step reasoning here."
        )