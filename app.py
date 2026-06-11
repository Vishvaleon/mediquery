import streamlit as st

from agent import run_agent


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="centered"
)

# --------------------------------
# Header
# --------------------------------

st.title("🏥 MediQuery — Clinical RAG Assistant")

st.caption(
    "Ask questions about clinical guidelines and medical documents."
)

# --------------------------------
# Input
# --------------------------------

query = st.text_input(
    "Enter your clinical question:",
    placeholder="e.g. What are symptoms of Type 2 diabetes?"
)

# --------------------------------
# Submit
# --------------------------------

if st.button("Ask") and query:

    with st.spinner(
        "Retrieving and reasoning..."
    ):

        answer = run_agent(query)

    st.markdown("### Answer")

    st.write(answer)