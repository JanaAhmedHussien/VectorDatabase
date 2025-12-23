# ui.py

import streamlit as st
from rag import retrieve, log_feedback
from generator import generate_answer

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 Retrieval-Augmented Generation System")

query = st.text_input("Enter your question:")

if query:
    retrieved_chunks = retrieve(query)
    answer = generate_answer(query, retrieved_chunks)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Answer")
        st.success(answer)

        feedback = st.radio(
            "Was this answer helpful?",
            ["👍 Yes", "👎 No"],
            horizontal=True
        )

        if st.button("Submit Feedback"):
            log_feedback(query, retrieved_chunks, feedback)
            st.info("Feedback recorded. Thank you!")

    with col2:
        st.subheader("Retrieved Context")
        for i, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Chunk {i + 1}"):
                st.write(chunk)
