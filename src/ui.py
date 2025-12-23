import streamlit as st
from rag import retrieve, log_feedback
from generator import generate_answer

st.set_page_config("RAG System", layout="wide")
st.title("📚 Retrieval-Augmented Generation System")

query = st.text_input("Enter your question:")

if query:
    retrieved = retrieve(query)
    context_texts = [text for _, text in retrieved]

    answer = generate_answer(query, context_texts)

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
            log_feedback(query, retrieved, feedback == "👍 Yes")
            st.info("Feedback saved. System will improve over time.")

    with col2:
        st.subheader("Retrieved Context")
        for i, (_, chunk) in enumerate(retrieved):
            with st.expander(f"Chunk {i + 1}"):
                st.write(chunk)
