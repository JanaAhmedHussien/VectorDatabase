
import streamlit as st
from generator import generate_answer
from rag import retrieve

st.markdown(
    """
    <style>
    .stExpanderHeader {
        font-weight: bold;
        color: #0ff;
    }
    .stSuccess {
        background-color: #001f00;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Optional: page settings
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📚 RAG Document Retrieval Demo")



# User query input
query = st.text_input("Enter your question:")

if query:
    top_chunks = retrieve(query)
    answer = generate_answer(query, top_chunks)

    # Columns layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Answer")
        st.success(answer)

    with col2:
        st.subheader("Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            with st.expander(f"Chunk {i+1}"):
                st.write(chunk)

