# rag.py

import os
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from helpers import clean_text, chunk_text

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "src", "vector_store")
FEEDBACK_FILE = os.path.join(VECTOR_STORE_DIR, "feedback.log")

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Document Loading
# -----------------------------
def load_documents():
    documents = []

    for category in os.listdir(DATA_FOLDER):
        category_path = os.path.join(DATA_FOLDER, category)

        if not os.path.isdir(category_path):
            continue

        for file in os.listdir(category_path):
            if file.endswith(".txt"):
                with open(os.path.join(category_path, file), encoding="utf-8") as f:
                    text = clean_text(f.read())

                chunks = chunk_text(text)
                for chunk in chunks:
                    documents.append((chunk, file, category))

    return documents


# -----------------------------
# Vector Database Creation
# -----------------------------
def create_vector_store(documents):
    embeddings = {}
    metadata = {}

    for idx, (text, filename, category) in enumerate(documents):
        vector = model.encode(text)
        vector = vector / norm(vector)  # normalize

        embeddings[str(idx)] = vector.tolist()
        metadata[str(idx)] = {
            "text": text,
            "source": filename,
            "category": category
        }

    with open(os.path.join(VECTOR_STORE_DIR, "embeddings.json"), "w") as f:
        json.dump(embeddings, f)

    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Vector store created with {len(documents)} chunks.")


# -----------------------------
# Vector Store Loading
# -----------------------------
def load_vector_store():
    with open(os.path.join(VECTOR_STORE_DIR, "embeddings.json")) as f:
        embeddings = json.load(f)

    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    embeddings = {k: np.array(v) for k, v in embeddings.items()}
    return embeddings, metadata


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, k=3):
    embeddings, metadata = load_vector_store()

    query_vec = model.encode(query)
    query_vec = query_vec / norm(query_vec)

    scores = {
        idx: np.dot(query_vec, emb)
        for idx, emb in embeddings.items()
    }

    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [metadata[i]["text"] for i in top_ids]


# -----------------------------
# Self-Learning Layer (BONUS)
# -----------------------------
def log_feedback(query, retrieved_chunks, feedback):
    """
    Store user feedback for future model improvement.
    """
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(f"{query}\t{feedback}\n")


# -----------------------------
# Build Vector Store
# -----------------------------
if __name__ == "__main__":
    docs = load_documents()
    if docs:
        create_vector_store(docs)
    else:
        print("No documents found.")
