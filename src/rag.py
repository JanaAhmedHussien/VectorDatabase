import os
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from helpers import clean_text, chunk_text

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
VECTOR_STORE = os.path.join(BASE_DIR, "src", "vector_store")
EMB_FILE = os.path.join(VECTOR_STORE, "embeddings.json")
META_FILE = os.path.join(VECTOR_STORE, "metadata.json")
FEEDBACK_FILE = os.path.join(VECTOR_STORE, "feedback.log")

os.makedirs(VECTOR_STORE, exist_ok=True)

# -----------------------------
# Models
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Document Loader
# -----------------------------
def load_documents():
    docs = []

    for category in os.listdir(DATA_FOLDER):
        cat_path = os.path.join(DATA_FOLDER, category)
        if not os.path.isdir(cat_path):
            continue

        for file in os.listdir(cat_path):
            if file.endswith(".txt"):
                with open(os.path.join(cat_path, file), encoding="utf-8") as f:
                    text = clean_text(f.read())

                for chunk in chunk_text(text):
                    docs.append((chunk, file, category))

    return docs


# -----------------------------
# Vector Store Creation
# -----------------------------
def build_vector_store(documents):
    embeddings = {}
    metadata = {}

    for idx, (text, source, category) in enumerate(documents):
        vec = embedder.encode(text)
        vec = vec / norm(vec)

        embeddings[str(idx)] = vec.tolist()
        metadata[str(idx)] = {
            "text": text,
            "source": source,
            "category": category
        }

    with open(EMB_FILE, "w") as f:
        json.dump(embeddings, f)

    with open(META_FILE, "w") as f:
        json.dump(metadata, f)

    print(f" Vector store built with {len(documents)} chunks.")


# -----------------------------
# Load Store
# -----------------------------
def load_vector_store():
    with open(EMB_FILE) as f:
        embeddings = json.load(f)

    with open(META_FILE) as f:
        metadata = json.load(f)

    embeddings = {k: np.array(v) for k, v in embeddings.items()}
    return embeddings, metadata


# -----------------------------
# Feedback-Aware Learning
# -----------------------------
def load_feedback_scores():
    scores = {}

    if not os.path.exists(FEEDBACK_FILE):
        return scores

    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            # OLD FORMAT: query\tfeedback
            if len(parts) == 2:
                continue  # skip old entries

            # NEW FORMAT: query\tchunk_id\tfeedback
            _, chunk_id, feedback = parts

            scores.setdefault(chunk_id, 0)
            scores[chunk_id] += 1 if feedback == "YES" else -1

    return scores



# -----------------------------
# Retrieval (SELF-LEARNING)
# -----------------------------
def retrieve(query, k=3):
    embeddings, metadata = load_vector_store()
    feedback_scores = load_feedback_scores()

    q_vec = embedder.encode(query)
    q_vec = q_vec / norm(q_vec)

    scores = {}
    for idx, emb in embeddings.items():
        score = np.dot(q_vec, emb)
        score += 0.15 * feedback_scores.get(idx, 0)  # learning effect
        scores[idx] = score

    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]

    return [(i, metadata[i]["text"]) for i in top_ids]


# -----------------------------
# Feedback Logger
# -----------------------------
def log_feedback(query, retrieved_chunks, is_helpful: bool):
    label = "YES" if is_helpful else "NO"

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        for chunk_id, _ in retrieved_chunks:
            f.write(f"{query}\t{chunk_id}\t{label}\n")


# -----------------------------
# Build Store (Run Once)
# -----------------------------
if __name__ == "__main__":
    docs = load_documents()
    if docs:
        build_vector_store(docs)
    else:
        print(" No documents found.")
