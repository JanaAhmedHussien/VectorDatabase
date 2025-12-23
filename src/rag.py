import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from helpers import clean_text, chunk_text
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")

def load_documents(base_folder=DATA_FOLDER):
    documents = []
    for category in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(category_path, filename), "r", encoding="utf-8") as f:
                        text = f.read()
                    text = clean_text(text)
                    chunks = chunk_text(text, chunk_size=300)
                    for chunk in chunks:
                        documents.append((chunk, filename, category))
    return documents


# Create embeddings and save to JSON
def create_vector_store(documents):
    embeddings = {}
    metadata = {}
    for i, (text, filename, category) in enumerate(documents):
        emb = model.encode(text).tolist()
        embeddings[str(i)] = emb
        metadata[str(i)] = {"text": text, "source": filename, "category": category}
    # Save JSON
    os.makedirs("vector_store", exist_ok=True)
    with open("vector_store/embeddings.json", "w") as f:
        json.dump(embeddings, f)
    with open("vector_store/metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Vector store created with {len(documents)} chunks!")

# Load embeddings and metadata
def load_vector_store():
    with open("vector_store/embeddings.json", "r") as f:
        embeddings = json.load(f)
    with open("vector_store/metadata.json", "r") as f:
        metadata = json.load(f)
    # Convert embeddings to numpy arrays
    embeddings = {k: np.array(v) for k, v in embeddings.items()}
    return embeddings, metadata

# Retrieve top-k similar chunks
def retrieve(query, k=3):
    embeddings, metadata = load_vector_store()
    query_emb = model.encode(query)
    scores = {}
    for idx, emb in embeddings.items():
        sim = np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
        scores[idx] = sim
    # Get top-k
    top_idxs = sorted(scores, key=scores.get, reverse=True)[:k]
    top_chunks = [metadata[i]["text"] for i in top_idxs]
    return top_chunks

# Example usage:
if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found! Check data/raw folder.")
    else:
        create_vector_store(docs)
        print("Vector store created successfully!")