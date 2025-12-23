from src.rag import retrieve
from src.generator import generate_answer

query = "What is a vector database?"
top_chunks = retrieve(query)
answer = generate_answer(query, top_chunks)
print("Answer:", answer)
