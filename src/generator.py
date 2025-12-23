# generator.py

def generate_answer(query, context_chunks):
    """
    Generate a context-aware answer grounded in retrieved chunks.
    This is an extractive generation approach (no LLM).
    """
    if not context_chunks:
        return "No relevant documents found for your query."

    answer_parts = []

    for chunk in context_chunks[:3]:
        sentences = chunk.split(". ")
        answer_parts.append(". ".join(sentences[:2]))

    answer = ". ".join(answer_parts)

    return answer.strip() + "..."
