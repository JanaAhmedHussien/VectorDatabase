from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def generate_answer(query, context_chunks):
    """
    Generate a free, context-aware answer without NLTK/OpenAI.
    """
    if not context_chunks:
        return "No relevant documents found."
    
    # Take top 3 chunks
    top_chunks = context_chunks[:3]
    
    # Simple summarization: pick first 2 sentences of each chunk
    answer_sentences = []
    for chunk in top_chunks:
        sentences = chunk.split(". ")  # naive sentence splitting
        answer_sentences.extend(sentences[:2])
    
    # Combine into a single answer
    answer = ". ".join(answer_sentences)
    
    # Add ellipsis if truncated
    if len(top_chunks) > 0:
        answer += "..."
    
    return answer

