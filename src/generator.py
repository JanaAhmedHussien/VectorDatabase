from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a small free model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(query, context_chunks, max_input_length=512, max_output_length=250):
    
    if not context_chunks:
        return "No relevant documents found."

    # Combine top chunks into a single context string
    context_text = " ".join(context_chunks[:5])  # take top 3 chunks

    # Create the prompt: query + context
    prompt = f"Answer the question based on the context below.\n\nContext: {context_text}\n\nQuestion: {query}\nAnswer:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=max_output_length,
        do_sample=True,  # adds some variation
        top_p=0.95,
        temperature=0.7
    )

    # Decode generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
