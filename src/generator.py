from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def generate_answer(query, context_chunks, max_input_len=512, max_output_len=200):

    if not context_chunks:
        return "No relevant documents found."

    context_text = " ".join(context_chunks[:5])

    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    )

    outputs = model.generate(
        **inputs,
        max_length=max_output_len,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
