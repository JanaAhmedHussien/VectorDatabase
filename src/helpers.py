import re

def clean_text(text: str) -> str:
    """Normalize text by removing extra spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 300):
    """Split text into word-based chunks."""
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
