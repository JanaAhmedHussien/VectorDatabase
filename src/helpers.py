# helpers.py

import re

def clean_text(text):
    """
    Normalize text by removing extra spaces and special characters.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=300):
    """
    Split text into chunks of fixed word length.
    """
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
