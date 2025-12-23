# utils.py

def clean_text(text):
    """
    Remove extra spaces and newlines from text.
    """
    return " ".join(text.split())

def chunk_text(text, chunk_size=300):
    """
    Split text into chunks of roughly `chunk_size` words.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
