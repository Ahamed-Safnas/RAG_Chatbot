from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks
