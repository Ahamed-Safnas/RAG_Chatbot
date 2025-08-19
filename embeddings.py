import numpy as np
from typing import List

VECTOR_DIM = 512  # match your Pinecone index dimension

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts with correct dimension.
    Currently returns random vectors for demonstration.
    """
    vectors = []
    for text in texts:
        vec = np.random.rand(VECTOR_DIM).tolist()  # match index dimension
        vectors.append(vec)
    return vectors
