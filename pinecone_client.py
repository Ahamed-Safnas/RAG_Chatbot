import os
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in .env")

# Create a Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to index
def get_index(index_name: str):
    dimension = 1536  # Gemini embedding size, adjust if needed
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
    # Use Index object to interact
    return pc.Index(index_name)

def upsert_chunks(index, vectors: List[List[float]], doc_id: str, raw_chunks: List[str]):
    items = []
    for i, emb in enumerate(vectors):
        items.append({
            "id": f"{doc_id}-{i}",
            "values": emb,
            "metadata": {"doc_id": doc_id, "chunk_id": i, "text": raw_chunks[i]}
        })
    batch = 100
    for start in range(0, len(items), batch):
        index.upsert(items[start:start+batch])

def semantic_search(index, query_embedding: List[float], top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict]:
    flt = {"doc_id": {"$eq": doc_id}} if doc_id else None
    res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=flt)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    return matches
