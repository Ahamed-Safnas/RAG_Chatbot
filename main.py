from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import google.api_core.exceptions 
import uuid
import os

from dotenv import load_dotenv
load_dotenv()

from pdf_loader import extract_text_from_pdf
from utils import chunk_text
from embeddings import embed_texts  
from pinecone_client import get_index, upsert_chunks, semantic_search
from gemini_client import answer_with_context  # Only for text generation

app = FastAPI(title="RAG_CHAT-BOT", version="1.0.0")

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME missing in .env")

index = get_index(INDEX_NAME)

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None

class UploadResponse(BaseModel):
    document_id: str
    chunks: int
    index: str

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    out_path = DATA_DIR / f"{doc_id}.pdf"
    with out_path.open("wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(str(out_path))
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    
    vectors = embed_texts(chunks)
    upsert_chunks(index, vectors=vectors, doc_id=doc_id, raw_chunks=chunks)

    return UploadResponse(document_id=doc_id, chunks=len(chunks), index=INDEX_NAME)

from fastapi import HTTPException
import google.api_core.exceptions


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Embed the query
    query_embedding = embed_texts([req.query])[0]

    # Semantic search in Pinecone
    matches = semantic_search(
        index=index,
        query_embedding=query_embedding,
        top_k=req.top_k,
        doc_id=req.document_id
    )

    context_snippets: List[str] = [
        m["metadata"]["text"] for m in matches if "metadata" in m and "text" in m["metadata"]
    ]

    try:
        answer = answer_with_context(req.query, context_snippets)

    except google.api_core.exceptions.ResourceExhausted as e:
        # Show the real Gemini error instead of a generic message
        raise HTTPException(
            status_code=429,
            detail=f"Gemini API returned RESOURCE_EXHAUSTED: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the answer: {str(e)}"
        )

    return JSONResponse({
        "answer": answer,
        "sources": [
            {
                "score": m.get("score"),
                "chunk_id": m.get("id"),
                "document_id": m.get("metadata", {}).get("doc_id")
            }
            for m in matches
        ]
    })


@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME}
 