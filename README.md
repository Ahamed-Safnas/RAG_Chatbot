# RAG Chatbot API

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and query them using natural language, powered by Pinecone for vector search and Gemini for text generation.

## Features

- **Document Upload**: Upload PDF files which are automatically processed and indexed
- **Semantic Search**: Find relevant document chunks based on user queries
- **Contextual Responses**: Generate answers using retrieved context from documents
- **Document Filtering**: Query specific documents by ID
- **API Health Check**: Verify service status

## Technologies

- **Backend**: FastAPI
- **Vector Database**: Pinecone
- **LLM**: Google Gemini
- **Document Processing**: PyPDF2
- **Environment**: Python 3.11

## Environment Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd ahamed-safnas-rag_chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv_RAG
   source venv_RAG/bin/activate  # Linux/Mac
   venv_RAG\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   GEMINI_API_KEY=your_gemini_api_key
   GENERATION_MODEL=gemini-1.5-flash  # Optional
   CHUNK_SIZE=1000  # Optional
   CHUNK_OVERLAP=150  # Optional
   ```

## API Endpoints

### POST `/upload`
Upload a PDF file for processing and indexing.

**Request**: 
- `file`: PDF file to upload

**Response**:
```json
{
  "document_id": "uuid-string",
  "chunks": 15,
  "index": "your-index-name"
}
```

### POST `/chat`
Query the indexed documents.

**Request Body**:
```json
{
  "query": "your question",
  "top_k": 5,
  "document_id": "optional-specific-doc-id"
}
```

**Response**:
```json
{
  "answer": "generated response",
  "sources": [
    {
      "score": 0.92,
      "chunk_id": "doc-id-0",
      "document_id": "doc-id"
    }
  ]
}
```

### GET `/health`
Check API status.

**Response**:
```json
{
  "status": "ok",
  "index": "your-index-name"
}
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs available at `http://localhost:8000/docs`.

## Configuration Options

- `CHUNK_SIZE`: Text chunk size for document processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `GENERATION_MODEL`: Gemini model to use (default: gemini-1.5-flash)

## Notes

- The current implementation uses random embeddings for demonstration. Replace the `embed_texts()` function in `embeddings.py` with actual embedding generation for production use.
- Ensure your Pinecone index dimension matches your embedding model's output dimension (default: 1536 for Gemini embeddings).

