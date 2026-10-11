# Google Drive RAG Service (FastAPI + Qdrant + Google Drive OAuth2)

A production-ready Retrieval-Augmented Generation (RAG) backend service that connects directly to Google Drive, extracts and chunks document content (Google Docs, Sheets, Slides, PDFs, Word DOCX, Markdown, Text), embeds them into a Qdrant Vector Database, and provides semantic search and citation-backed question-answering APIs.

---

## 🚀 Key Features

1. **Google Drive OAuth2 Flow**:
   - Web authorization endpoint (`/api/v1/auth/login`) with CSRF state generation and auto-redirect support.
   - Callback handler (`/api/v1/auth/callback`) supporting both GET (browser redirect) and POST (SPA / mobile clients).
   - Automated token persistence, expiration checking, and automatic refresh token rotation.
   - Authentication status inspection (`/api/v1/auth/status`).

2. **Google Drive Ingestion & Content Extraction**:
   - Direct listing and filtering of Google Drive files (`/api/v1/drive/files`).
   - Content extraction for:
     - Google Docs, Slides, and Sheets (via Google Drive export API)
     - PDF documents (`pypdf` extraction)
     - Microsoft Word (`python-docx` extraction)
     - Plain text, Markdown, CSV, and JSON files

3. **Smart Token-Aware Chunking**:
   - Paragraph-boundary and sliding window token chunking with configurable overlap (`TextChunker` using `tiktoken`).

4. **Qdrant Vector Database Integration**:
   - Connection support for local Qdrant server, Qdrant Cloud, or in-memory mode (`:memory:`).
   - Cosine similarity vector indexing.
   - Rich metadata storage and payload filtering by `file_id`.
   - Idempotent re-sync with point clean-up.

5. **Multi-Provider LLM & Semantic Search & Citation-Backed RAG QA**:
   - Support for **OpenRouter** (Claude 3.5 Sonnet, DeepSeek, Llama 3, Mistral, etc.) and **OpenAI** (GPT-4o, GPT-4o-mini).
   - Automatic failover support between OpenRouter and OpenAI.
   - `/api/v1/rag/search`: Semantic vector retrieval returning ranked chunks with similarity scores.
   - `/api/v1/rag/ask`: Full RAG pipeline extracting relevant context, prompting LLM (OpenRouter / OpenAI or fallback extractor), and providing structured source citations.

---

## 📁 Project Architecture

```
├── .env.example              # Environment variables template
├── docker-compose.yml        # Multi-container setup (Qdrant + API)
├── Dockerfile                # Production Docker container image
├── pyproject.toml            # Project packaging & dependencies
├── requirements.txt          # Python dependencies
├── src/
│   └── app/
│       ├── api/
│       │   ├── deps.py       # FastAPI dependency injection
│       │   └── v1/
│       │       ├── auth.py   # Google OAuth2 endpoints
│       │       ├── drive.py  # Drive listing and sync endpoints
│       │       └── rag.py    # Search and Question Answering endpoints
│       ├── core/
│       │   ├── config.py     # Pydantic Settings configuration
│       │   └── logging.py    # Structured logging
│       ├── schemas/
│       │   └── rag.py        # Pydantic data schemas
│       ├── services/
│       │   ├── chunker.py    # Token-aware sliding window chunker
│       │   ├── embedding.py  # OpenAI, OpenRouter & Mock embeddings
│       │   ├── google_auth.py# Google OAuth2 & Token service
│       │   ├── google_drive.py# Drive API client & text extraction
│       │   ├── llm.py         # Unified LLM provider (OpenRouter / OpenAI)
│       │   ├── qdrant_store.py# Qdrant client & vector operations
│       │   └── rag_pipeline.py# Sync, Search, & QA orchestration
│       └── main.py           # FastAPI application entrypoint
└── tests/
    ├── conftest.py           # Pytest fixtures and mocks
    ├── integration/
    │   ├── test_api_endpoints.py # Integration test for all API endpoints
    │   └── test_error_handling.py# Integration error cases
    └── unit/
        ├── test_edge_cases.py    # Chunking & LLM edge cases
        ├── test_google_auth.py   # Auth service unit tests
        ├── test_google_drive.py  # Drive extraction unit tests
        ├── test_rag_pipeline.py  # RAG pipeline unit tests
        └── test_vector_store.py  # Qdrant & Embedding unit tests
```

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.11+ or Docker
- Google Cloud project with Google Drive API enabled and OAuth2 Client credentials
- (Optional) OpenAI API Key for live embeddings and GPT answers (mock provider available out-of-the-box for offline testing)
- (Optional) Qdrant running locally or via Docker

### 2. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your `.env` variables:
```env
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/callback

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=google_drive_rag
QDRANT_IN_MEMORY=false

OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 3. Run Locally with UV / Python

```bash
# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run FastAPI server
uvicorn src.app.main:app --reload --port 8000
```

### 4. Run with Docker Compose

```bash
docker compose up -d
```

Access the interactive Swagger documentation at **`http://localhost:8000/docs`**.

---

## 📡 API Reference

### 1. Authentication (`/api/v1/auth`)

- **`GET /api/v1/auth/login`**:
  Generates the Google OAuth2 consent URL. Pass `redirect=true` to immediately redirect browser.
- **`GET /api/v1/auth/callback?code=...&state=...`**:
  Handles browser redirect from Google after user grants permissions.
- **`POST /api/v1/auth/callback`**:
  Accepts `{ "code": "...", "state": "..." }` payload for Single Page Apps (SPA) or mobile apps.
- **`GET /api/v1/auth/status?user_id=default_user`**:
  Checks if tokens exist and are valid.

### 2. Google Drive (`/api/v1/drive`)

- **`GET /api/v1/drive/files`**:
  Lists accessible files in Google Drive.
- **`GET /api/v1/drive/files/{file_id}`**:
  Returns metadata for a specific file.
- **`POST /api/v1/drive/sync`**:
  Triggers downloading, extracting, chunking, embedding, and storing files into Qdrant.
  ```json
  {
    "user_id": "default_user",
    "folder_id": null,
    "file_ids": null,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "force_resync": true
  }
  ```

### 3. RAG Search & Question Answering (`/api/v1/rag`)

- **`POST /api/v1/rag/search`**:
  Vector similarity search in Qdrant:
  ```json
  {
    "query": "What are our Q4 revenue targets?",
    "top_k": 5,
    "file_ids": ["optional-file-filter"]
  }
  ```
- **`POST /api/v1/rag/ask`**:
  Full question-answering with context retrieval and structured citations:
  ```json
  {
    "question": "Summarize the project architecture outlined in our design doc",
    "top_k": 5
  }
  ```
  **Response:**
  ```json
  {
    "question": "Summarize the project architecture outlined in our design doc",
    "answer": "Based on [File: Architecture_Guide.md], the architecture integrates FastAPI with Qdrant...",
    "citations": [
      {
        "file_id": "1A2B3C...",
        "file_name": "Architecture_Guide.md",
        "chunk_index": 0,
        "score": 0.89,
        "text_snippet": "Google Drive RAG architecture integrates FastAPI...",
        "web_view_link": "https://docs.google.com/..."
      }
    ],
    "confidence_score": 0.89,
    "model_used": "gpt-4o-mini"
  }
  ```

### 4. Health Check

- **`GET /health`**:
  Verifies application status and Qdrant connectivity.

---

## 🧪 Testing

Run unit and integration test suite:

```bash
# Run pytest with code coverage
pytest -v --cov=src --cov-report=term-missing

# Run lint checks with ruff
ruff check src tests
```
