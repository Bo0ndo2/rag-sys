# CV-Matching RAG System

End-to-end Retrieval-Augmented Generation system built with FastAPI, Qdrant, and Gemini.
Designed for the NLP Engineering course final project.

---

## Quick Start (Docker)

```bash
cp .env.example .env        # Add your GEMINI_API_KEY
docker-compose up --build   # Starts API + Qdrant + MongoDB
```

API is available at **http://localhost:8000**
Swagger docs at **http://localhost:8000/docs**

---

## Project Structure

```
rag_project/
├── main.py                         # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── TECHNICAL_REPORT.md             # Phase 5 documentation
└── src/
    ├── routes/                     # HTTP layer (VIEW in MVC)
    │   ├── ingest.py               # POST /api/v1/ingest
    │   ├── query.py                # POST /api/v1/query
    │   ├── health.py               # GET  /api/v1/health
    │   └── schema/schemas.py       # Pydantic request/response models
    ├── controllers/                # Business logic (CONTROLLER in MVC)
    │   ├── ingest_controller.py    # PDF → parse → chunk → embed → store
    │   └── query_controller.py     # query → embed → search → generate
    ├── helpers/                    # Utility modules
    │   ├── config.py               # Pydantic settings
    │   ├── pdf_parser.py           # PyMuPDF + Arabic normalisation
    │   └── chunker.py              # Sliding-window token chunker
    ├── models/
    │   └── db_schemes/document.py  # MongoDB document schema
    └── stores/                     # Data layer (MODEL in MVC)
        ├── llm/
        │   ├── __init__.py         # LLMFactory (Factory Pattern)
        │   ├── provider/
        │   │   ├── openai_client.py
        │   │   ├── gemini_client.py
        │   │   └── ollama_client.py
        │   └── template/prompts.py # EN + AR system prompts
        └── vectordb/
            ├── __init__.py         # VectorDBFactory
            └── provider/
                └── qdrant_client.py
```

---

## Phase 1 — Data Processing & Vectorization

**Parser:** PyMuPDF extracts text from raw, unstructured PDFs with `sort=True` to preserve reading order. Arabic pages (>40% Arabic Unicode characters) are detected automatically and routed through a normalisation pipeline: tashkeel removal, alef unification (إأآ → ا), teh marbuta → heh, tatweel removal.

**Chunking strategy:** Sliding-window token chunking with chunk_size=500 and overlap=50 (10%). A typical 2-page CV (~600 tokens) yields ~2 chunks — aligning with natural CV sections. Each chunk stores `doc_id`, `file_name`, `page_number`, and `chunk_index` for full traceability.

**Embedding model:** `gemini-embedding-001` — 3072 dimensions, 100+ languages including Arabic, free tier via Gemini API.

**Vectorization:** Embeddings are stored in Qdrant using cosine similarity as the distance metric.

---

## Phase 2 — System Architecture (FastAPI & Docker)

The API follows the MVC pattern with strict separation of concerns:

- `routes/` — HTTP layer only. Accepts requests, returns responses. No business logic.
- `controllers/` — Orchestrates the RAG pipeline. No HTTP awareness.
- `stores/` — Communicates with Qdrant and LLM providers. No routing logic.

Docker Compose runs 3 services connected via a shared `rag_network` bridge:

| Service | Image | Port | Role |
|---|---|---|---|
| `fastapi-app` | Custom Dockerfile | 8000 | RAG API |
| `qdrant` | qdrant/qdrant:v1.9.2 | 6333 | Vector database |
| `mongodb` | mongo:7.0 | 27017 | Document metadata store |

Health checks on Qdrant and MongoDB ensure the API only starts when both dependencies are ready.

---

## Phase 3 — RAG Logic & Retrieval

Query pipeline:

1. User sends a natural-language question to `POST /api/v1/query`
2. Query is embedded using `gemini-embedding-001` (same model as ingestion)
3. Qdrant performs cosine similarity search and returns top-k chunks
4. Retrieved chunks are formatted into a structured context string with filename, page, and score
5. Context is injected into the LLM system prompt
6. LLM generates a grounded answer citing source documents by filename
7. Response returns the answer and all retrieved chunks with similarity scores

Context format injected into the LLM:
```
[Chunk 1 | File: cv_ahmed.pdf | Page: 2 | Score: 0.921]
Python developer with 3 years NLP experience...
```

---

## Phase 4 — Evaluation & Error Analysis

### Edge Case 1 — Scanned Image PDF

Upload a PDF that is image-only (no embedded text layer), then query:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@scanned_cv.pdf"

curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Does this candidate have a Computer Science degree?", "top_k": 5, "lang": "en"}'
```

**Expected failure:** PyMuPDF returns no text → chunks are empty or garbled → retrieval score < 0.4 → LLM says "no information found."

**Root cause:** No embedded text layer in scanned PDFs.

**Fix:** Tesseract OCR fallback when extracted text < 50 characters per page.

---

### Edge Case 2 — Multi-Skill Query (Split Across Chunks)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has experience with both machine learning and mobile development?", "top_k": 3, "lang": "en"}'
```

**Expected failure:** Each skill exists in a different chunk. With `top_k=3`, neither chunk scores high enough individually → LLM cannot find a complete match.

**Root cause:** Sliding-window chunking splits semantically related skills across boundaries.

**Fix:** Increase `top_k` to 8–10, or implement BM25 + dense hybrid retrieval.

---

### Edge Case 3 — Arabic Synonym Mismatch

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في تطوير التطبيقات؟", "top_k": 5, "lang": "ar"}'
```

**Expected failure:** CV uses "تطوير البرمجيات" (software development) instead of "تطوير التطبيقات" (app development). Embedding similarity ~0.71 — below reliable retrieval threshold → LLM hallucinates a generic answer.

**Root cause:** Arabic synonym gap; embedding model treats the two phrases as different concepts.

**Fix:** Arabic query expansion — generate 2–3 synonym variants and take the union of results.

---

## Phase 5 — Technical Documentation

### Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai` \| `gemini` \| `ollama` |
| `GEMINI_API_KEY` | Your Gemini key (free at aistudio.google.com) |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `CHUNK_SIZE` | Tokens per chunk (default: 500) |
| `CHUNK_OVERLAP` | Overlap tokens (default: 50) |
| `TOP_K` | Chunks to retrieve (default: 5) |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | System health check |
| POST | `/api/v1/ingest` | Upload and ingest a PDF |
| POST | `/api/v1/query` | Query the RAG system |

### Example: Ingest

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@cv.pdf"
```

### Example: Query (English)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has experience in NLP?", "top_k": 5, "lang": "en"}'
```

### Example: Query (Arabic)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في Python؟", "top_k": 5, "lang": "ar"}'
```

### Example: Override LLM provider per request

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "provider": "gemini"}'
```

### Docker Deployment

```bash
git clone <repo-url> && cd rag_project
cp .env.example .env
docker-compose up --build
curl http://localhost:8000/api/v1/health
```

Switch provider with one line in `.env`:
```bash
LLM_PROVIDER=openai   # or gemini or ollama
docker-compose restart fastapi-app
```

---

## Bonus Features

### LLM Factory Pattern (+5%)

One unified interface — switch providers with a single `.env` change, zero code changes. All providers implement `BaseLLMClient` with `generate()` and `embed()`. Per-request override supported via `"provider"` field in the query body.

| Provider | Model | Use Case |
|---|---|---|
| `openai` | gpt-4o-mini | Highest quality |
| `gemini` | gemini-2.5-flash | Free tier |
| `ollama` | mistral | Fully offline, no API key |

### Arabic Language Support (+10%)

Detection → normalisation → RTL extraction → Arabic system prompt when `lang=ar`.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في Python؟", "top_k": 5, "lang": "ar"}'
```