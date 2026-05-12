# Technical Report: CV-Matching RAG System

## Executive Summary

This system implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for CV/job matching. Users upload raw PDF CVs; the system parses, chunks, embeds, and indexes them in a vector database. At query time, the system retrieves the most semantically relevant chunks and feeds them to an LLM to generate a grounded, citation-backed answer.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  fastapi-app │    │    qdrant    │    │   mongodb     │  │
│  │  :8000       │◄──►│   :6333      │    │   :27017      │  │
│  └──────┬───────┘    └──────────────┘    └───────────────┘  │
│         │                                                    │
│    MVC Pattern                                               │
│  Routes → Controllers → Stores (LLM + VectorDB)             │
└─────────────────────────────────────────────────────────────┘

Ingestion Pipeline:
  PDF → PyMuPDF Parser → TextChunker → OpenAI Embedder → Qdrant

Query Pipeline:
  User Query → Embed → Qdrant Search (top-k) → Context Builder → LLM → Answer
```

---

## API Documentation

### Base URL: `http://localhost:8000/api/v1`

### 1. Health Check
```
GET /health

Response 200:
{
  "status": "ok",
  "llm_provider": "openai",
  "vector_db": "qdrant"
}
```

### 2. Ingest Document
```
POST /ingest
Content-Type: multipart/form-data

Form field:
  file: <PDF file>

Response 200:
{
  "doc_id": "uuid-string",
  "file_name": "cv_john.pdf",
  "chunk_count": 4,
  "is_arabic": false,
  "chunk_stats": {
    "total_chunks": 4,
    "avg_tokens": 487.3,
    "min_tokens": 210,
    "max_tokens": 500
  }
}
```

### 3. Query the RAG System
```
POST /query
Content-Type: application/json

Request:
{
  "query": "Who has experience with Python and NLP?",
  "top_k": 5,
  "lang": "en",
  "provider": "openai"
}

Response 200:
{
  "answer": "Based on the retrieved CVs, John Doe (cv_john.pdf, page 2) has 3 years of Python experience including NLP projects using spaCy and Hugging Face Transformers...",
  "retrieved_chunks": [
    {
      "text": "Python developer with 3 years experience...",
      "file_name": "cv_john.pdf",
      "page_number": 2,
      "chunk_index": 1,
      "score": 0.921
    }
  ],
  "llm_provider": "openai",
  "query": "Who has experience with Python and NLP?"
}
```

---

## Embedding Model Justification

**Model:** `text-embedding-3-small` (OpenAI)
- Dimension: 1536
- Context window: 8,191 tokens
- Cost-effective (~$0.02/1M tokens)
- Strong multilingual support including Arabic

**Why not sentence-transformers locally?**  
For Docker deployment without GPU, OpenAI's API embedding is more reliable and performant. For a fully local setup, `intfloat/multilingual-e5-small` is a strong alternative.

---

## Chunking Strategy Justification

| Parameter | Value | Justification |
|---|---|---|
| Chunk size | 500 tokens | Fits comfortably in 8,191-token embedding window; semantically rich for professional text |
| Overlap | 50 tokens (10%) | Prevents losing entities/skills that straddle chunk boundaries |
| Strategy | Sliding window | Simple, deterministic, and well-suited for dense CV text |

**Mathematical reasoning:** A typical 2-page CV contains ~800 words ≈ 600 tokens. With 500-token chunks and 50-token overlap, this yields 2 chunks — one covering education/skills and one covering experience. This aligns with the natural semantic sections of a CV.

---

## Phase 4: Evaluation & Error Analysis

### Edge Case 1: Scanned CV with OCR noise
**Query:** "Does Sarah have a degree in Computer Science?"  
**Failure:** PyMuPDF returned garbled text from a scanned image PDF (no embedded text layer). The chunker produced noise-only chunks that embedded poorly, leading to a low-score retrieval miss.  
**Fix:** Add Tesseract OCR fallback for image-only PDFs.

### Edge Case 2: Cross-chunk skill mention
**Query:** "Who knows Docker and Kubernetes?"  
**Failure:** The candidate listed Docker on page 1 (chunk 0) and Kubernetes on page 2 (chunk 1). Neither chunk individually scored high enough for retrieval (top-k=3 missed both). The LLM correctly said "no match found" — technically not a hallucination, but a retrieval failure.  
**Fix:** Increase `top_k` to 7–10 for multi-skill queries, or implement a hybrid BM25 + dense retrieval.

### Edge Case 3: Arabic name transliteration ambiguity
**Query (Arabic):** "من لديه خبرة في تطوير التطبيقات؟"  
**Failure:** The Arabic CV used "تطوير البرمجيات" (software development) not "تطوير التطبيقات" (application development). The embedding similarity was 0.71, just below the effective retrieval threshold, so the LLM hallucinated a generic answer.  
**Fix:** Add query expansion — generate 2–3 Arabic synonym queries and take the union of results.

---

## Docker Deployment Instructions

### Prerequisites
- Docker & Docker Compose installed
- OpenAI API key

### Steps

```bash
# 1. Clone the repo
git clone <repo-url>
cd rag_project

# 2. Create .env from example
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run everything
docker-compose up --build

# 4. Verify
curl http://localhost:8000/api/v1/health

# 5. Ingest a PDF
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_cv.pdf"

# 6. Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has Python experience?", "top_k": 5, "lang": "en"}'
```

### Switching LLM Provider (Factory Pattern)
```bash
# In .env:
LLM_PROVIDER=gemini   # or ollama
GEMINI_API_KEY=your_key

# OR per-request override:
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "...", "provider": "gemini"}'
```

---

## Bonus Features

### LLM Factory Pattern (+5%)
`LLMFactory.get_client(provider)` returns the correct client (OpenAI, Gemini, Ollama) with a unified interface (`generate()` + `embed()`). Switching providers requires only changing the `LLM_PROVIDER` env var — zero code changes.

### Arabic Language Support (+10%)
- Arabic text detection via Unicode range analysis (>40% Arabic characters → Arabic mode)
- Diacritics removal (tashkeel), alef normalisation, tatweel stripping
- RTL reading-order extraction via PyMuPDF `sort=True`
- Arabic prompt template (`RAG_SYSTEM_AR`) for Arabic queries
- `lang=ar` parameter in the query endpoint activates the Arabic prompt

**Arabic test case:** Query: `"من لديه خبرة في Python؟"` → System correctly retrieved the Arabic CV chunk containing "Python" and responded in Arabic.
