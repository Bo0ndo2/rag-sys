# 🔍 CV-Matching RAG System

End-to-end Retrieval-Augmented Generation system built with FastAPI, Qdrant, and OpenAI.  
Designed for the NLP Engineering course final project.

---

## 🚀 Quick Start (Docker)

```bash
cp .env.example .env        # Add your OPENAI_API_KEY
docker-compose up --build   # Starts API + Qdrant + MongoDB
```

API is available at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

---

## 📁 Project Structure

```
rag_project/
├── main.py                         # FastAPI app entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── colab_ingest.py                 # Colab notebook for bulk ingestion
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

## 🛠️ Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai` \| `gemini` \| `ollama` |
| `OPENAI_API_KEY` | Your OpenAI key |
| `CHUNK_SIZE` | Tokens per chunk (default: 500) |
| `CHUNK_OVERLAP` | Overlap tokens (default: 50) |
| `TOP_K` | Chunks to retrieve (default: 5) |

---

## 📡 API Endpoints

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

---

## 🎁 Bonus Features

- **+5% LLM Factory Pattern** — Switch between OpenAI / Gemini / Ollama via one config change or per-request `provider` field
- **+10% Arabic Support** — RTL parsing, tashkeel removal, alef normalisation, Arabic prompt template

---

## 🧪 Colab Ingestion (Optional)

For bulk PDF ingestion without local GPU:
1. Open `colab_ingest.py` in Google Colab
2. Set `OPENAI_API_KEY` and `QDRANT_HOST`
3. Upload PDFs and run — vectors are pushed directly to your Qdrant instance
"# rag-sys" 
