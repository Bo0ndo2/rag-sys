# ============================================================
# RAG Project — Colab Ingestion Script
# ============================================================
# Run this on Google Colab to:
#   1. Install dependencies
#   2. Parse your PDFs
#   3. Chunk & embed them
#   4. Push vectors directly to your running Qdrant instance
#      (or save to a file if Qdrant isn't up yet)
#
# Usage: upload your PDFs to Colab, set the config below, run.
# ============================================================

# ── Cell 1: Install ──────────────────────────────────────────
# !pip install PyMuPDF openai qdrant-client python-dotenv -q

import os, uuid, re, json
from pathlib import Path
from typing import Optional
import fitz                          # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Cell 2: Config ───────────────────────────────────────────
OPENAI_API_KEY   = "sk-YOUR_KEY_HERE"
QDRANT_HOST      = "localhost"        # or your server IP
QDRANT_PORT      = 6333
COLLECTION_NAME  = "cv_chunks"
EMBEDDING_MODEL  = "text-embedding-3-small"
EMBEDDING_DIM    = 1536
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
PDF_FOLDER       = "./pdfs"           # folder with your PDFs

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai_client = OpenAI()
qdrant        = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ── Cell 3: Ensure collection exists ────────────────────────
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME}")

# ── Cell 4: Helpers ──────────────────────────────────────────

def normalise_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = text.replace("ة", "ه").replace("\u0640", "")
    return text

def parse_pdf(path: str) -> dict:
    doc = fitz.open(path)
    pages, is_ar = [], False
    for page in doc:
        raw = page.get_text("text", sort=True).strip()
        if not raw:
            continue
        ar_chars = len(re.findall(r"[\u0600-\u06FF]", raw))
        all_alpha = len(re.findall(r"[A-Za-z\u0600-\u06FF]", raw))
        if all_alpha and ar_chars / all_alpha > 0.4:
            raw = normalise_arabic(raw)
            is_ar = True
        pages.append(raw)
    meta = {**doc.metadata, "page_count": len(doc),
            "file_name": Path(path).name, "is_arabic": is_ar}
    doc.close()
    return {"pages": pages, "metadata": meta}

def chunk_text(pages: list[str], doc_id: str, file_name: str, meta: dict) -> list[dict]:
    chunks, idx = [], 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for pnum, page in enumerate(pages, 1):
        tokens = page.split()
        start  = 0
        while start < len(tokens):
            end  = min(start + CHUNK_SIZE, len(tokens))
            text = " ".join(tokens[start:end])
            if len(text.strip()) >= 30:
                chunks.append({
                    "text": text, "doc_id": doc_id,
                    "file_name": file_name, "page_number": pnum,
                    "chunk_index": idx, **meta,
                })
                idx += 1
            if end == len(tokens):
                break
            start += step
    return chunks

def embed(text: str) -> list[float]:
    return openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=text
    ).data[0].embedding

# ── Cell 5: Ingest all PDFs ──────────────────────────────────
pdf_paths = list(Path(PDF_FOLDER).glob("*.pdf"))
print(f"Found {len(pdf_paths)} PDF(s)")

total_chunks = 0
for pdf_path in pdf_paths:
    print(f"\n→ Processing: {pdf_path.name}")
    parsed   = parse_pdf(str(pdf_path))
    doc_id   = str(uuid.uuid4())
    chunks   = chunk_text(parsed["pages"], doc_id, pdf_path.name, parsed["metadata"])
    print(f"  Chunks: {len(chunks)}  |  Arabic: {parsed['metadata']['is_arabic']}")

    points = []
    for i, chunk in enumerate(chunks):
        vec = embed(chunk["text"])
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=chunk))
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i+1}/{len(chunks)}")

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    total_chunks += len(chunks)
    print(f"  ✓ Upserted {len(chunks)} chunks for {pdf_path.name}")

print(f"\n✅ Done! Total chunks in Qdrant: {total_chunks}")

# ── Cell 6: Quick test query ─────────────────────────────────
TEST_QUERY = "Who has experience in Python and machine learning?"

q_vec = embed(TEST_QUERY)
results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=3)
print(f"\n🔍 Test Query: {TEST_QUERY}\n")
for r in results:
    print(f"Score: {r.score:.3f} | File: {r.payload.get('file_name')} | Page: {r.payload.get('page_number')}")
    print(f"  {r.payload['text'][:200]}...\n")
