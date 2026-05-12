"""API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id:      str
    file_name:   str
    chunk_count: int
    is_arabic:   bool
    chunk_stats: dict


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:    str  = Field(..., min_length=3, description="User question")
    top_k:    int  = Field(5,  ge=1, le=20,  description="Number of chunks to retrieve")
    lang:     str  = Field("en", pattern="^(en|ar)$", description="Response language: en or ar")
    provider: Optional[str] = Field(None, description="Override LLM provider for this request")


class RetrievedChunk(BaseModel):
    text:        str
    file_name:   str
    page_number: Optional[int]
    chunk_index: int
    score:       float


class QueryResponse(BaseModel):
    answer:          str
    retrieved_chunks: list[RetrievedChunk]
    llm_provider:    str
    query:           str


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    llm_provider: str
    vector_db:   str
