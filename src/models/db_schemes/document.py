"""MongoDB document schema for storing ingested document metadata."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class DocumentRecord(BaseModel):
    doc_id:      str
    file_name:   str
    title:       str
    author:      str
    page_count:  int
    chunk_count: int
    is_arabic:   bool = False
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_stats: dict = Field(default_factory=dict)
