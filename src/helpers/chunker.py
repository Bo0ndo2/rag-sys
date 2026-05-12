"""
Chunking Strategy — Phase 1
============================
Strategy: Sliding-window token-based chunking
  - Chunk size  : 500 tokens
  - Overlap     : 50  tokens  (10% of chunk size)

Mathematical justification
--------------------------
  * OpenAI text-embedding-3-small has a context window of 8191 tokens.
    500 tokens per chunk comfortably fits, leaving room for the query.
  * 10% overlap (50 tok) ensures named entities or skill phrases that straddle
    a boundary are captured in at least one of the two adjacent chunks.
  * For a typical 2-page CV (~800 words ≈ 600 tokens) this yields ~2 chunks,
    which is semantically meaningful (e.g. education section vs experience section).
  * Larger chunks (1 000+) reduce retrieval precision; smaller (<200) hurt context
    coherence. 500 tok is the empirically sweet-spot for dense professional text.

Each chunk carries metadata (source doc, page number, chunk index) so the
retrieval result is always traceable back to the original document.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tokeniser shim — approximate word-level tokenisation
# (avoids the tiktoken dependency inside Docker; swap in tiktoken if preferred)
# ---------------------------------------------------------------------------

def _approx_tokenise(text: str) -> list[str]:
    """Split on whitespace and punctuation — ~1 word ≈ 1.3 tokens for English."""
    return re.findall(r"\S+", text)


def _detokenise(tokens: list[str]) -> str:
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text:        str
    doc_id:      str                # unique document identifier
    file_name:   str
    page_number: Optional[int]      # None if derived from full_text, not per-page
    chunk_index: int
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "doc_id":      self.doc_id,
            "file_name":   self.file_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Sliding-window token chunker.

    Parameters
    ----------
    chunk_size    : target tokens per chunk  (default 500)
    chunk_overlap : overlap tokens            (default 50)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step         = chunk_size - chunk_overlap  # stride

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        pages: list[str],
        doc_id: str,
        file_name: str,
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Chunk a list of page strings into overlapping token windows.
        Preserves page attribution whenever possible.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page_num, page_text in enumerate(pages, start=1):
            page_text = page_text.strip()
            if not page_text:
                continue

            tokens = _approx_tokenise(page_text)
            start  = 0

            while start < len(tokens):
                end        = min(start + self.chunk_size, len(tokens))
                chunk_text = _detokenise(tokens[start:end])

                # Skip near-empty chunks (e.g. page with only whitespace)
                if len(chunk_text.strip()) < 30:
                    start += self._step
                    continue

                chunks.append(
                    Chunk(
                        text        = chunk_text,
                        doc_id      = doc_id,
                        file_name   = file_name,
                        page_number = page_num,
                        chunk_index = chunk_index,
                        metadata    = extra_metadata or {},
                    )
                )
                chunk_index += 1

                if end == len(tokens):
                    break
                start += self._step

        return chunks

    # ------------------------------------------------------------------
    # Stats helper (used in the technical report / logging)
    # ------------------------------------------------------------------

    def stats(self, chunks: list[Chunk]) -> dict:
        sizes = [len(_approx_tokenise(c.text)) for c in chunks]
        if not sizes:
            return {}
        return {
            "total_chunks": len(sizes),
            "avg_tokens":   round(sum(sizes) / len(sizes), 1),
            "min_tokens":   min(sizes),
            "max_tokens":   max(sizes),
        }
