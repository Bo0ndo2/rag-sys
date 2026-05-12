
import uuid
from pathlib import Path

from src.helpers.pdf_parser import PDFParser
from src.helpers.chunker import TextChunker
from src.helpers.config import get_settings
from src.routes.schema.schemas import IngestResponse

settings = get_settings()


class IngestController:

    def __init__(self, vdb, llm):
        self._vdb     = vdb
        self._llm     = llm
        self._chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    async def ingest_pdf(self, file_bytes: bytes, original_filename: str) -> IngestResponse:
        # ── 1. Persist temp file ─────────────────────────────────────────────
        tmp_path = Path(f"/tmp/{uuid.uuid4()}_{original_filename}")
        tmp_path.write_bytes(file_bytes)

        try:
            # ── 2. Parse PDF ─────────────────────────────────────────────────
            parser      = PDFParser(tmp_path)
            parse_result = parser.parse()

            doc_id = str(uuid.uuid4())

            # ── 3. Chunk ──────────────────────────────────────────────────────
            chunks = self._chunker.chunk_document(
                pages=parse_result.pages,
                doc_id=doc_id,
                file_name=original_filename,
                extra_metadata=parse_result.metadata,
            )
            stats = self._chunker.stats(chunks)

            if not chunks:
                raise ValueError("No text could be extracted from the PDF.")

            # ── 4. Embed all chunks ───────────────────────────────────────────
            embeddings = []
            for chunk in chunks:
                vec = await self._llm.embed(chunk.text)
                embeddings.append(vec)

            # ── 5. Upsert to VectorDB ─────────────────────────────────────────
            upserted = await self._vdb.upsert_chunks(chunks, embeddings)

            return IngestResponse(
                doc_id=doc_id,
                file_name=original_filename,
                chunk_count=upserted,
                is_arabic=parse_result.metadata.get("is_arabic", False),
                chunk_stats=stats,
            )

        finally:
            tmp_path.unlink(missing_ok=True)
