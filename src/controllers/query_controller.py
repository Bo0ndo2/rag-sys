"""
Query Controller — Phase 3: RAG Logic & Retrieval
===================================================
Pipeline:
  user query → embed → vector search (top-k) → build context → LLM generate → return
"""

from src.helpers.config import get_settings
from src.stores.llm import LLMFactory
from src.stores.llm.template.prompts import build_system_prompt
from src.routes.schema.schemas import QueryRequest, QueryResponse, RetrievedChunk

settings = get_settings()


class QueryController:

    def __init__(self, vdb, llm):
        self._vdb = vdb
        self._llm = llm   # default provider from settings

    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        # ── 1. Override LLM provider per-request if requested (Factory bonus) ──
        llm = self._llm
        if request.provider and request.provider != settings.LLM_PROVIDER:
            llm = LLMFactory.get_client(request.provider)

        # ── 2. Embed the user query ───────────────────────────────────────────
        query_vector = await llm.embed(request.query)

        # ── 3. Vector search ──────────────────────────────────────────────────
        raw_results = await self._vdb.search(
            query_vector=query_vector,
            top_k=request.top_k,
        )

        if not raw_results:
            return QueryResponse(
                answer="No relevant documents found in the database.",
                retrieved_chunks=[],
                llm_provider=llm.provider_name(),
                query=request.query,
            )

        # ── 4. Build context string ───────────────────────────────────────────
        context_parts = []
        for idx, r in enumerate(raw_results, 1):
            context_parts.append(
                f"[Chunk {idx} | File: {r.get('file_name','?')} | "
                f"Page: {r.get('page_number','?')} | Score: {r.get('score',0):.3f}]\n"
                f"{r['text']}"
            )
        context = "\n\n".join(context_parts)

        # ── 5. Build prompt and generate answer ───────────────────────────────
        system_prompt = build_system_prompt(context, lang=request.lang)

        answer = await llm.generate(
            system_prompt=system_prompt,
            user_message=request.query,
        )

        # ── 6. Assemble response ──────────────────────────────────────────────
        retrieved = [
            RetrievedChunk(
                text=r["text"],
                file_name=r.get("file_name", ""),
                page_number=r.get("page_number"),
                chunk_index=r.get("chunk_index", 0),
                score=r.get("score", 0.0),
            )
            for r in raw_results
        ]

        return QueryResponse(
            answer=answer,
            retrieved_chunks=retrieved,
            llm_provider=llm.provider_name(),
            query=request.query,
        )
