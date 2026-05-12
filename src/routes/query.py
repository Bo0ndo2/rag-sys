"""POST /api/v1/query — Ask a question against the ingested documents."""

from fastapi import APIRouter, Request, HTTPException
from src.controllers.query_controller import QueryController
from src.routes.schema.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query(request: Request, body: QueryRequest):
    """
    Send a natural-language question (in English or Arabic).
    The system will:
      1. Embed the query using the same model as ingestion
      2. Retrieve the top-k most relevant chunks from Qdrant
      3. Inject chunks as context into the LLM prompt
      4. Return the generated answer + source chunks

    **lang** parameter: `en` (default) or `ar` (Arabic prompt template).
    **provider** parameter: optionally override the LLM (openai | gemini | ollama).
    """
    controller = QueryController(vdb=request.app.state.vdb, llm=request.app.state.llm)
    try:
        result = await controller.handle_query(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    return result
