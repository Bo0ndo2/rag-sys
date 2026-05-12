"""POST /api/v1/query — Ask a question against the ingested documents."""

from fastapi import APIRouter, Request, HTTPException
from src.controllers.query_controller import QueryController
from src.routes.schema.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query(request: Request, body: QueryRequest):
   
    controller = QueryController(vdb=request.app.state.vdb, llm=request.app.state.llm)
    try:
        result = await controller.handle_query(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    return result
