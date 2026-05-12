"""POST /api/v1/ingest — Upload a PDF and ingest it into the vector store."""

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from src.controllers.ingest_controller import IngestController
from src.routes.schema.schemas import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a PDF document")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    controller = IngestController(vdb=request.app.state.vdb, llm=request.app.state.llm)
    try:
        result = await controller.ingest_pdf(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return result
