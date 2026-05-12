from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.routes import ingest, query, health
from src.stores.vectordb import VectorDBFactory
from src.stores.llm import LLMFactory
from src.helpers.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise VectorDB collection
    vdb = VectorDBFactory.get_client(settings.VECTOR_DB_PROVIDER)
    await vdb.create_collection_if_not_exists(settings.QDRANT_COLLECTION)

    # Attach singletons to app state so routes can reach them
    app.state.vdb = vdb
    app.state.llm = LLMFactory.get_client(settings.LLM_PROVIDER)

    print(f"[startup] VectorDB={settings.VECTOR_DB_PROVIDER}  LLM={settings.LLM_PROVIDER}")
    yield
    print("[shutdown] Closing connections...")


app = FastAPI(
    title="CV-Matching RAG API",
    description="End-to-end Retrieval-Augmented Generation system for CV / job matching",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(query.router,  prefix="/api/v1", tags=["Query"])
