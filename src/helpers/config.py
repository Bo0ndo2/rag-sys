from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    LLM_PROVIDER: str = "openai"          # openai | gemini | ollama
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "mistral"

    # VectorDB
    VECTOR_DB_PROVIDER: str = "qdrant"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "cv_chunks"

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding
    EMBEDDING_DIM: int = 1536

    # Chunking strategy
    CHUNK_SIZE: int = 500           # tokens per chunk
    CHUNK_OVERLAP: int = 50         # token overlap between chunks
    TOP_K: int = 5                  # chunks to retrieve per query

    # MongoDB (optional — stores raw doc metadata)
    MONGO_URI: str = "mongodb://mongodb:27017"
    MONGO_DB: str = "rag_db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
