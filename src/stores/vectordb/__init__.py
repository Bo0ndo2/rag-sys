"""
VectorDB Factory
=================
Same factory pattern as LLMFactory — allows swapping between Qdrant, FAISS, etc.
"""

from abc import ABC, abstractmethod
from src.helpers.chunker import Chunk


class BaseVectorDB(ABC):

    @abstractmethod
    async def create_collection_if_not_exists(self, collection: str) -> None: ...

    @abstractmethod
    async def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks. Returns count of upserted records."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        collection: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return top_k most similar chunks as dicts with score."""
        ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str, collection: str) -> int: ...


class VectorDBFactory:
    _registry: dict[str, type[BaseVectorDB]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[BaseVectorDB]):
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def get_client(cls, provider: str) -> BaseVectorDB:
        if provider not in cls._registry:
            raise ValueError(f"Unknown VectorDB provider '{provider}'")
        return cls._registry[provider]()


from src.stores.vectordb.provider import qdrant_client  # noqa: E402, F401
