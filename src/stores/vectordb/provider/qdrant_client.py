"""Qdrant vector database provider."""

import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue,
)

from src.stores.vectordb import VectorDBFactory, BaseVectorDB
from src.helpers.chunker import Chunk
from src.helpers.config import get_settings

settings = get_settings()


@VectorDBFactory.register("qdrant")
class QdrantStore(BaseVectorDB):

    def __init__(self):
        self._client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self._collection = settings.QDRANT_COLLECTION

    async def create_collection_if_not_exists(self, collection: str) -> None:
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if collection not in names:
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[qdrant] Created collection '{collection}'")
        else:
            print(f"[qdrant] Collection '{collection}' already exists")

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=chunk.to_dict(),
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        collection: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        col = collection or self._collection
        results = await self._client.search(
            collection_name=col,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {**r.payload, "score": r.score}
            for r in results
        ]

    async def delete_by_doc_id(self, doc_id: str, collection: str | None = None) -> int:
        col = collection or self._collection
        result = await self._client.delete(
            collection_name=col,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        return result.status.value if hasattr(result, "status") else 0
