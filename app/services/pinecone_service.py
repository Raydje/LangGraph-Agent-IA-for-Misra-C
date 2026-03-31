import asyncio
from pinecone import Pinecone, ServerlessSpec
from app.config import get_settings
from app.utils import logger

_index = None


def _get_index():
    global _index
    if _index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )

        _index = pc.Index(settings.pinecone_index_name)
    return _index


async def query_pinecone(
    vector: list[float],
    top_k: int = 5,
    filter: dict | None = None,
) -> dict:
    index = _get_index()
    results = await asyncio.to_thread(
        index.query,
        vector=vector,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
    )
    logger.info(f"Pinecone query returned {len(results.matches)} matches.")
    return {
        "matches": [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata or {},
            }
            for m in results.matches
        ]
    }


async def upsert_vectors(vectors: list[dict]) -> int:
    index = _get_index()
    batch_size = 100
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        await asyncio.to_thread(index.upsert, vectors=batch)
        logger.info(f"Successfully upserted {len(batch)} vectors to Pinecone!")
        total += len(batch)
    return total
