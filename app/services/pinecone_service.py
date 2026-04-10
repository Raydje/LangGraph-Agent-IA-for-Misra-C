import asyncio

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index

from app.config import get_settings
from app.utils import logger


class PineconeService:
    def __init__(self) -> None:
        settings = get_settings()
        self.index: Index | None = None
        try:
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

            self.index = pc.Index(settings.pinecone_index_name)
            logger.info("[Startup] Pinecone connected", index=settings.pinecone_index_name)
        except Exception:
            logger.warning("[Startup] Pinecone unavailable — starting in degraded mode")
            self.index = None

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> dict:
        if self.index is None:
            logger.warning("Pinecone index unavailable, skipping query.")
            return {"matches": []}
        settings = get_settings()
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.index.query,
                    vector=vector,
                    top_k=top_k,
                    filter=filter,
                    include_metadata=True,
                ),
                timeout=settings.pinecone_timeout,
            )
            logger.info("Pinecone query returned matches.", number_matches=len(results.matches))  # type: ignore[union-attr]
            return {
                "matches": [
                    {
                        "id": m.id,
                        "score": m.score,
                        "metadata": m.metadata or {},
                    }
                    for m in results.matches  # type: ignore[union-attr]
                ]
            }
        except TimeoutError:
            logger.error(
                "Pinecone query timed out seconds. try healthy check on Pinecone index.",
                timeout=settings.pinecone_timeout,
            )
            return {"matches": []}

    async def upsert_vectors(self, vectors: list[dict]) -> int:
        if self.index is None:
            logger.warning("Pinecone index unavailable, skipping upsert.")
            return 0
        batch_size = 100
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            await asyncio.to_thread(self.index.upsert, vectors=batch)  # type: ignore[arg-type]  # noqa: E501
            logger.info("Successfully upserted vectors to Pinecone.", number_upserted=len(batch))
            total += len(batch)
        return total
