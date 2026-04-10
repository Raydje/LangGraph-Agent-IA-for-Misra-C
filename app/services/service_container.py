from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.mongodb_service import MongoDBCheckpointService, MongoDBService
from app.services.pinecone_service import PineconeService


@dataclass
class ServiceContainer:
    """Holds the single shared instance of every infrastructure service."""

    mongodb: MongoDBService
    mongodb_checkpoint: MongoDBCheckpointService
    pinecone: PineconeService
    embedding: EmbeddingService


@asynccontextmanager
async def create_service_container() -> AsyncIterator[ServiceContainer]:
    """
    Async context manager that creates all infrastructure services and guarantees
    their cleanup on exit — whether the caller exits normally or via an exception.

    Usage (FastAPI lifespan, CLI scripts, integration tests):

        async with create_service_container() as container:
            # container.mongodb, container.pinecone, etc. are ready
            ...
        # connections are closed here automatically
    """
    container = ServiceContainer(
        mongodb=MongoDBService(),
        mongodb_checkpoint=MongoDBCheckpointService(),
        pinecone=PineconeService(),
        embedding=EmbeddingService(),
    )
    try:
        yield container
    finally:
        container.mongodb.close()
        if container.pinecone.index is not None:
            container.pinecone.index.close()
        container.mongodb_checkpoint.close()
