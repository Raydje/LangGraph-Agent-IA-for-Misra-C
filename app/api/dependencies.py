from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.config import get_settings
from app.services.mongodb_service import MongoDBService, MongoDBCheckpointService
from app.services.pinecone_service import PineconeService
from app.services.embedding_service import EmbeddingService


def get_mongodb_service(request: Request) -> MongoDBService:
    return request.app.state.mongodb

def get_mongodb_checkpoint_service(request: Request) -> MongoDBCheckpointService:
    return request.app.state.mongodb_checkpoint

def get_pinecone_service(request: Request) -> PineconeService:
    return request.app.state.pinecone

def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding

def get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


def _redis_reachable(uri: str) -> bool:
    try:
        import redis as _redis
        client = _redis.from_url(uri, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


try:
    _redis_uri = get_settings().redis_uri
    _storage_uri = _redis_uri if _redis_reachable(_redis_uri) else None
    limiter = Limiter(
        key_func=get_real_ip,
        storage_uri=_storage_uri,
        strategy="moving-window",
    )
except Exception:
    limiter = Limiter(key_func=get_real_ip, strategy="moving-window")


def get_compiled_graph(request: Request):
    return request.app.state.graph

def get_mongodb_database(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.mongodb.db

def get_pinecone_index(request: Request):
    return request.app.state.pinecone.index
