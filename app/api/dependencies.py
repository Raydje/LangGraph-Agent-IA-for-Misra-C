from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import get_settings


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


def get_mongo_db(request: Request):
    return request.app.state.mongodb.db


def get_pinecone_index(request: Request):
    return request.app.state.pinecone.index
