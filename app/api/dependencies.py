from functools import lru_cache
from app.graph.builder import build_graph
from app.services.mongodb_service import _get_db
from app.services.pinecone_service import _get_index


@lru_cache()
def get_compiled_graph():
    return build_graph()


def get_mongo_db():
    try:
        return _get_db()
    except Exception:
        return None


def get_pinecone_index():
    try:
        return _get_index()
    except Exception:
        return None
