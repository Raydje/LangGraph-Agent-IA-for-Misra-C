from contextlib import asynccontextmanager

import redis as redis_sync
import redis.asyncio as redis_async
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from langgraph.checkpoint.mongodb import MongoDBSaver
from limits.storage import RedisStorage
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.dependencies import limiter
from app.api.v1.routes import router
from app.auth.router import auth_router
from app.config import get_settings
from app.graph.builder import build_graph
from app.services.service_container import create_service_container
from app.services.usage_service import UsageService
from app.utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all services and the LangGraph agent on startup; tear down on shutdown."""
    settings = get_settings()
    logger.info("[Startup] Loaded config for standard: MISRA C:2023")
    logger.info("[Startup] Gemini model", model_name=settings.gemini_model)
    logger.info("[Startup] MongoDB", database=settings.mongodb_database)
    logger.info(
        "[Startup] Pinecone index",
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    # Fail-fast on Redis connection to ensure rate-limiting configuration
    # errors are caught during startup rather than at runtime.
    redis_client = None
    try:
        _r = redis_sync.from_url(settings.redis_uri, socket_connect_timeout=3)
        _r.ping()
        _r.close()
        logger.info("[Startup] Redis connected", host=settings.redis_host, port=settings.redis_port)
        redis_client = redis_async.from_url(
            settings.redis_uri,
            encoding="utf-8",
            decode_responses=True,
        )
    except Exception as e:
        logger.warning("[Startup] Redis unavailable — per-user rate limiting degraded to pass-through", error=str(e))

    app.state.redis = redis_client

    async with create_service_container() as container:
        # The service container manages connection lifecycles. We attach singletons to
        # app.state so FastAPI route dependencies can access them without expensive re-instantiation per request.
        app.state.mongodb = container.mongodb
        app.state.pinecone = container.pinecone
        app.state.mongodb_checkpoint = container.mongodb_checkpoint
        app.state.embedding = container.embedding

        checkpointer = MongoDBSaver(
            container.mongodb_checkpoint.client,
            db_name=container.mongodb_checkpoint.db.name,
            collection=container.mongodb_checkpoint.collection.name,
        )
        app.state.graph = await build_graph(checkpointer=checkpointer)

        # Create auth indexes. MongoDB index creation is idempotent, making this safe to call on every boot.
        auth_db = container.mongodb.db
        await auth_db["users"].create_index("email", unique=True)
        await auth_db["api_keys"].create_index("key_id")
        await auth_db["api_keys"].create_index("user_id")
        logger.info("[Startup] Auth indexes ensured (users.email, api_keys.key_id, api_keys.user_id)")

        usage_service = UsageService(db=auth_db)
        await usage_service.create_indexes()
        app.state.usage_service = usage_service

        yield

    # --- Shutdown (container finally block closes all service connections) ---
    if redis_client is not None:
        try:
            await redis_client.aclose()
            logger.info("[Shutdown] Async Redis client closed")
        except Exception as e:
            logger.warning("[Shutdown] Async Redis close error", error=str(e))

    try:
        if isinstance(limiter._storage, RedisStorage):
            limiter._storage.storage.close()
            logger.info("[Shutdown] SlowAPI Redis connection closed")
    except Exception as e:
        logger.warning("[Shutdown] SlowAPI Redis close error", error=str(e))


app = FastAPI(
    title="MISRA C:2023 Compliance Agent",
    description="Autonomous regulatory compliance analysis using a LangGraph multi-agent system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "error": exc.__class__.__name__, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on", exc_info=True, request=request.method, request_url=request.url.path, exc=exc)
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "error": "InternalServerError",
            "detail": "An unexpected internal server error occurred.",
        },
    )


app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
