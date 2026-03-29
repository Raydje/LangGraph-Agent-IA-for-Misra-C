from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.config import get_settings
from fastapi.responses import RedirectResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Logs configuration on startup to ensure environment variables are loaded."""
    settings = get_settings()
    print(f"[Startup] Loaded config for standard: MISRA C:2023")
    print(f"[Startup] Gemini model: {settings.gemini_model}")
    print(f"[Startup] MongoDB: {settings.mongodb_uri}/{settings.mongodb_database}")
    print(f"[Startup] Pinecone index: {settings.pinecone_index_name}")
    yield


# Initialize FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="MISRA C:2023 Compliance Agent",
    description="Autonomous regulatory compliance analysis using a LangGraph multi-agent system.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# Include the routes defined in routes.py
app.include_router(router, prefix="/api/v1")