from fastapi import APIRouter, HTTPException
from app.models.requests import ComplianceQueryRequest
from app.models.responses import (
    ComplianceQueryResponse,
    HealthResponse,
    IngestResponse,
)
from app.api.dependencies import get_compiled_graph
from app.services.mongodb_service import get_rules_by_metadata
from app.config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Checks the health of the API and backing databases (MongoDB & Pinecone)."""
    from app.services.mongodb_service import _get_db
    from app.services.pinecone_service import _get_index

    mongo_ok = False
    pinecone_ok = False

    try:
        db = _get_db()
        await db.command("ping")
        mongo_ok = True
    except Exception:
        pass

    try:
        index = _get_index()
        index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass

    status = "healthy" if (mongo_ok and pinecone_ok) else "degraded"
    return HealthResponse(
        status=status,
        mongodb_connected=mongo_ok,
        pinecone_connected=pinecone_ok,
    )

@router.post("/query", response_model=ComplianceQueryResponse)
async def query_compliance(request: ComplianceQueryRequest):
    """Main endpoint to trigger the LangGraph multi-agent compliance check."""
    graph = get_compiled_graph()

    # Initialize the LangGraph State
    initial_state = {
        "query": request.query,
        "code_snippet": request.code_snippet or "",
        "standard": request.standard,
        "iteration_count": 0,
        "max_iterations": settings.max_critique_iterations,
        "critique_history": [],
    }

    try:
        # Run the graph asynchronously
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map the resulting state to our Pydantic response model
    return ComplianceQueryResponse(
        intent=result.get("intent", "unknown"),
        final_response=result.get("final_response", ""),
        is_compliant=result.get("is_compliant"),
        confidence_score=result.get("confidence_score"),
        cited_rules=result.get("cited_rules", []),
        critique_iterations=result.get("iteration_count", 0),
        critique_passed=result.get("critique_approved", True),
        critique_history=result.get("critique_history", []),
        retrieved_rule_ids=[r.get("rule_id", "") for r in result.get("retrieved_rules", [])],
        error=result.get("error"),
    )

@router.post("/seed", response_model=IngestResponse)
async def seed_database():
    """Endpoint to trigger the ingestion of rules into MongoDB and Pinecone."""
    from app.data.ingest import main as ingest
    result = await ingest()
    return IngestResponse(
        message="Seed data ingested successfully",
        rules_ingested=result.get("rules_ingested", 0),
        vectors_upserted=result.get("vectors_upserted", 0),
    )