from fastapi import APIRouter, HTTPException, Depends
from app.models.requests import ComplianceQueryRequest
from app.models.responses import (
    ComplianceQueryResponse,
    HealthResponse,
    IngestResponse,
    MetadataUsage,  
)
from app.api.dependencies import get_compiled_graph, get_mongo_db, get_pinecone_index
from app.config import get_settings
from app.data.ingest import main as ingest

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    db=Depends(get_mongo_db),
    index=Depends(get_pinecone_index),
):
    """Checks the health of the API and backing databases (MongoDB & Pinecone)."""
    mongo_ok = False
    pinecone_ok = False

    if db is not None:
        try:
            await db.command("ping")
            mongo_ok = True
        except Exception:
            pass

    if index is not None:
        try:
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
async def query_compliance(
    request: ComplianceQueryRequest,
    graph=Depends(get_compiled_graph),
):
    """Main endpoint to trigger the LangGraph multi-agent compliance check."""

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
        # Metadata
        total_tokens_usage=MetadataUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            orchestrator_tokens=result.get("orchestrator_tokens", 0),
            validation_tokens=result.get("validation_tokens", 0),
            critique_tokens=result.get("critique_tokens", 0),
            estimated_cost=result.get("estimated_cost", 0.0),
        ),
    )

@router.post("/seed", response_model=IngestResponse)
async def seed_database():
    """Endpoint to trigger the ingestion of rules into MongoDB and Pinecone."""
    result = await ingest()
    return IngestResponse(
        message="Seed data ingested successfully",
        rules_ingested=result.get("rules_ingested", 0),
        vectors_upserted=result.get("vectors_upserted", 0),
    )