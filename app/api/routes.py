import uuid
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from app.models.requests import ComplianceQueryRequest
from app.models.responses import (
    ComplianceQueryResponse,
    HealthResponse,
    IngestResponse,
    MetadataUsage,
    ThreadHistoryEntry,
    ThreadHistoryResponse,
)
from app.api.dependencies import get_compiled_graph, get_mongo_db, get_pinecone_index, limiter
from app.config import get_settings
from app.data.ingest import main as ingest
from app.utils import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(
    request: Request,
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
@limiter.limit("5/minute")
async def query_compliance(
    request: Request,
    body: ComplianceQueryRequest,
    graph=Depends(get_compiled_graph),
):
    """Main endpoint to trigger the LangGraph multi-agent compliance check."""
    settings = get_settings()
    # Initialize the LangGraph State
    initial_state = {
        "query": body.query,
        "code_snippet": body.code_snippet or "",
        "standard": body.standard,
        "iteration_count": 0,
        "max_iterations": settings.max_critique_iterations,
        "critique_history": [],
    }

    try:
        # Use caller-supplied thread_id for conversation continuity, or mint a new one
        thread_id = body.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unable to process query. Please try again later or use health check endpoints.")

    return _build_response(thread_id, result)

@router.post("/seed", response_model=IngestResponse)
@limiter.limit("2/minute")
async def seed_database(request: Request):
    """Endpoint to trigger the ingestion of rules into MongoDB and Pinecone."""
    result = await ingest()
    return IngestResponse(
        message="Seed data ingested successfully",
        rules_ingested=result.get("rules_ingested", 0),
        vectors_upserted=result.get("vectors_upserted", 0),
    )

@router.post("/replay/{thread_id}/{checkpoint_id}", response_model=ComplianceQueryResponse)
@limiter.limit("10/minute")
async def replay_from_checkpoint(
    request: Request,
    thread_id: str = Path(..., description="Thread ID of the session to replay"),
    checkpoint_id: str = Path(..., description="Checkpoint ID to fork execution from"),
    graph=Depends(get_compiled_graph),
):
    """
    Forks graph execution from a specific SQLite-backed checkpoint.
    Loads the state saved at checkpoint_id and re-runs from that node forward.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    # Validate the checkpoint exists before attempting replay
    checkpoint_state = await graph.aget_state(config)
    if not checkpoint_state or not checkpoint_state.values:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'",
        )

    try:
        # None input signals LangGraph to resume from the checkpoint's saved state
        result = await graph.ainvoke(None, config=config)
    except Exception as e:
        logger.exception("Replay failed for thread=%s checkpoint=%s", thread_id, checkpoint_id)
        raise HTTPException(status_code=500, detail=f"Replay failed: maybe wrong checkpoint_id or gemini is down")

    return _build_response(thread_id, result)

@router.get("/history/{thread_id}", response_model=ThreadHistoryResponse)
@limiter.limit("20/minute")
async def get_thread_history(
    request: Request,
    thread_id: str,
    graph=Depends(get_compiled_graph)
):
    """Retrieves all checkpoint snapshots for a thread, newest-to-oldest (for debugging)."""
    config = {"configurable": {"thread_id": thread_id}}
    history: list[ThreadHistoryEntry] = []

    async for state in graph.aget_state_history(config):
        history.append(ThreadHistoryEntry(
            checkpoint_id=state.config["configurable"].get("checkpoint_id"),
            next_node=state.next,
            values={k: v for k, v in state.values.items() if k != "code_snippet"},
        ))

    if not history:
        raise HTTPException(status_code=404, detail=f"No history found for thread_id '{thread_id}'")

    return ThreadHistoryResponse(thread_id=thread_id, history=history)


def _build_response(thread_id: str, result: dict) -> ComplianceQueryResponse:
    """Maps a raw LangGraph state dict to the API response model."""
    return ComplianceQueryResponse(
        thread_id=thread_id,
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
        fixed_code_snippet=result.get("fixed_code_snippet"),
        remediation_explanation=result.get("remediation_explanation"),
        total_tokens_usage=MetadataUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            orchestrator_tokens=result.get("orchestrator_tokens", 0),
            validation_tokens=result.get("validation_tokens", 0),
            critique_tokens=result.get("critique_tokens", 0),
            remediation_tokens=result.get("remediation_tokens", 0),
            estimated_cost=result.get("estimated_cost", 0.0),
        ),
    )
