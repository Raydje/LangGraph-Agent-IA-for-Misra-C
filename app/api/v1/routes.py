import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, Security

from app.api.dependencies import (
    get_compiled_graph,
    get_embedding_service,
    get_mongodb_database,
    get_mongodb_service,
    get_pinecone_index,
    get_pinecone_service,
    get_usage_service,
    limiter,
)
from app.api.rate_limit import enforce_user_budget, enforce_user_rate_limit
from app.api.v1.requests import ComplianceQueryRequest
from app.api.v1.responses import (
    ComplianceQueryResponse,
    HealthResponse,
    IngestResponse,
    MetadataUsage,
    ThreadHistoryEntry,
    ThreadHistoryResponse,
    UsageResponse,
)
from app.auth.dependencies import get_current_principal
from app.auth.models import Principal
from app.config import get_settings
from app.data.ingest import main as ingest
from app.services.usage_service import UsageService
from app.utils import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(
    request: Request,
    db=Depends(get_mongodb_database),
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
    response: Response,
    body: ComplianceQueryRequest,
    graph=Depends(get_compiled_graph),
    embedding_service=Depends(get_embedding_service),
    mongo_db=Depends(get_mongodb_service),
    pinecone_service=Depends(get_pinecone_service),
    usage_service: UsageService = Depends(get_usage_service),
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
    _rate: None = Depends(enforce_user_rate_limit),
    _budget: None = Depends(enforce_user_budget),
):
    """Main endpoint to trigger the LangGraph multi-agent compliance check."""
    settings = get_settings()
    initial_state = {
        "query": body.query,
        "code_snippet": body.code_snippet or "",
        "standard": body.standard,
        "iteration_count": 0,
        "max_iterations": settings.max_critique_iterations,
        "critique_history": [],
    }

    thread_id = body.thread_id or str(uuid.uuid4())
    status_code = 200
    result = None  # Initialize before try block for finally access

    config = {
        "configurable": {
            "thread_id": thread_id,
            "mongo_db": mongo_db,
            "pinecone_service": pinecone_service,
            "embedding_service": embedding_service,
        }
    }
    nodes_visited = None
    try:
        result = await graph.ainvoke(initial_state, config=config)
        # Compute nodes visited only on successful execution
        nodes_visited = []
        async for state in graph.aget_state_history(config):
            if state.next:
                nodes_visited.append(state.next[0])
    except Exception as e:
        status_code = 500
        logger.exception("Compliance query failed for thread_id=%s", thread_id)
        logger.error("Error details: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Unable to process query. Please try again later or use health check endpoints."
        ) from e
    finally:
        # Record usage regardless of success/failure so cost is always tracked.
        # On exception the result dict may not exist — use safe defaults.
        _result = result if result is not None else {}
        await usage_service.record_usage(
            user_id=principal.user_id,
            endpoint="/api/v1/query",
            method="POST",
            thread_id=thread_id,
            prompt_tokens=_result.get("prompt_tokens", 0),
            completion_tokens=_result.get("completion_tokens", 0),
            total_tokens=_result.get("total_tokens", 0),
            estimated_cost=_result.get("estimated_cost", 0.0),
            critique_iterations=_result.get("iteration_count", 0),
            nodes_visited=nodes_visited,
            status_code=status_code,
        )

    return _build_response(thread_id, result)


@router.post("/seed", response_model=IngestResponse)
@limiter.limit("2/minute")
async def seed_database(
    request: Request,
    response: Response,
    principal: Principal = Security(get_current_principal, scopes=["admin:seed"]),
    embedding_service=Depends(get_embedding_service),
    mongo_db=Depends(get_mongodb_service),
    pinecone_service=Depends(get_pinecone_service),
    _rate: None = Depends(enforce_user_rate_limit),
):
    """Endpoint to trigger the ingestion of rules into MongoDB and Pinecone."""
    result = await ingest(mongodb=mongo_db, pinecone=pinecone_service, embedder=embedding_service)
    return IngestResponse(
        message="Seed data ingested successfully",
        rules_ingested=result.get("rules_ingested", 0),
        vectors_upserted=result.get("vectors_upserted", 0),
    )


@router.post("/replay/{thread_id}/{checkpoint_id}", response_model=ComplianceQueryResponse)
@limiter.limit("10/minute")
async def replay_from_checkpoint(
    request: Request,
    response: Response,
    thread_id: str = Path(..., description="Thread ID of the session to replay"),
    checkpoint_id: str = Path(..., description="Checkpoint ID to fork execution from"),
    graph=Depends(get_compiled_graph),
    embedding_service=Depends(get_embedding_service),
    mongo_db=Depends(get_mongodb_service),
    pinecone_service=Depends(get_pinecone_service),
    principal: Principal = Security(get_current_principal, scopes=["admin:replay"]),
    _rate: None = Depends(enforce_user_rate_limit),
):
    """
    Forks graph execution from a specific MongoDB-backed checkpoint.
    Loads the state saved at checkpoint_id and re-runs from that node forward.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "mongo_db": mongo_db,
            "pinecone_service": pinecone_service,
            "embedding_service": embedding_service,
        }
    }

    checkpoint_state = await graph.aget_state(config)
    if not checkpoint_state or not checkpoint_state.values:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'",
        )

    try:
        result = await graph.ainvoke(None, config=config)
    except Exception as e:
        logger.exception("Replay failed for thread=%s checkpoint=%s", thread_id, checkpoint_id)
        logger.error("Error details: %s", str(e))
        raise HTTPException(status_code=500, detail="Replay failed: maybe wrong checkpoint_id or gemini is down") from e

    return _build_response(thread_id, result)


@router.get("/history/{thread_id}", response_model=ThreadHistoryResponse)
@limiter.limit("20/minute")
async def get_thread_history(
    request: Request,
    response: Response,
    thread_id: str,
    graph=Depends(get_compiled_graph),
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
    _rate: None = Depends(enforce_user_rate_limit),
):
    """Retrieves all checkpoint snapshots for a thread, newest-to-oldest (for debugging)."""
    config = {"configurable": {"thread_id": thread_id}}
    history: list[ThreadHistoryEntry] = []

    async for state in graph.aget_state_history(config):
        history.append(
            ThreadHistoryEntry(
                checkpoint_id=state.config["configurable"].get("checkpoint_id"),
                next_node=state.next,
                values={k: v for k, v in state.values.items() if k != "code_snippet"},
            )
        )

    if not history:
        raise HTTPException(status_code=404, detail=f"No history found for thread_id '{thread_id}'")

    return ThreadHistoryResponse(thread_id=thread_id, history=history)


@router.get("/usage", response_model=UsageResponse)
@limiter.limit("30/minute")
async def get_usage(
    request: Request,
    response: Response,
    usage_service: UsageService = Depends(get_usage_service),
    principal: Principal = Security(get_current_principal, scopes=["query:read"]),
    _rate: None = Depends(enforce_user_rate_limit),
):
    """Retrieve usage summary for the authenticated user."""
    usage_data = await usage_service.get_user_usage(principal.user_id)
    if not usage_data:
        raise HTTPException(status_code=404, detail="User not found")
    return UsageResponse(**usage_data)


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
