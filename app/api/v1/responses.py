from datetime import datetime
from typing import Any

from pydantic import BaseModel


class CritiqueDetail(BaseModel):
    iteration: int
    issues_found: list[str]
    approved: bool


class UsageLogEntry(BaseModel):
    user_id: str
    endpoint: str
    method: str
    timestamp: datetime
    thread_id: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    critique_iterations: int = 0
    nodes_visited: list[str] | None = None
    status_code: int


class UsageResponse(BaseModel):
    user_id: str
    email: str | None
    total_cost: float
    total_requests: int
    recent_logs: list[UsageLogEntry]


class MetadataUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    orchestrator_tokens: int | None = None
    validation_tokens: int | None = None
    critique_tokens: int | None = None
    remediation_tokens: int | None = None
    estimated_cost: float | None = None


class ComplianceQueryResponse(BaseModel):
    thread_id: str
    intent: str
    final_response: str
    is_compliant: bool | None = None
    confidence_score: float | None = None
    cited_rules: list[str] = []
    critique_iterations: int = 0
    critique_passed: bool = True
    critique_history: list[CritiqueDetail] = []
    retrieved_rule_ids: list[str] = []
    error: str | None = None
    fixed_code_snippet: str | None = None
    remediation_explanation: str | None = None
    total_tokens_usage: MetadataUsage


class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    pinecone_connected: bool


class IngestResponse(BaseModel):
    message: str
    rules_ingested: int
    vectors_upserted: int


class ThreadHistoryEntry(BaseModel):
    checkpoint_id: str | None
    next_node: tuple
    values: dict[str, Any]


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    history: list[ThreadHistoryEntry]
