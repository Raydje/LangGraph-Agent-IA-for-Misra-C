from __future__ import annotations
from typing import TypedDict, Annotated, Literal
import operator


class RetrievedRule(TypedDict):
    rule_id: str
    standard: str
    section: str
    dal_level: str
    title: str
    full_text: str
    relevance_score: float


class CritiqueEntry(TypedDict):
    iteration: int
    issues_found: list[str]
    approved: bool


class ComplianceState(TypedDict):
    # Input
    query: str
    code_snippet: str
    standard: str

    # Orchestrator
    intent: Literal["search", "validate", "explain"]
    orchestrator_reasoning: str

    # RAG
    retrieved_rules: list[RetrievedRule]
    rag_query_used: str
    metadata_filters_applied: dict

    # Validation
    validation_result: str
    is_compliant: bool
    confidence_score: float
    cited_rules: list[str]

    # Critique loop
    critique_feedback: str
    critique_approved: bool
    iteration_count: int
    max_iterations: int
    critique_history: Annotated[list[CritiqueEntry], operator.add]

    # Output
    final_response: str
    error: str
