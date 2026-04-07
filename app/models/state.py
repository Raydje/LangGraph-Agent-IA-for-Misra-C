from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class RetrievedRule(TypedDict):
    rule_id: str
    standard: str
    section: str
    category: str
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
    standard: str
    code_snippet: NotRequired[str]

    # Orchestrator
    intent: NotRequired[Literal["search", "validate", "explain"]]
    orchestrator_reasoning: NotRequired[str]

    # RAG
    retrieved_rules: NotRequired[list[RetrievedRule]]
    rag_query_used: NotRequired[str]
    metadata_filters_applied: NotRequired[dict]

    # Validation
    validation_result: NotRequired[str]
    is_compliant: NotRequired[bool]
    confidence_score: NotRequired[float]
    cited_rules: NotRequired[list[str]]

    # Critique loop
    critique_feedback: NotRequired[str]
    critique_approved: NotRequired[bool]
    iteration_count: NotRequired[int]
    max_iterations: NotRequired[int]
    critique_history: NotRequired[Annotated[list[CritiqueEntry], operator.add]]

    # Remediation
    fixed_code_snippet: NotRequired[str]
    remediation_explanation: NotRequired[str]

    # Metadata
    prompt_tokens: NotRequired[Annotated[int, operator.add]]
    completion_tokens: NotRequired[Annotated[int, operator.add]]
    total_tokens: NotRequired[Annotated[int, operator.add]]
    orchestrator_tokens: NotRequired[Annotated[int, operator.add]]
    validation_tokens: NotRequired[Annotated[int, operator.add]]
    critique_tokens: NotRequired[Annotated[int, operator.add]]
    remediation_tokens: NotRequired[Annotated[int, operator.add]]
    estimated_cost: NotRequired[Annotated[float, operator.add]]

    # Output
    final_response: NotRequired[str]
    error: NotRequired[str]
