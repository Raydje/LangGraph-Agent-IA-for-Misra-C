from typing import Any

import structlog

from app.config import get_settings


def calculate_gemini_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the estimated cost for Gemini 2.5 Flash usage."""
    settings = get_settings()
    input_cost = (prompt_tokens / 1_000_000) * settings.gemini_2_5_flash_input_cost_per_1m
    output_cost = (completion_tokens / 1_000_000) * settings.gemini_2_5_flash_output_cost_per_1m
    return input_cost + output_cost


def extracting_tokens_metadata(raw_result: dict[str, Any]) -> dict:
    usage = getattr(raw_result.get("raw"), "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    estimated_cost = calculate_gemini_cost(input_tokens, output_tokens)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
    }


logger = structlog.get_logger()
