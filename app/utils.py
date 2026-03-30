import json
import re
import structlog
import logging
from app.config import get_settings


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = text.strip()
    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    return json.loads(cleaned)

def calculate_gemini_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the estimated cost for Gemini 2.5 Flash usage."""
    settings = get_settings()
    input_cost = (prompt_tokens / 1_000_000) * settings.gemini_2_5_flash_input_cost_per_1m
    output_cost = (completion_tokens / 1_000_000) * settings.gemini_2_5_flash_output_cost_per_1m
    return input_cost + output_cost

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            # Use JSON in production, ConsoleRenderer for local dev
            structlog.dev.ConsoleRenderer() 
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()
