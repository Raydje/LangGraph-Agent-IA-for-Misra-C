import json
import re
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
