from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings

T = TypeVar('T', bound=BaseModel)

def get_llm(temperature: float = 0.7, timeout: int = 120) -> ChatGoogleGenerativeAI:
    """
    Returns a standard Gemini model for general text generation.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
        request_timeout=timeout,
    )

def get_structured_llm(
    response_schema: Type[T],
    temperature: float = 0.0,
    timeout: int = 120,
    raw_bool: bool = True,
) -> Runnable:
    """
    Returns a Gemini model bound to a specific Pydantic schema.
    Essential for Orchestrator, Validation, and Critique nodes to guarantee JSON output.
    Uses temperature 0.0 by default for deterministic structural adherence.
    """
    llm = get_llm(temperature=temperature, timeout=timeout)
    # with_structured_output forces the LLM to return data matching the Pydantic schema
    return llm.with_structured_output(response_schema, include_raw=raw_bool)