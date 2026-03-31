from pydantic_settings import BaseSettings
from functools import lru_cache
from app.models_pricing import models_pricing


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    
    # Pricing dollar per 1M tokens (Standard Google AI Studio Pay-as-you-go)
    gemini_2_5_flash_input_cost_per_1m: float = models_pricing[gemini_model][0]
    gemini_2_5_flash_output_cost_per_1m: float = models_pricing[gemini_model][1]
    
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "compliance-rules"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # MongoDB
    mongodb_uri: str
    mongodb_database: str = "compliance_db"
    mongodb_collection: str = "rules"

    # Graph control
    max_critique_iterations: int = 4
    confidence_threshold: float = 0.85

    # Node temperatures
    orchestrator_temperature: float = 0.0
    validation_temperature: float = 0.1
    critique_temperature: float = 0.0
    remediation_temperature: float = 0.2

    # CORS
    cors_allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8501", "http://localhost:8080"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
