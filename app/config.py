from pydantic import model_validator
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
    # Populated at runtime by set_model_pricing based on the active gemini_model
    gemini_2_5_flash_input_cost_per_1m: float = 0.0
    gemini_2_5_flash_output_cost_per_1m: float = 0.0

    @model_validator(mode="after")
    def set_model_pricing(self) -> "Settings":
        fallback = models_pricing.get("gemini-2.5-flash", [0.0, 0.0])
        pricing = models_pricing.get(self.gemini_model, fallback)
        self.gemini_2_5_flash_input_cost_per_1m = pricing[0]
        self.gemini_2_5_flash_output_cost_per_1m = pricing[1]
        return self
    
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "compliance-rules"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_timeout: int = 15  # seconds

    # MongoDB for MISRA rules storage + checkpoints storage for LangGraph
    mongodb_uri: str
    mongodb_database: str = "compliance_db"
    mongodb_collection: str = "rules"
    mongodb_checkpoints_collection: str = "checkpoints"
    mongodb_timeout: int = 15  # seconds

    # LLM timeout
    llm_timeout: int = 30  # seconds

    # Input validation
    max_input_length: int = 3000

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

    # Redis (rate limiting)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_user: str = "default"
    redis_password: str = ""  # empty = no auth (safe local default)

    @property
    def redis_uri(self) -> str:
        if self.redis_password or self.redis_user != "default":
            return f"redis://{self.redis_user}:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
