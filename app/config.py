from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

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
    max_critique_iterations: int = 3
    confidence_threshold: float = 0.85

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
