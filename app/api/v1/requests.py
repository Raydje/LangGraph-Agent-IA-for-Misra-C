from pydantic import BaseModel, Field, model_validator

from app.config import get_settings


class ComplianceQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language compliance question",
    )
    code_snippet: str | None = Field(
        None,
        description="Code or requirement text to validate",
    )
    standard: str = Field(
        "MISRA C:2023",
        min_length=1,
        description="Technical standard to query against",
    )
    thread_id: str | None = Field(
        None,
        max_length=100,
        description="Optional thread ID to continue an existing conversation; a new UUID is generated if omitted",
    )

    @model_validator(mode="after")
    def check_max_lengths(self):
        limit = get_settings().max_input_length
        if len(self.query) > limit:
            raise ValueError(f"query must be at most {limit} characters")
        if self.code_snippet is not None and len(self.code_snippet) > limit:
            raise ValueError(f"code_snippet must be at most {limit} characters")
        if len(self.standard) > limit:
            raise ValueError(f"standard must be at most {limit} characters")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Does this function violate any MISRA C:2023 rules?",
                "code_snippet": "def calculate_altitude(pressure: float) -> float:\n    return 44330 * (1 - (pressure / 1013.25) ** 0.1903)",
                "standard": "MISRA C:2023",
            }
        }
    }
