from __future__ import annotations

from pydantic import BaseModel, Field


class RallyStats(BaseModel):
    hits: int = Field(ge=0)
    duration: float = Field(ge=0.0, description="Seconds")


class AnalyzeResponse(BaseModel):
    total_rallies: int = Field(ge=0)
    rallies: list[RallyStats]

