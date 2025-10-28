from typing import Optional
from pydantic import BaseModel, Field


class SummaryState(BaseModel):
    chunk_summary: list[str] = []
    final_summary: str = ""
