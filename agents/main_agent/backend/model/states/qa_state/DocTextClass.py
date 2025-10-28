from typing import Optional
from pydantic import BaseModel, Field


class Meta(BaseModel):
    doc_name: str = ""
    referenece_number: int = 0
    review_id: str = ""
    reviewer: str = ""
    date: str = ""
    error_summary: str = ""
    error_type: str = ""
    criticality: str = ""
    rationale: str = ""


class DocTextClass(BaseModel):
    meta: Meta = Meta()
    chunk: str = ""
