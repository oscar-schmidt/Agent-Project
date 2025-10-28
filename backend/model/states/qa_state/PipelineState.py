from pydantic import BaseModel
from typing import List
from backend.model.states.qa_state.DocTextClass import DocTextClass


class PipelineState(BaseModel):
    doc_path: str = ""
    doc_name: str = ""
    chunked_pdf_text: List[DocTextClass] = []
    logs: list[str] = []
    final_summary: str = ""
