from pydantic import BaseModel

from backend.model.states.qa_state.DocTextClass import DocTextClass


class DocState(BaseModel):
    doc_path: str = ""
    doc_name: str = ""
    chunked_doc_text: list[DocTextClass] = []
    top_k_kb: str = ""
    is_upload: bool = False
    is_processed: bool = False
