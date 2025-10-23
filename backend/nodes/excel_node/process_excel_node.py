
import os
from dotenv import load_dotenv
import pandas as pd
from backend.embedding.chroma_setup import insert_data_row
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.qa_state.DocTextClass import DocTextClass, Meta
from backend.utils import clean_text, get_embedding, log_decorator


load_dotenv()

doc_path = os.getenv("DOC_PATH")
doc_name = os.path.splitext(os.path.basename(doc_path))[0]


@log_decorator
def process_excel_node(state: GraphState) -> GraphState:
    cols = ['ReviewID', 'Reviewer', 'Date', 'ReviewText',
            'ErrorSummary', 'ErrorType', 'Criticality', 'Rationale']

    df = pd.read_excel(state.qa_state.doc_path, usecols=cols)
    data_row = []

    for index, row in df.iterrows():
        review_text = clean_text(row['ReviewText'])

        meta = Meta(
            doc_name=doc_name,
            referenece_number=index,
            review_id=clean_text(row['ReviewID']),
            reviewer=clean_text(row['Reviewer']),
            date=clean_text(row['Date']),
            error_summary=clean_text(row['ErrorSummary']),
            error_type=clean_text(row['ErrorType']),
            criticality=clean_text(row['Criticality']),
            rationale=clean_text(row['Rationale']),
        )

        full_text_for_embedding = "\n".join([
            review_text,
            f"Criticality: {meta.criticality}",
            f"Rationale: {meta.rationale}"
        ])

        embedding = get_embedding(full_text_for_embedding)

        insert_data_row(full_text_for_embedding, embedding, meta.__dict__)

        data_row.append(DocTextClass(
            chunk=review_text, meta=meta))

    state.qa_state.chunked_doc_text = data_row

    return state
