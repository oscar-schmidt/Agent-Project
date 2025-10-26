from typing import List
import os
from dotenv import load_dotenv
import numpy as np
from backend.model.states.graph_state.GraphState import GraphState
from backend.model.states.qa_state.DocTextClass import Meta, DocTextClass
from backend.utils import get_embedding, log_decorator

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = int(os.getenv("TOP_K", 50))


@log_decorator
def rag_retrieval_node(state: GraphState, collection, embeded_query) -> tuple[List[DocTextClass], str, float]:
    try:
        num = collection.count()
        result = collection.query(
            query_embeddings=embeded_query, n_results=TOP_K, include=["documents", "distances", "metadatas"])
    except Exception as e:
        print(e)
        return [], None, 0.0

    if result.get("documents"):
        top_k_result = top_k_result_to_log(state, result)
        top_k_kb = "\n".join([pdf_text.chunk for pdf_text in top_k_result])
        distances = np.array(result['distances'][0])
        top_k_scores = np.exp(-distances)
        top_score = top_k_scores.max()

        # top_3_avg_score = sum(sorted(top_k_scores, reverse=True)
        #                       [:3])/min(3, len(top_k_scores))
    else:
        top_k_kb = None
        top_score = 0.0
    return top_k_result, top_k_kb, top_score


def top_k_result_to_log(state, query) -> List[DocTextClass]:
    top_k_result: List[DocTextClass] = []
    for doc_list, meta_list in zip(query.get('documents', []), query.get('metadatas', [])):
        for doc, meta in zip(doc_list, meta_list):
            pdf_text_obj = DocTextClass(
                chunk=doc,
                meta=Meta(**meta)
            )
            pdf_text_obj.meta.doc_name = state.qa_state.doc_name
            top_k_result.append(pdf_text_obj)
            state.logs.append(
                f"[TOP_{TOP_K}_RESULT] doc_name: {pdf_text_obj.meta.doc_name}, "
                f"page_number: {pdf_text_obj.meta.referenece_number}, "
                # f"chunk_summary: {pdf_text_obj.meta.chunk_summary}"
                f"chunk_content: {pdf_text_obj.chunk}"
            )
    return top_k_result
