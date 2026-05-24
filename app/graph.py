from typing import TypedDict, List, Optional

from langchain.schema import Document


class GraphState(TypedDict):
    """
    Local PlayStation Manual RAG graph state.
    """

    question: str
    needs_ingestion: bool
    context: List[Document]
    answer: str
    sources: List[dict]
    vectorstore_path: Optional[str]
    search_metrics: Optional[dict]
    answer_metrics: Optional[dict]
    retry_retrieval: bool
    retry_generation: bool
    retrieval_attempts: int
    generation_attempts: int
