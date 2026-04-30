from typing import Any, TypedDict
from dataclass import Document, ArticlePolicy, KeywordNode

class WorkflowState(TypedDict, total=False):
    # Inputs
    document_paths: list[str]
    chunk_chars: int
    overlap_chars: int

    # Mapping
    mapping_min_chars: int
    mapping_max_chunks: int
    mapping_candidate_chunks: int

    # Derived/working
    documents: list[Document]
    full_text: str
    # article_policies: list[ArticlePolicy]
    relevant_articles: list[ArticlePolicy]
    scope: dict[str, Any]
    chunks: list[dict[str, Any]]
    article_store: dict[int, list[dict[str, Any]]]

    # Dummy-RAG
    rag_articles: dict[int, dict[str, Any]]  # article_number -> {text, summary, used}

    # Checks + outputs
    findings: list[dict[str, Any]]
    hil_queue: list[dict[str, Any]]
    report: dict[str, Any]

