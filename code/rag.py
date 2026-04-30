from typing import Any

def dummy_rag_fetch_article(article_number: int) -> dict[str, Any]:
    """Dummy RAG: pretend we retrieved the GDPR article text.

    - Returns `text` for small articles
    - Returns `summary` for large articles (simulated)

    Replace this with a real retriever later (vector DB, files, API, etc.).
    """
    # Simulate size by a deterministic rule: every 3rd article is "large".
    is_large = (article_number % 3) == 0
    if is_large:
        return {
            "used": "summary",
            "text": None,
            "summary": (
                f"Compressed summary for GDPR Article {article_number}. "
                "Key obligations are condensed for checker input."
            ),
        }
    return {
        "used": "text",
        "text": (
            f"Full retrieved text for GDPR Article {article_number}. "
            "This is dummy placeholder content for the workflow."
        ),
        "summary": None,
    }

