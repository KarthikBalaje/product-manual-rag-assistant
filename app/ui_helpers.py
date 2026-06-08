"""Pure UI helper functions shared by Streamlit and tests."""

from __future__ import annotations

import pandas as pd


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_metrics_table(evaluation: dict) -> pd.DataFrame:
    search = evaluation.get("search") or {}
    answer = evaluation.get("answer") or {}
    llm = evaluation.get("llm") or {}

    rows = [
        ("Search", "MRR", "First relevant doc", "After retrieval", safe_float(search.get("mrr"))),
        ("Search", "Precision", "Accuracy", "After retrieval", safe_float(search.get("precision"))),
        ("Search", "Recall", "Coverage", "After retrieval", safe_float(search.get("recall"))),
        ("Search", "F1", "Balance", "After retrieval", safe_float(search.get("f1"))),
        ("Search", "MAP", "Ranking quality", "After retrieval", safe_float(search.get("map"))),
        ("Search", "NDCG", "Position-aware ranking", "After retrieval", safe_float(search.get("ndcg"))),
        ("Answer", "ROUGE", "Word overlap", "After generation", safe_float(answer.get("rouge"))),
        ("Answer", "Fuzzy", "String similarity", "After generation", safe_float(answer.get("fuzzy"))),
        ("Answer", "Semantic", "Meaning similarity", "After generation", safe_float(answer.get("semantic"))),
        ("LLM", "Grounding", "Source support", "After generation", safe_float(llm.get("grounding"))),
        ("LLM", "Precision", "Hallucination detection", "After generation", safe_float(llm.get("precision"))),
        ("LLM", "Relevance", "Helpfulness", "After generation", safe_float(llm.get("relevance"))),
    ]

    metrics_df = pd.DataFrame(rows, columns=["Type", "Metric", "Purpose", "When", "Value"])
    metrics_df["Value"] = metrics_df["Value"].map(lambda value: f"{round(value, 4):.4f}")
    return metrics_df


def build_details_table(evaluation: dict) -> pd.DataFrame:
    search = evaluation.get("search") or {}
    meta = evaluation.get("meta") or {}

    rows = [
        ("Query", meta.get("question", "-")),
        ("Retrieved Chunks", search.get("num_docs", 0)),
        ("Relevant Chunks", search.get("relevant_docs", 0)),
        ("Source Pages", meta.get("source_count", 0)),
        ("Answer Length", meta.get("answer_length", 0)),
    ]
    return pd.DataFrame(rows, columns=["Detail", "Value"])


def build_confidence_formula(evaluation: dict) -> tuple[str, str]:
    answer = evaluation.get("answer") or {}
    llm = evaluation.get("llm") or {}

    semantic = safe_float(answer.get("semantic"))
    grounding = safe_float(llm.get("grounding"))
    relevance = safe_float(llm.get("relevance"))
    precision = safe_float(llm.get("precision"))
    confidence = safe_float(evaluation.get("confidence"))

    formula = (
        "Confidence = (0.35 x Semantic) + (0.35 x Grounding) + "
        "(0.20 x Relevance) + (0.10 x Precision)"
    )
    substituted = (
        f"= (0.35 x {semantic:.4f}) + (0.35 x {grounding:.4f}) + "
        f"(0.20 x {relevance:.4f}) + (0.10 x {precision:.4f}) = {confidence:.4f}"
    )
    return formula, substituted

