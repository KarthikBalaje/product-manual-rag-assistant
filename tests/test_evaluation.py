"""Tests for evaluation utilities."""

from __future__ import annotations

from evaluation.search_evaluator import SearchEvaluator
from app.ui_helpers import build_confidence_formula, build_metrics_table, safe_float


def test_search_evaluator_scores_retrieved_docs() -> None:
    evaluator = SearchEvaluator(min_relevance=0.2)
    metrics = evaluator.evaluate(
        "reset controller",
        [{"id": "doc1", "text": "reset the controller"}],
        [{"id": "doc1", "text": "reset the controller"}, {"id": "doc2", "text": "network settings"}],
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["num_docs"] == 1


def test_safe_float_handles_bad_values() -> None:
    assert safe_float("0.5") == 0.5
    assert safe_float("not-a-number") == 0.0


def test_build_metrics_table_has_expected_rows() -> None:
    table = build_metrics_table({"search": {}, "answer": {}, "llm": {}})

    assert len(table) == 12
    assert {"Type", "Metric", "Purpose", "When", "Value"}.issubset(table.columns)


def test_build_confidence_formula_uses_values() -> None:
    formula, substituted = build_confidence_formula(
        {
            "answer": {"semantic": 0.5},
            "llm": {"grounding": 0.5, "relevance": 0.5, "precision": 0.5},
            "confidence": 0.5,
        }
    )

    assert "Confidence =" in formula
    assert "0.5000" in substituted
