"""Tests for RAG agent request validation."""

from __future__ import annotations

from app.agent import RAGAgent


def test_query_rejects_empty_question_without_graph(monkeypatch) -> None:
    monkeypatch.setattr(RAGAgent, "__init__", lambda self: None)
    agent = RAGAgent()

    result = agent.query(" ")

    assert "Please enter a question" in result["answer"]
    assert result["evaluation"]["confidence"] == 0

