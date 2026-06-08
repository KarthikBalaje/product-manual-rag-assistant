"""Tests for chunking helpers."""

from __future__ import annotations

import pytest
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from app.chunking_strategies import (
    AdvancedChunkingStrategies,
    AgenticChunker,
    compare_all_strategies,
    detect_content_type,
    detect_section_type,
)


class FakeLLM:
    def invoke(self, _prompt: str):
        return type("Response", (), {"content": "10, 30"})()


def test_detect_section_type() -> None:
    assert detect_section_type("WARNING: unplug the console") == "warning"
    assert detect_section_type("Reset the controller") == "reset"


def test_detect_content_type() -> None:
    assert detect_content_type("Step 1. Turn on the console") == "procedural"
    assert detect_content_type("The console supports this feature") == "informational"


def test_chunk_params_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        AdvancedChunkingStrategies.manual_chunking(chunk_size=100, chunk_overlap=100)


def test_agentic_chunker_returns_document_chunks() -> None:
    chunker = AgenticChunker(FakeLLM(), max_chunk_size=20, overlap=2)
    docs = chunker.split_documents([Document(page_content="a" * 60, metadata={"source": "manual"})])

    assert docs
    assert docs[0].metadata["source"] == "manual"
    assert "section_type" in docs[0].metadata


def test_compare_all_strategies_returns_stats() -> None:
    results = compare_all_strategies("Step 1. Connect the console.\n\nStep 2. Turn it on." * 30)

    assert "manual" in results
    assert results["manual"]["chunks"] >= 1
