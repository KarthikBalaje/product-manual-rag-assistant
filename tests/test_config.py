"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from app.config import ConfigurationError, load_config, validate_config


def valid_config() -> dict:
    return {
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "manual"},
        "embeddings": {"model_name": "all-MiniLM-L6-v2"},
        "vectordb": {"persist_directory": "./vectorstore", "collection_name": "playstation_manual"},
        "retrieval": {"top_k": 3},
        "llm": {"model": "gpt-4.1-mini", "temperature": 0.0},
        "app": {"log_level": "INFO"},
    }


def test_validate_config_accepts_valid_config() -> None:
    validate_config(valid_config())


def test_validate_config_rejects_bad_chunk_overlap() -> None:
    config = valid_config()
    config["chunking"]["chunk_overlap"] = 1000

    with pytest.raises(ConfigurationError):
        validate_config(config)


def test_validate_config_rejects_bad_strategy() -> None:
    config = valid_config()
    config["chunking"]["strategy"] = "random"

    with pytest.raises(ConfigurationError):
        validate_config(config)


def test_load_config_reads_project_config() -> None:
    config = load_config("config.yaml")

    assert config["vectordb"]["collection_name"] == "playstation_manual"

