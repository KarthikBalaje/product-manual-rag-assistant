"""Configuration loading and validation for the PlayStation Manual RAG app."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SUPPORTED_CHUNKING_STRATEGIES = {"manual", "semantic", "recursive", "token_based", "agentic"}


class ConfigurationError(RuntimeError):
    """Raised when app configuration is missing or invalid."""


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate YAML config."""

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config sections and values."""

    required_sections = {"chunking", "embeddings", "vectordb", "retrieval", "llm", "app"}
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ConfigurationError(f"Missing config sections: {sorted(missing_sections)}")

    chunking = config["chunking"]
    chunk_size = int(chunking.get("chunk_size", 0))
    chunk_overlap = int(chunking.get("chunk_overlap", 0))
    strategy = chunking.get("strategy", "manual")
    if chunk_size <= 0:
        raise ConfigurationError("chunking.chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ConfigurationError("chunking.chunk_overlap must be >= 0 and smaller than chunk_size.")
    if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ConfigurationError(f"chunking.strategy must be one of {sorted(SUPPORTED_CHUNKING_STRATEGIES)}.")

    if not str(config["embeddings"].get("model_name", "")).strip():
        raise ConfigurationError("embeddings.model_name is required.")

    vectordb = config["vectordb"]
    if not str(vectordb.get("persist_directory", "")).strip():
        raise ConfigurationError("vectordb.persist_directory is required.")
    if not str(vectordb.get("collection_name", "")).strip():
        raise ConfigurationError("vectordb.collection_name is required.")

    top_k = int(config["retrieval"].get("top_k", 0))
    if top_k <= 0:
        raise ConfigurationError("retrieval.top_k must be greater than zero.")

    if not str(config["llm"].get("model", "")).strip():
        raise ConfigurationError("llm.model is required.")
    temperature = float(config["llm"].get("temperature", 0.0))
    if not 0 <= temperature <= 2:
        raise ConfigurationError("llm.temperature must be between 0 and 2.")


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a clear configuration error."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key in {"your_key_here", "your-openai-api-key-here"}:
        raise ConfigurationError("OPENAI_API_KEY is missing. Add it to .env or your environment.")
    return api_key


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

