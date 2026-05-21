# PlayStation Manual RAG Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG assistant that answers questions from **PlayStation user manual PDFs** stored locally in `data/pdfs/`.
Answers are grounded in the manuals and include source + page citations.

---

## What It Does

- Loads PDFs from `data/pdfs/*.pdf` (example: `PS5.pdf`, `PS4.pdf`, etc.)
- Chunks the manuals (default: `manual` chunking to keep procedures intact)
- Embeds with local SentenceTransformers (`all-MiniLM-L6-v2`)
- Stores vectors in ChromaDB (`./vectorstore`, collection `playstation_manual`)
- Retrieves top-k chunks and generates a grounded answer (OpenAI)

---

## Quick Start

### Prerequisites

- Python 3.8+
- `OPENAI_API_KEY` set (in `.env` or environment variable)

### Install

```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### How To Run (Windows PowerShell)

From the project folder:

```bash
# 1) Create + activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your API key (in .env or as an environment variable)
# .env example:
# OPENAI_API_KEY=sk-...

# 4) Ingest PDFs from data/pdfs/
python -m app.ingest manual

# 5) Run the app
# CLI:
python -m app.main

# Streamlit:
streamlit run streamlit_app.py
```

### Ingest Manuals

Put your PlayStation manual PDFs in `data/pdfs/` and run:

```bash
python -m app.ingest manual
```

Other strategies:

```bash
python -m app.ingest semantic
python -m app.ingest recursive
python -m app.ingest token_based
python -m app.ingest agentic
```

### Ask Questions (CLI)

```bash
python -m app.main
```

### Ask Questions (Streamlit)

```bash
streamlit run streamlit_app.py
```

---

## Configuration

Edit `config.yaml`:

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  strategy: "manual"

embeddings:
  model_name: "all-MiniLM-L6-v2"

vectordb:
  persist_directory: "./vectorstore"
  collection_name: "playstation_manual"

retrieval:
  top_k: 3

llm:
  model: "gpt-4.1-mini"
  temperature: 0.0
```

---

## Project Structure

```
product-manual-rag-assistant/
├── app/
│   ├── agent.py                # RAG agent (retrieve + generate)
│   ├── ingest.py               # Local PDF ingestion pipeline
│   ├── chunking_strategies.py  # Chunking strategies (manual/semantic/...)
│   ├── graph.py                # LangGraph state
│   └── main.py                 # CLI chat
├── data/
│   └── pdfs/                    # Put PlayStation manual PDFs here
├── vectorstore/                 # Chroma persistence
├── streamlit_app.py
└── config.yaml
```

---

## Troubleshooting

### No PDFs found

- Ensure manuals exist in `data/pdfs/` and are real text PDFs (not image-only scans).

### OPENAI_API_KEY missing

- Add `OPENAI_API_KEY=...` to `.env` (or set it in your shell).

### pypdf AES / cryptography error

- If you see `cryptography>=3.1 is required for AES algorithm`, install dependencies again:
  `pip install -r requirements.txt`
