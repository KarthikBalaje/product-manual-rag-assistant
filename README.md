# PlayStation Manual RAG Assistant

RAG assistant that answers questions from local PlayStation PDF manuals in `data/pdfs/`.

## What This Pipeline Does

1. Loads PDFs from `data/pdfs/`
2. Splits them into chunks (default: `manual` strategy)
3. Embeds chunks with `all-MiniLM-L6-v2`
4. Stores vectors in Chroma (`./vectorstore`, collection: `playstation_manual`)
5. Retrieves top-k chunks and generates grounded answers with OpenAI
6. Runs evaluation checks (search relevance + answer grounding/relevance)
7. Displays confidence and metric trends in Streamlit

## Prerequisites

- Python 3.10+ recommended
- OpenAI API key

## Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_key_here
```

## Run The RAG Pipeline

### 1) Put manuals in place

Add your PDF manuals to:

```text
data/pdfs/
```

### 2) Ingest PDFs into vector store

```powershell
python -m app.ingest manual
```

Other chunking strategies:

```powershell
python -m app.ingest semantic
python -m app.ingest recursive
python -m app.ingest token_based
python -m app.ingest agentic
```

### 3) Query from CLI

```powershell
python -m app.main
```

### 4) Query from Streamlit UI

```powershell
streamlit run streamlit_app.py
```

## Evaluation Mechanism

Each query returns:

- `evaluation.search.num_docs`
- `evaluation.answer.grounding`
- `evaluation.answer.relevance`
- `evaluation.answer.semantic`
- `evaluation.answer.confidence`

The graph also supports bounded retries for retrieval and generation.

## Tech Stack Notes

- Current code uses:
  - `langchain_community.embeddings.HuggingFaceEmbeddings`
  - `langchain_community.vectorstores.Chroma`
- This is the expected import path for the current project files.

## Run Smoke Tests

```powershell
python test_suite.py
```

## Configuration

Main settings are in `config.yaml`:

- `chunking` (strategy, size, overlap)
- `embeddings.model_name`
- `vectordb.persist_directory` and `collection_name`
- `retrieval.top_k`
- `llm.model` and `temperature`

## Project Structure

```text
product-manual-rag-assistant/
|-- app/
|   |-- agent.py
|   |-- ingest.py
|   |-- chunking_strategies.py
|   |-- graph.py
|   |-- main.py
|-- evaluation/
|   |-- search_evaluator.py
|   |-- answer_evaluator.py
|   |-- llm_judge.py
|   |-- evaluator_node.py
|-- data/
|   |-- pdfs/
|-- vectorstore/
|-- streamlit_app.py
|-- config.yaml
|-- test_suite.py
```

## Troubleshooting

- If ingestion/query returns no useful results, run ingestion again:
  `python -m app.ingest manual`
- If API key issues occur, verify `.env` is loaded and key is valid.
- Some environments show Chroma telemetry warnings; these are non-blocking for core RAG execution.
- If you see Streamlit `ScriptRunContext` warnings, run via:
  `streamlit run streamlit_app.py`
