# PlayStation Manual RAG Assistant

RAG assistant that answers questions from local PlayStation PDF manuals in `data/pdfs/`.

## What This App Does

1. Loads PDF manuals from `data/pdfs/`
2. Splits documents into chunks
3. Embeds chunks with `all-MiniLM-L6-v2`
4. Stores vectors in Chroma at `./vectorstore`
5. Retrieves top matching chunks for a user query
6. Generates grounded answers with OpenAI
7. Evaluates retrieval quality, answer similarity, and LLM-based answer quality
8. Shows a two-pane Streamlit UI:
   Left pane: application details, evaluation metrics table, hover formulas, confidence score, confidence formula
   Right pane: chat interface with answer and sources

## Prerequisites

- Python 3.10+ recommended
- OpenAI API key

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
```

## Run The App

### 1) Add manuals

Place PDF manuals in:

```text
data/pdfs/
```

### 2) Ingest manuals into the vector store

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

### 3) Run the CLI app

```powershell
python -m app.main
```

### 4) Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

## Streamlit UI

The Streamlit app uses a wide two-column layout.

- Left pane shows:
  - `Application Details`
  - `Evaluation Metrics`
  - metric values in a table with columns `Type`, `Metric`, `Purpose`, `When`, `Value`
  - hover tooltips that display the formula or scoring definition for each metric
  - `Confidence Score`
  - confidence calculation formula and substituted values
- Right pane shows:
  - chat conversation
  - latest answer
  - cited source pages

## Evaluation Metrics

The app computes and displays the following metrics for each query:

| Type   | Metric    | Purpose                 | When             |
| ------ | --------- | ----------------------- | ---------------- |
| Search | MRR       | First relevant doc      | After retrieval  |
| Search | Precision | Accuracy                | After retrieval  |
| Search | Recall    | Coverage                | After retrieval  |
| Search | F1        | Balance                 | After retrieval  |
| Search | MAP       | Ranking quality         | After retrieval  |
| Search | NDCG      | Position-aware ranking  | After retrieval  |
| Answer | ROUGE     | Word overlap            | After generation |
| Answer | Fuzzy     | String similarity       | After generation |
| Answer | Semantic  | Meaning similarity      | After generation |
| LLM    | Grounding | Source support          | After generation |
| LLM    | Precision | Hallucination detection | After generation |
| LLM    | Relevance | Helpfulness             | After generation |

### Confidence Formula

The app computes confidence from the latest generated answer using:

```text
Confidence = (0.35 x Semantic) + (0.35 x Grounding) + (0.20 x Relevance) + (0.10 x Precision)
```

The Streamlit UI also shows the substituted formula values for the latest query.

## Evaluation Payload

Each query returns an evaluation object shaped like:

```text
evaluation.search
evaluation.answer
evaluation.llm
evaluation.confidence
```

Common fields include:

- `evaluation.search.mrr`
- `evaluation.search.precision`
- `evaluation.search.recall`
- `evaluation.search.f1`
- `evaluation.search.map`
- `evaluation.search.ndcg`
- `evaluation.search.num_docs`
- `evaluation.search.relevant_docs`
- `evaluation.answer.rouge`
- `evaluation.answer.rouge1`
- `evaluation.answer.rougeL`
- `evaluation.answer.fuzzy`
- `evaluation.answer.semantic`
- `evaluation.answer.confidence`
- `evaluation.llm.grounding`
- `evaluation.llm.precision`
- `evaluation.llm.relevance`

## Notes On Metric Quality

- Search metrics in this project are heuristic because the app does not yet use labeled ground-truth relevant documents per query.
- Answer metrics compare the generated answer against retrieved context, not against a curated gold reference answer set.
- LLM metrics are judge-model scores from `0.0` to `1.0`.

These metrics are useful for application monitoring and debugging, but they are not a substitute for a benchmark dataset.

## Tech Stack Notes

- `langchain`
- `langgraph`
- `chromadb`
- `sentence-transformers`
- `langchain-openai`
- `openai`
- `streamlit`
- `rapidfuzz`
- `rouge-score`

Current vector and embedding integrations use:

- `langchain_community.embeddings.HuggingFaceEmbeddings`
- `langchain_community.vectorstores.Chroma`

## Run Smoke Tests

```powershell
python test_suite.py
```

## Configuration

Main settings are in `config.yaml`:

- `chunking` strategy, size, overlap
- `embeddings.model_name`
- `vectordb.persist_directory`
- `vectordb.collection_name`
- `retrieval.top_k`
- `llm.model`
- `llm.temperature`

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
|-- README.md
```

## Troubleshooting

- If retrieval quality looks poor, re-run ingestion:
  `python -m app.ingest manual`
- If `OPENAI_API_KEY` is missing, set it in `.env` or your deployment environment.
- If you see Streamlit `ScriptRunContext` warnings, run the app with:
  `streamlit run streamlit_app.py`
- Some Chroma telemetry warnings are non-blocking.
