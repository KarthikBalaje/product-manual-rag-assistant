import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.agent import RAGAgent

load_dotenv()

st.set_page_config(
    page_title="PlayStation Manual Assistant",
    page_icon="🎮",
    layout="wide",
)


def check_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner=False)
def load_agent() -> RAGAgent:
    return RAGAgent()


def format_source(source_meta: dict) -> str:
    raw = source_meta.get("source", "manual")
    name = Path(raw).stem.replace("-", " ").replace("_", " ").title()
    page = source_meta.get("page", "N/A")
    return f"**{name}** - page {page}"


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


def metric_formulas() -> dict[str, str]:
    return {
        "MRR": "MRR = 1 / rank of the first relevant retrieved document",
        "Precision": "Precision = relevant retrieved documents / total retrieved documents",
        "Recall": "Recall = relevant retrieved documents / total relevant documents",
        "F1": "F1 = 2 x (Precision x Recall) / (Precision + Recall)",
        "MAP": "MAP = average of precision values at ranks where relevant documents appear",
        "NDCG": "NDCG = DCG / IDCG, where DCG = sum(relevance / log2(rank + 1))",
        "ROUGE": "ROUGE = average overlap score derived from ROUGE-1 and ROUGE-L F1",
        "Fuzzy": "Fuzzy = RapidFuzz partial_ratio(reference, answer) / 100",
        "Semantic": "Semantic = cosine similarity between embedding(reference) and embedding(answer)",
        "Grounding": "Grounding = LLM judge score from 0 to 1 for how well the answer is supported by retrieved context",
        "Relevance": "Relevance = LLM judge score from 0 to 1 for how directly the answer addresses the question",
    }


def formula_for_row(metric_type: str, metric_name: str) -> str:
    if metric_type == "LLM" and metric_name == "Precision":
        return (
            "LLM Precision = LLM judge score from 0 to 1 for factual support and lack of hallucinated claims"
        )

    return metric_formulas().get(metric_name, "Formula not available")


def render_metrics_table(evaluation: dict) -> None:
    metrics_df = build_metrics_table(evaluation)
    rows_html = []

    for row in metrics_df.to_dict(orient="records"):
        formula = escape(formula_for_row(row["Type"], row["Metric"]))
        row_cells = "".join(
            [
                f"<td title=\"{formula}\">{escape(str(row['Type']))}</td>",
                f"<td title=\"{formula}\">{escape(str(row['Metric']))}</td>",
                f"<td title=\"{formula}\">{escape(str(row['Purpose']))}</td>",
                f"<td title=\"{formula}\">{escape(str(row['When']))}</td>",
                f"<td title=\"{formula}\">{escape(str(row['Value']))}</td>",
            ]
        )
        rows_html.append(f"<tr>{row_cells}</tr>")

    table_html = f"""
    <style>
      .metrics-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
      }}
      .metrics-table th, .metrics-table td {{
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 0.45rem 0.5rem;
        text-align: left;
        vertical-align: top;
      }}
      .metrics-table th {{
        background: rgba(128, 128, 128, 0.12);
        font-weight: 600;
      }}
      .metrics-table tr:hover td {{
        background: rgba(70, 130, 180, 0.08);
      }}
    </style>
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Type</th>
          <th>Metric</th>
          <th>Purpose</th>
          <th>When</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("Hover over any metric row cell to see the formula used for that metric.")


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


def render_chat_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"- {format_source(source)}")


with st.sidebar:
    st.title("PlayStation Manual Assistant")
    st.caption("Grounded answers from PDFs in data/pdfs/")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_evaluation = {}
        st.rerun()


st.title("PlayStation User Manual RAG")
st.caption("Metrics appear on the left for the latest query, and the chat stays on the right.")

if not check_api_key():
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "Add it to a `.env` file locally, or set it as an environment variable in Railway."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_evaluation" not in st.session_state:
    st.session_state.latest_evaluation = {}

agent = load_agent()

metrics_col, chat_col = st.columns([1.15, 2.2], gap="large")

with metrics_col:
    latest_evaluation = st.session_state.latest_evaluation or {}

    st.subheader("Application Details")
    st.dataframe(
        build_details_table(latest_evaluation),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Evaluation Metrics")
    st.caption("Each row shows the live metric value for the latest query, retrieval step, and generated answer.")

    render_metrics_table(latest_evaluation)

    confidence = safe_float(latest_evaluation.get("confidence"))
    st.markdown("### Confidence")
    st.progress(max(0.0, min(confidence, 1.0)))
    st.metric("Confidence Score", round(confidence, 4))

    formula, substituted = build_confidence_formula(latest_evaluation)
    st.caption("Confidence formula")
    st.code(formula, language="text")
    st.code(substituted, language="text")

    if confidence >= 0.75:
        st.success("High confidence response")
    elif confidence >= 0.5:
        st.warning("Medium confidence response")
    elif latest_evaluation:
        st.error("Low confidence response")
        st.caption("Try rephrasing the question if the answer looks incomplete.")

with chat_col:
    st.subheader("Chat")

    for message in st.session_state.messages:
        render_chat_message(message)

    prompt = st.chat_input("Ask a question about PlayStation manuals...")

    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        render_chat_message(user_message)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching manuals..."):
                    result = agent.query(prompt)
            except Exception as exc:
                st.error(f"System error while answering your question: {exc}")
                st.stop()

            answer = result["answer"]
            sources = result.get("sources", [])
            evaluation = result.get("evaluation", {})
            evaluation["meta"] = {
                "question": prompt,
                "source_count": len(sources),
                "answer_length": len(answer.split()),
            }

            st.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=False):
                    for source in sources:
                        st.markdown(f"- {format_source(source)}")

        st.session_state.latest_evaluation = evaluation
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        st.rerun()
