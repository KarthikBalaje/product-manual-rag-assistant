import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.agent import RAGAgent

load_dotenv()

st.set_page_config(
    page_title="PlayStation Manual Assistant",
    page_icon="🎮",
    layout="centered",
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
    return f"**{name}** — page {page}"


with st.sidebar:
    st.title("PlayStation Manual Assistant")
    st.caption("Grounded answers from PDFs in data/pdfs/")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("PlayStation User Manual RAG")
st.caption("Answers cite the manual pages used. If it's not in the manuals, it will say so.")

if not check_api_key():
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "Add it to a `.env` file locally, or set it as an environment variable in Railway."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"- {format_source(s)}")

import pandas as pd

agent = load_agent()

if prompt := st.chat_input("Ask a question about PlayStation manuals…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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
        confidence = evaluation.get("confidence", 0)

        st.markdown(answer)

        # -------------------------
        # 📚 Sources
        # -------------------------
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.markdown(f"- {format_source(s)}")

        # -------------------------
        # 📊 Confidence Score
        # -------------------------
        st.markdown("### 📊 Confidence Score")

        st.progress(float(confidence))

        if confidence > 0.75:
            st.success(f"High Confidence: {confidence}")
        elif confidence > 0.5:
            st.warning(f"Medium Confidence: {confidence}")
        else:
            st.error(f"Low Confidence: {confidence}")
            
        if confidence < 0.4:
            st.warning("⚠️ This answer may not be fully reliable. Try rephrasing your question.")
        # -------------------------
        # 📈 Evaluation Metrics Chart
        # -------------------------
        answer_metrics = evaluation.get("answer", {})

        if answer_metrics:
            st.markdown("### 📈 Evaluation Metrics")

            try:
                df = pd.DataFrame({
                    "Metric": ["Semantic", "Grounding", "Relevance"],
                    "Score": [
                        answer_metrics.get("semantic", 0),
                        float(answer_metrics.get("grounding", 0)),
                        float(answer_metrics.get("relevance", 0))
                    ]
                })

                st.bar_chart(df.set_index("Metric"))

            except Exception:
                st.info("Metrics could not be visualized.")

        # -------------------------
        # 🎯 Metric Breakdown
        # -------------------------
        st.markdown("### 🎯 Metric Breakdown")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Semantic", round(answer_metrics.get("semantic", 0), 2))

        with col2:
            st.metric("Grounding", answer_metrics.get("grounding", 0))

        with col3:
            st.metric("Relevance", answer_metrics.get("relevance", 0))

        # -------------------------
        # 📉 Trend Dashboard (Session)
        # -------------------------
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "confidence": confidence,
            "semantic": answer_metrics.get("semantic", 0)
        })

        history_df = pd.DataFrame(st.session_state.history)

        if len(history_df) > 1:
            st.markdown("### 📉 Confidence Trend")
            st.line_chart(history_df)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
