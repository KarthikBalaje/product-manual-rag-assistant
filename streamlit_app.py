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

agent = load_agent()

if prompt := st.chat_input("Ask a question about PlayStation manuals…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching manuals…"):
            result = agent.query(prompt)

        answer = result["answer"]
        sources = result.get("sources", [])

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.markdown(f"- {format_source(s)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
