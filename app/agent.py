"""
PlayStation User Manual RAG Agent (local PDFs).

Workflow:
- Ensure vectorstore exists (auto-ingest if missing)
- Retrieve top-k chunks from Chroma
- Generate an answer grounded ONLY in retrieved context
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from app.graph import GraphState
from app.ingest import DocumentIngestion

load_dotenv()


class RAGAgent:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings"]["model_name"]
        )

        self.llm = ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a PlayStation User Manual Assistant.

Answer ONLY using the provided manual context.

RULES:
1. Give step-by-step instructions when applicable
2. Do NOT make assumptions beyond the manual
3. If the answer is not found in the manual, say:
   "I couldn't find this in the provided PlayStation manual(s)."
4. Be precise and technical

Context:
{context}
""".strip(),
                ),
                ("human", "{question}"),
            ]
        )

    def check_vectorstore(self, state: GraphState):
        persist_dir = self.config["vectordb"]["persist_directory"]
        state["vectorstore_path"] = persist_dir

        # Heuristic: treat missing directory (or empty) as "needs ingestion".
        path = Path(persist_dir)
        state["needs_ingestion"] = not path.exists() or not any(path.iterdir())
        return state

    def ingest_node(self, state: GraphState):
        if state["needs_ingestion"]:
            ingestion = DocumentIngestion(
                config_path="config.yaml",
                strategy=self.config.get("chunking", {}).get("strategy", "manual"),
            )
            ingestion.run()
        return state

    def retrieve_node(self, state: GraphState):
        db = Chroma(
            persist_directory=state["vectorstore_path"],
            embedding_function=self.embeddings,
            collection_name=self.config["vectordb"]["collection_name"],
        )

        docs = db.similarity_search(
            state["question"],
            k=self.config["retrieval"]["top_k"],
        )

        state["context"] = docs
        state["sources"] = [
            {"source": d.metadata.get("source", "manual"), "page": d.metadata.get("page", "N/A")}
            for d in docs
        ]
        return state

    def generate_node(self, state: GraphState):
        if not state["context"]:
            state["answer"] = "I couldn't find this in the provided PlayStation manual(s)."
            return state

        context_text = "\n\n".join(
            [f"[Source: {d.metadata.get('source')} | Page {d.metadata.get('page')}]\n{d.page_content}" for d in state["context"]]
        )

        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_text, "question": state["question"]})
        state["answer"] = response.content
        return state

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("check", self.check_vectorstore)
        workflow.add_node("ingest", self.ingest_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)

        workflow.set_entry_point("check")

        # Always attempt ingestion if needed, otherwise retrieve directly.
        workflow.add_conditional_edges(
            "check",
            lambda s: "ingest" if s["needs_ingestion"] else "retrieve",
            {"ingest": "ingest", "retrieve": "retrieve"},
        )

        workflow.add_edge("ingest", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def query(self, question: str):
        graph = self.build_graph()
        result = graph.invoke(
            {
                "question": question,
                "needs_ingestion": False,
                "context": [],
                "answer": "",
                "sources": [],
                "vectorstore_path": None,
            }
        )
        return {"question": question, "answer": result["answer"], "sources": result["sources"]}

