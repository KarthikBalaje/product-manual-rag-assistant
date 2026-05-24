"""
PlayStation User Manual RAG Agent (evaluated).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app.graph import GraphState
from app.ingest import DocumentIngestion
from evaluation.answer_evaluator import AnswerEvaluator
from evaluation.llm_judge import LLMJudge

load_dotenv()


class RAGAgent:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings"]["model_name"]
        )

        self.llm = ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
        )

        self.answer_evaluator = AnswerEvaluator()
        self.llm_judge = LLMJudge()

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
3. If the answer is not found, say:
   "I couldn't find this in the provided PlayStation manual(s)."
4. Be precise and technical

Context:
{context}
""".strip(),
                ),
                ("human", "{question}"),
            ]
        )

        self.graph = self.build_graph()

    # -------------------------
    # GRAPH NODES
    # -------------------------

    def check_vectorstore(self, state: GraphState):
        path = Path(self.config["vectordb"]["persist_directory"])
        state["vectorstore_path"] = str(path)
        state["needs_ingestion"] = (not path.exists()) or (not any(path.iterdir()))
        return state

    def ingest_node(self, state: GraphState):
        if state.get("needs_ingestion"):
            ingestion = DocumentIngestion(config_path="config.yaml")
            ingestion.run()
        return state

    def retrieve_node(self, state: GraphState):
        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1

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
            {
                "source": d.metadata.get("source", "manual"),
                "page": d.metadata.get("page", "N/A"),
            }
            for d in docs
        ]
        return state

    def evaluate_search_node(self, state: GraphState):
        docs = state.get("context", [])

        state["search_metrics"] = {"num_docs": len(docs)}

        # Retry retrieval at most once if we got nothing.
        state["retry_retrieval"] = len(docs) == 0 and state.get("retrieval_attempts", 0) < 2
        return state

    def generate_node(self, state: GraphState):
        state["generation_attempts"] = state.get("generation_attempts", 0) + 1

        if not state.get("context"):
            state["answer"] = "I couldn't find this in the provided PlayStation manual(s)."
            return state

        context_text = "\n\n".join(
            [
                f"[Source: {d.metadata.get('source')} | Page {d.metadata.get('page')}]\n{d.page_content}"
                for d in state["context"]
            ]
        )

        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_text, "question": state["question"]})
        state["answer"] = response.content
        return state

    def evaluate_answer_node(self, state: GraphState):
        answer = state.get("answer", "")
        question = state.get("question", "")
        docs = state.get("context", [])

        context_text = "\n".join([d.page_content for d in docs])

        grounding = self.llm_judge.grounding_check(question, answer, context_text)
        relevance = self.llm_judge.relevancy_check(question, answer)
        semantic = self.answer_evaluator.semantic_similarity(context_text, answer)

        state["answer_metrics"] = {
            "grounding": grounding,
            "relevance": relevance,
            "semantic": semantic,
        }

        state["retry_generation"] = (semantic < 0.4 or "0" in str(relevance)) and state.get(
            "generation_attempts", 0
        ) < 2

        try:
            grounding_score = float(str(grounding).strip())
        except Exception:
            grounding_score = 0.0

        try:
            relevance_score = float(str(relevance).strip())
        except Exception:
            relevance_score = 0.0

        confidence = (0.4 * float(semantic)) + (0.4 * grounding_score) + (0.2 * relevance_score)
        state["answer_metrics"]["confidence"] = round(confidence, 2)

        if state.get("generation_attempts", 0) >= 2 and state.get("retry_generation"):
            state["answer"] = "I couldn't confidently find this in the manuals."
            state["retry_generation"] = False

        # If we generated a non-fallback answer, accept it.
        if answer and "couldn't find" not in answer.lower():
            state["retry_generation"] = False

        return state

    # -------------------------
    # GRAPH BUILDING
    # -------------------------

    def build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("check", self.check_vectorstore)
        workflow.add_node("ingest", self.ingest_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("eval_search", self.evaluate_search_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("eval_answer", self.evaluate_answer_node)

        workflow.set_entry_point("check")

        workflow.add_conditional_edges(
            "check",
            lambda s: "ingest" if s["needs_ingestion"] else "retrieve",
            {"ingest": "ingest", "retrieve": "retrieve"},
        )

        workflow.add_edge("ingest", "retrieve")
        workflow.add_edge("retrieve", "eval_search")

        workflow.add_conditional_edges(
            "eval_search",
            lambda s: "retrieve" if s.get("retry_retrieval") else "generate",
            {"retrieve": "retrieve", "generate": "generate"},
        )

        workflow.add_edge("generate", "eval_answer")

        workflow.add_conditional_edges(
            "eval_answer",
            lambda s: "generate" if s.get("retry_generation") else "end",
            {"generate": "generate", "end": END},
        )

        return workflow.compile()

    # -------------------------
    # QUERY ENTRY
    # -------------------------

    def query(self, question: str):
        try:
            result = self.graph.invoke(
                {
                    "question": question,
                    "needs_ingestion": False,
                    "context": [],
                    "answer": "",
                    "sources": [],
                    "vectorstore_path": None,
                    "search_metrics": None,
                    "answer_metrics": None,
                    "retry_retrieval": False,
                    "retry_generation": False,
                    "retrieval_attempts": 0,
                    "generation_attempts": 0,
                },
                config={"recursion_limit": 10},
            )

            return {
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "evaluation": {
                    "search": result.get("search_metrics"),
                    "answer": result.get("answer_metrics"),
                    "confidence": (result.get("answer_metrics") or {}).get("confidence", 0),
                },
            }
        except GraphRecursionError:
            return {
                "question": question,
                "answer": "I'm having trouble finding a reliable answer right now. Please try rephrasing your question.",
                "sources": [],
                "evaluation": {"confidence": 0},
            }
        except Exception:
            return {
                "question": question,
                "answer": "Something went wrong while processing your request. Please try again.",
                "sources": [],
                "evaluation": {"confidence": 0},
            }
