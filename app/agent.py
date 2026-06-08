"""
PlayStation User Manual RAG Agent (evaluated).
"""

from __future__ import annotations

from dotenv import load_dotenv
try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app.graph import GraphState
from app.config import ConfigurationError, load_config, require_openai_api_key, resolve_project_path

load_dotenv()


class RAGAgent:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        require_openai_api_key()

        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from evaluation.answer_evaluator import AnswerEvaluator
        from evaluation.search_evaluator import SearchEvaluator
        from evaluation.llm_judge import LLMJudge

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings"]["model_name"]
        )

        self.llm = ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
        )

        self.answer_evaluator = AnswerEvaluator()
        self.search_evaluator = SearchEvaluator()
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
        path = resolve_project_path(self.config["vectordb"]["persist_directory"])
        state["vectorstore_path"] = str(path)
        state["needs_ingestion"] = (not path.exists()) or (not any(path.iterdir()))
        return state

    def ingest_node(self, state: GraphState):
        if state.get("needs_ingestion"):
            from app.ingest import DocumentIngestion

            ingestion = DocumentIngestion(config_path="config.yaml")
            ingestion.run()
        return state

    def retrieve_node(self, state: GraphState):
        from langchain_community.vectorstores import Chroma

        question = str(state.get("question", "")).strip()
        if not question:
            raise ValueError("Question must not be empty.")

        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1

        db = Chroma(
            persist_directory=state["vectorstore_path"],
            embedding_function=self.embeddings,
            collection_name=self.config["vectordb"]["collection_name"],
        )

        top_k = self.config["retrieval"]["top_k"]
        eval_k = max(top_k * 2, top_k + 3)

        docs = db.similarity_search(
            question,
            k=top_k,
        )
        retrieval_pool = db.similarity_search(
            question,
            k=eval_k,
        )

        state["context"] = docs
        state["retrieval_pool"] = retrieval_pool
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
        retrieval_pool = state.get("retrieval_pool", docs)

        def doc_to_eval_row(doc):
            source = doc.metadata.get("source", "manual")
            page = doc.metadata.get("page", "N/A")
            return {
                "id": f"{source}:{page}:{abs(hash(doc.page_content))}",
                "text": doc.page_content,
            }

        retrieved_rows = [doc_to_eval_row(doc) for doc in docs]
        candidate_rows = [doc_to_eval_row(doc) for doc in retrieval_pool]
        state["search_metrics"] = self.search_evaluator.evaluate(
            state.get("question", ""),
            retrieved_rows,
            candidate_rows,
        )

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
        answer_metrics = self.answer_evaluator.evaluate(context_text, answer)
        llm_metrics = {
            "grounding": self.llm_judge.grounding_check(question, answer, context_text),
            "precision": self.llm_judge.precision_check(answer, context_text),
            "relevance": self.llm_judge.relevancy_check(question, answer),
        }

        state["answer_metrics"] = answer_metrics
        state["llm_metrics"] = llm_metrics

        semantic = float(answer_metrics.get("semantic", 0.0))
        relevance = float(llm_metrics.get("relevance", 0.0))
        grounding_score = float(llm_metrics.get("grounding", 0.0))
        precision_score = float(llm_metrics.get("precision", 0.0))

        state["retry_generation"] = (semantic < 0.4 or relevance < 0.4) and state.get(
            "generation_attempts", 0
        ) < 2

        confidence = (
            (0.35 * semantic)
            + (0.35 * grounding_score)
            + (0.2 * relevance)
            + (0.1 * precision_score)
        )
        state["answer_metrics"]["confidence"] = round(confidence, 4)

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
        question = (question or "").strip()
        if not question:
            return {
                "question": "",
                "answer": "Please enter a question about the PlayStation manuals.",
                "sources": [],
                "evaluation": {"confidence": 0},
            }

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
                    "llm_metrics": None,
                    "retry_retrieval": False,
                    "retry_generation": False,
                    "retrieval_attempts": 0,
                    "generation_attempts": 0,
                    "retrieval_pool": [],
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
                    "llm": result.get("llm_metrics"),
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
        except ConfigurationError as exc:
            return {
                "question": question,
                "answer": f"Configuration error: {exc}",
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
