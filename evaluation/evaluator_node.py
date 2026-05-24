from evaluation.search_evaluator import SearchEvaluator
from evaluation.answer_evaluator import AnswerEvaluator
from evaluation.llm_judge import LLMJudge


class EvaluatorNode:

    def __init__(self):
        self.search_eval = SearchEvaluator()
        self.answer_eval = AnswerEvaluator()
        self.llm_judge = LLMJudge()

    def evaluate_all(self, query, retrieved_docs, relevant_docs, answer, ground_truth):
        retrieved_texts = [self._doc_to_text(doc) for doc in retrieved_docs]
        relevant_texts = [self._doc_to_text(doc) for doc in relevant_docs]
        context_text = "\n\n".join(retrieved_texts)

        search_metrics = self.search_eval.evaluate(
            retrieved_texts,
            relevant_texts,
            [1 if doc in relevant_texts else 0 for doc in retrieved_texts],
        )

        answer_metrics = self.answer_eval.evaluate(ground_truth, answer)

        llm_checks = {
            "grounding": self.llm_judge.grounding_check(query, answer, context_text),
            "hallucination": self.llm_judge.hallucination_check(answer, context_text),
            "relevancy": self.llm_judge.relevancy_check(query, answer),
        }

        return {
            "search": search_metrics,
            "answer": answer_metrics,
            "llm": llm_checks,
        }

    @staticmethod
    def _doc_to_text(doc) -> str:
        if hasattr(doc, "page_content"):
            return doc.page_content
        return str(doc)
