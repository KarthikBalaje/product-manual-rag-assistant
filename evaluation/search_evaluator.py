import re

import numpy as np


class SearchEvaluator:
    def __init__(self, min_relevance: float = 0.35):
        self.min_relevance = min_relevance

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b[a-z0-9]{2,}\b", (text or "").lower()))

    def _coverage_score(self, query: str, document_text: str) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        doc_tokens = self._tokenize(document_text)
        if not doc_tokens:
            return 0.0

        return len(query_tokens & doc_tokens) / len(query_tokens)

    @staticmethod
    def precision(retrieved, relevant):
        true_positives = len(set(retrieved) & set(relevant))
        return true_positives / len(retrieved) if retrieved else 0.0

    @staticmethod
    def recall(retrieved, relevant):
        true_positives = len(set(retrieved) & set(relevant))
        return true_positives / len(relevant) if relevant else 0.0

    @staticmethod
    def f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mrr(retrieved, relevant):
        for index, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1 / index
        return 0.0

    @staticmethod
    def average_precision(retrieved, relevant):
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        hit_count = 0
        precision_sum = 0.0
        seen_relevant = set()

        for index, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set and doc_id not in seen_relevant:
                seen_relevant.add(doc_id)
                hit_count += 1
                precision_sum += hit_count / index

        return precision_sum / len(relevant_set)

    @staticmethod
    def dcg(relevance_scores):
        return sum(score / np.log2(index + 2) for index, score in enumerate(relevance_scores))

    def ndcg(self, relevance_scores):
        ideal = sorted(relevance_scores, reverse=True)
        ideal_dcg = self.dcg(ideal)
        if not ideal or ideal_dcg == 0:
            return 0.0
        return self.dcg(relevance_scores) / ideal_dcg

    def evaluate(self, query: str, retrieved_docs: list[dict], candidate_docs: list[dict]):
        deduped_candidates = {}
        for doc in candidate_docs:
            doc_id = doc["id"]
            coverage = self._coverage_score(query, doc.get("text", ""))
            existing = deduped_candidates.get(doc_id)
            if existing is None or coverage > existing["coverage"]:
                deduped_candidates[doc_id] = {
                    **doc,
                    "coverage": coverage,
                }

        scored_candidates = list(deduped_candidates.values())

        relevant_ids = [
            doc["id"]
            for doc in scored_candidates
            if doc["coverage"] >= self.min_relevance
        ]

        if not relevant_ids and scored_candidates:
            best_doc = max(scored_candidates, key=lambda item: item["coverage"])
            if best_doc["coverage"] > 0:
                relevant_ids = [best_doc["id"]]

        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        retrieved_scores = []
        for doc in retrieved_docs:
            matched_doc = deduped_candidates.get(doc["id"])
            retrieved_scores.append(matched_doc["coverage"] if matched_doc else 0.0)

        precision = self.precision(retrieved_ids, relevant_ids)
        recall = self.recall(retrieved_ids, relevant_ids)

        return {
            "mrr": round(self.mrr(retrieved_ids, relevant_ids), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(self.f1(precision, recall), 4),
            "map": round(self.average_precision(retrieved_ids, relevant_ids), 4),
            "ndcg": round(self.ndcg(retrieved_scores), 4),
            "num_docs": len(retrieved_docs),
            "relevant_docs": len(relevant_ids),
        }
