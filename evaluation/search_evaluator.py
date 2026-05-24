import numpy as np


class SearchEvaluator:

    def precision(self, retrieved, relevant):
        tp = len(set(retrieved) & set(relevant))
        return tp / len(retrieved) if retrieved else 0

    def recall(self, retrieved, relevant):
        tp = len(set(retrieved) & set(relevant))
        return tp / len(relevant) if relevant else 0

    def f1(self, precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def mrr(self, retrieved, relevant):
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1 / (i + 1)
        return 0

    def dcg(self, relevance_scores):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

    def ndcg(self, relevance_scores):
        ideal = sorted(relevance_scores, reverse=True)
        ideal_dcg = self.dcg(ideal)
        if not ideal or ideal_dcg == 0:
            return 0
        return self.dcg(relevance_scores) / ideal_dcg

    def evaluate(self, retrieved, relevant, relevance_scores):
        p = self.precision(retrieved, relevant)
        r = self.recall(retrieved, relevant)

        return {
            "precision": p,
            "recall": r,
            "f1": self.f1(p, r),
            "mrr": self.mrr(retrieved, relevant),
            "ndcg": self.ndcg(relevance_scores)
        }
