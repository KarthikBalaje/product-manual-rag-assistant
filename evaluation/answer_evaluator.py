from rapidfuzz import fuzz
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


class AnswerEvaluator:

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def rouge_score(self, reference, generated):
        scores = self.rouge.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }

    def fuzzy_score(self, reference, generated):
        return fuzz.partial_ratio(reference, generated)

    def semantic_similarity(self, reference, generated):
        emb1 = self.model.encode(reference, convert_to_tensor=True)
        emb2 = self.model.encode(generated, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())

    def evaluate(self, reference, generated):
        return {
            **self.rouge_score(reference, generated),
            "fuzzy": self.fuzzy_score(reference, generated),
            "semantic": self.semantic_similarity(reference, generated)
        }
