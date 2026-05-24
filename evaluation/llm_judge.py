from openai import OpenAI


class LLMJudge:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o-mini"

    @staticmethod
    def _clamp_score(value) -> float:
        try:
            return max(0.0, min(1.0, float(str(value).strip())))
        except Exception:
            return 0.0

    def grounding_check(self, question, answer, context):
        prompt = f"""
Check if the answer is grounded in the context.
Return only a number between 0 and 1.

Question: {question}
Context: {context}
Answer: {answer}
"""
        return self._call_llm(prompt)

    def precision_check(self, answer, context):
        prompt = f"""
Evaluate factual precision for the answer using the context.
Score 1.0 when the answer is fully supported and contains no hallucinated claims.
Score 0.0 when the answer is largely unsupported.
Return only a number between 0 and 1.

Context: {context}
Answer: {answer}
"""
        return self._call_llm(prompt)

    def relevancy_check(self, question, answer):
        prompt = f"""
Score how helpful, direct, and relevant the answer is for the question.
Return only a number between 0 and 1.

Question: {question}
Answer: {answer}
"""
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=20,
            )
            return self._clamp_score(response.choices[0].message.content)
        except Exception:
            return 0.0
