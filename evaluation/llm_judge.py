from openai import OpenAI

class LLMJudge:

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o-mini"

    def grounding_check(self, question, answer, context):
        prompt = f"""
Check if the answer is grounded in context.
Return only a number between 0 and 1.

Question: {question}
Context: {context}
Answer: {answer}
"""
        return self._call_llm(prompt)

    def relevancy_check(self, question, answer):
        prompt = f"""
Is the answer relevant and direct?
Return 1 (good) or 0 (bad).

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
                timeout=20
            )
            return response.choices[0].message.content
        except Exception:
            return "0"