
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class ReasoningSkill:

    def solve(self, problem, context=None):

        if not problem or not problem.strip():
            raise ValueError("Problem cannot be empty.")

        prompt = f"""
You are a logical problem-solving assistant.

Problem:
{problem}

Context:
{context or "None"}

Approach:

1. Understand the problem.
2. Identify known information.
3. Identify missing information.
4. Break the problem into smaller parts.
5. Evaluate possible solutions.
6. Explain calculations and assumptions.
7. Present the most justified answer.
8. State uncertainty when information is insufficient.

Do not invent missing data or claim certainty without evidence.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()
        return response.json()["message"]["content"]

    def handle(self, action="solve", problem=None, context=None):
        return self.solve(problem or action, context)
