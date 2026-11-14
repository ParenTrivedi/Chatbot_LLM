
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class PlanningSkill:

    def create_plan(self, goal, constraints=None):

        if not goal or not goal.strip():
            raise ValueError("Goal cannot be empty.")

        prompt = f"""
You are a project planning assistant.

Goal:
{goal}

Constraints:
{constraints or "None"}

Create a practical plan with:

1. Goal summary
2. Required resources
3. Main phases
4. Individual tasks
5. Task dependencies
6. Potential risks
7. Expected deliverables
8. Completion checklist

Do not invent deadlines or available resources.
Clearly identify information that needs confirmation.
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

    def handle(self, action="plan", goal=None, constraints=None):
        return self.create_plan(goal or action, constraints)
