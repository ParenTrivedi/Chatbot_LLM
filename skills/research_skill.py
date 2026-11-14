
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class ResearchSkill:

    def research(self, topic, context=None):
        if not topic or not topic.strip():
            raise ValueError("Research topic cannot be empty.")

        prompt = f"""
You are a research assistant.

Research topic: {topic}

Additional context:
{context or "None"}

Prepare a structured research response containing:

1. Research overview
2. Important concepts
3. Key findings
4. Supporting evidence available in the supplied context
5. Limitations and uncertainties
6. Further research questions
7. Conclusion

Rules:
- Do not invent facts, statistics, or references.
- Clearly identify information that cannot be verified.
- Separate established facts from interpretations.
- Do not fabricate academic citations.
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

    def handle(self, action="research", topic=None, context=None):
        if action != "research":
            raise ValueError("Unknown research action.")

        return self.research(topic, context)
