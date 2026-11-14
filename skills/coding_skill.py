
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class CodingSkill:

    def process(self, task, code=None, error=None, language="Python"):

        prompt = f"""
You are an experienced software development assistant.

Task: {task}
Programming language: {language}

Code:
{code or "No code supplied"}

Error:
{error or "No error supplied"}

Instructions:
- Understand the task before responding.
- Identify the root cause of errors when evidence is available.
- Provide corrected code when requested.
- Preserve existing functionality unless changes are required.
- Explain important modifications.
- Never claim code was executed or tested unless it actually was.
- Identify assumptions and missing information.
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

    def handle(self, action="explain", task=None, code=None,
               error=None, language="Python"):

        return self.process(
            task=task or action,
            code=code,
            error=error,
            language=language
        )
