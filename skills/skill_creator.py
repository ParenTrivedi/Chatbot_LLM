
import ast
import json
import os
import re
import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(BASE_DIR, "skills")
CUSTOM_DIR = os.path.join(SKILLS_DIR, "custom")
DRAFT_DIR = os.path.join(CUSTOM_DIR, "drafts")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class SkillCreator:

    def __init__(self):
        os.makedirs(DRAFT_DIR, exist_ok=True)

    def _ask_model(self, prompt):
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

    def _safe_name(self, name):
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = name.strip("_")

        if not name or not name.isidentifier():
            raise ValueError("Invalid skill name.")

        return name

    def create_instruction_skill(self, name, description):

        prompt = f"""
Create a reusable AI skill definition.

Name: {name}
Description: {description}

Return ONLY valid JSON with these fields:
- name
- description
- instructions
- example_requests

Do not include markdown fences.
Do not invent external tools.
"""

        raw = self._ask_model(prompt)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "The model did not return valid JSON."
            ) from error

        required = {"name", "description", "instructions", "example_requests"}

        if not required.issubset(result):
            raise ValueError("Generated skill definition is incomplete.")

        filename = self._safe_name(name) + ".json"
        path = os.path.join(CUSTOM_DIR, filename)

        if os.path.exists(path):
            raise FileExistsError("This custom skill already exists.")

        with open(path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        return {
            "status": "created",
            "type": "instruction",
            "path": path,
            "skill": result
        }

    def create_python_skill_draft(self, name, description):

        safe_name = self._safe_name(name)
        filename = safe_name + "_skill.py"
        path = os.path.join(DRAFT_DIR, filename)

        if os.path.exists(path):
            raise FileExistsError("A draft with this name already exists.")

        prompt = f"""
Write a Python skill module.

Skill name: {safe_name}
Purpose: {description}

Requirements:
- Create a class named Skill.
- Provide a handle(self, action, **kwargs) method.
- Use only Python standard-library imports unless necessary.
- Do not use eval or exec.
- Do not execute shell commands.
- Do not access environment variables or credentials.
- Do not delete or modify existing project databases.
- Do not make network requests.
- Validate inputs.
- Include clear error handling.
- Return only Python source code.
- Do not include markdown fences.
"""

        source = self._ask_model(prompt)

        source = re.sub(
            r"^\s*```(?:python)?\s*|\s*```\s*$",
            "",
            source.strip(),
            flags=re.IGNORECASE
        )

        try:
            ast.parse(source)
        except SyntaxError as error:
            raise ValueError(
                f"Generated Python has a syntax error: {error}"
            ) from error

        with open(path, "x", encoding="utf-8") as file:
            file.write(source + "\n")

        return {
            "status": "draft_created",
            "path": path,
            "message": (
                "Review this file before approving or activating it."
            )
        }

    def list_custom_skills(self):

        results = []

        for filename in os.listdir(CUSTOM_DIR):
            if filename.endswith(".json"):
                results.append(filename)

        drafts = [
            filename for filename in os.listdir(DRAFT_DIR)
            if filename.endswith(".py")
        ]

        return {
            "instruction_skills": results,
            "python_drafts": drafts
        }

    def handle(self, action, name=None, description=None):

        if action == "create_instruction":
            return self.create_instruction_skill(name, description)

        if action == "create_python":
            return self.create_python_skill_draft(name, description)

        if action == "list":
            return self.list_custom_skills()

        raise ValueError(f"Unknown skill creator action: {action}")
