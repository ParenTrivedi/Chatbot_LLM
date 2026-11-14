
import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "database")
MEMORY_FILE = os.path.join(MEMORY_DIR, "assistant_memory.json")


class MemorySkill:

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)

        if not os.path.exists(MEMORY_FILE):
            self._save([])

    def _load(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError("Memory file must contain a list.")

            return data

        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(f"Could not read memory: {error}") from error

    def _save(self, memories):
        temporary_file = MEMORY_FILE + ".tmp"

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(memories, file, indent=4, ensure_ascii=False)

        os.replace(temporary_file, MEMORY_FILE)

    def add(self, content):
        if not content or not content.strip():
            raise ValueError("Memory cannot be empty.")

        memories = self._load()

        memory = {
            "id": max(
                (item.get("id", 0) for item in memories),
                default=0
            ) + 1,
            "content": content.strip()
        }

        memories.append(memory)
        self._save(memories)

        return memory

    def list_memories(self):
        return self._load()

    def search(self, keyword):
        keyword = keyword.lower()

        return [
            item for item in self._load()
            if keyword in item["content"].lower()
        ]

    def delete(self, memory_id):
        memories = self._load()
        updated = [item for item in memories if item["id"] != memory_id]

        if len(updated) == len(memories):
            return False

        self._save(updated)
        return True

    def handle(self, action, content=None, keyword=None, memory_id=None):

        if action == "add":
            return self.add(content)

        if action == "list":
            return self.list_memories()

        if action == "search":
            return self.search(keyword or "")

        if action == "delete":
            return self.delete(memory_id)

        raise ValueError(f"Unknown memory action: {action}")
