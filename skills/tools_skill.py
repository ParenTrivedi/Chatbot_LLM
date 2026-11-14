
class ToolsSkill:

    def __init__(self):
        self.tools = {}

    def register_tool(self, name, function, description=""):
        if not name or not callable(function):
            raise ValueError("Invalid tool registration.")

        key = name.strip().lower()

        if key in self.tools:
            raise ValueError(f"Tool already exists: {key}")

        self.tools[key] = {
            "function": function,
            "description": description
        }

    def list_tools(self):
        return [
            {
                "name": name,
                "description": details["description"]
            }
            for name, details in self.tools.items()
        ]

    def execute(self, name, **kwargs):
        key = name.strip().lower()

        if key not in self.tools:
            raise ValueError(f"Tool not available: {name}")

        return self.tools[key]["function"](**kwargs)

    def handle(self, action, name=None, **kwargs):

        if action == "list":
            return self.list_tools()

        if action == "execute":
            if not name:
                raise ValueError("Tool name is required.")

            return self.execute(name, **kwargs)

        raise ValueError(f"Unknown tools action: {action}")
