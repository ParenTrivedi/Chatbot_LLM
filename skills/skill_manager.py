
class SkillManager:
    def __init__(self):
        self.skills = {}

    def register_skill(self, name, skill):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Skill name cannot be empty.")

        if not callable(getattr(skill, "handle", None)):
            raise TypeError("A skill must provide a callable handle() method.")

        key = name.strip().lower()

        if key in self.skills:
            raise ValueError(f"Skill already registered: {key}")

        self.skills[key] = skill

    def get_skill(self, name):
        return self.skills.get(name.strip().lower())

    def execute_skill(self, name, *args, **kwargs):
        skill = self.get_skill(name)

        if skill is None:
            raise ValueError(f"Skill not found: {name}")

        return skill.handle(*args, **kwargs)

    def list_skills(self):
        return list(self.skills.keys())

    def unregister_skill(self, name):
        return self.skills.pop(name.strip().lower(), None) is not None
