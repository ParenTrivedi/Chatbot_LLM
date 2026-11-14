
from skills.skill_manager import SkillManager
from skills.document_skill import DocumentSkill
from skills.research_skill import ResearchSkill
from skills.coding_skill import CodingSkill
from skills.reasoning_skill import ReasoningSkill
from skills.memory_skill import MemorySkill
from skills.tools_skill import ToolsSkill
from skills.planning_skill import PlanningSkill
from skills.skill_creator import SkillCreator


def run_test(name, function):
    print(f"\n{'=' * 50}")
    print(f"TESTING: {name}")
    print("=" * 50)

    try:
        result = function()
        print("STATUS: PASSED")
        print("RESULT:", str(result)[:500])
    except Exception as error:
        print("STATUS: FAILED")
        print(f"ERROR: {type(error).__name__}: {error}")


# Register all eight skills
manager = SkillManager()

manager.register_skill("document", DocumentSkill())
manager.register_skill("research", ResearchSkill())
manager.register_skill("coding", CodingSkill())
manager.register_skill("reasoning", ReasoningSkill())
manager.register_skill("memory", MemorySkill())
manager.register_skill("tools", ToolsSkill())
manager.register_skill("planning", PlanningSkill())
manager.register_skill("skill_creator", SkillCreator())


print("\nCHATBOT SKILL SYSTEM TEST")
print("Registered skills:", manager.list_skills())


# 1. Document skill
run_test(
    "Document Intelligence",
    lambda: manager.execute_skill("document", action="list")
)


# 2. Research skill
run_test(
    "Research Assistant",
    lambda: manager.execute_skill(
        "research",
        action="research",
        topic="Explain the basic working principle of artificial intelligence."
    )
)


# 3. Coding skill
run_test(
    "Coding Assistant",
    lambda: manager.execute_skill(
        "coding",
        action="explain",
        task="Explain this Python code.",
        code="print('Hello World')"
    )
)


# 4. Reasoning skill
run_test(
    "Reasoning Assistant",
    lambda: manager.execute_skill(
        "reasoning",
        action="solve",
        problem="A machine produces 120 components in 4 hours. "
                "How many components will it produce in 7 hours "
                "at the same rate?"
    )
)


# 5. Memory skill
run_test(
    "Memory Assistant",
    lambda: manager.execute_skill("memory", action="list")
)


# 6. Tools skill
def sample_tool(number):
    return number * 2


def test_tools():
    tool_skill = manager.get_skill("tools")
    tool_skill.register_tool(
        "double_number",
        sample_tool,
        "Doubles a number."
    )

    return tool_skill.handle(
        action="execute",
        name="double_number",
        number=10
    )


run_test("Tool Usage", test_tools)


# 7. Planning skill
run_test(
    "Planning Assistant",
    lambda: manager.execute_skill(
        "planning",
        action="plan",
        goal="Build a simple Python calculator."
    )
)


# 8. Skill Creator
run_test(
    "Skill Creator",
    lambda: manager.execute_skill(
        "skill_creator",
        action="list"
    )
)


print("\n" + "=" * 50)
print("ALL SKILL TESTS COMPLETED")
print("=" * 50)
