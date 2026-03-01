"""
09 — Workflow with steps and dependencies.

Demonstrates:
  • Building a step-based workflow
  • Declaring step dependencies for execution ordering
  • Steps with overridden instructions and tools
"""

import os

from gauss import Agent, ProviderType, ToolDef, Workflow


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]

    # ── Create agents for each step ──────────────────────────────────
    planner = Agent(
        name="planner",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=api_key,
        system_prompt="You are a project planner. Break tasks into actionable steps.",
    )

    designer = Agent(
        name="designer",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=api_key,
        system_prompt="You are a system designer. Create technical designs from plans.",
    )

    implementer = Agent(
        name="implementer",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=api_key,
        system_prompt="You are a developer. Write implementation details from designs.",
    )

    tester = Agent(
        name="tester",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=api_key,
        system_prompt="You are a QA engineer. Create test plans for implementations.",
    )

    # ── Build workflow with dependencies ──────────────────────────────
    #   plan → design → implement
    #                  → test (depends on both design and implement)
    with Workflow() as wf:
        wf.add_step("plan", planner, instructions="Create a high-level project plan")
        wf.add_step(
            "design",
            designer,
            tools=[
                ToolDef(
                    name="lookup_patterns",
                    description="Look up common design patterns",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
        )
        wf.add_step("implement", implementer)
        wf.add_step("test", tester)

        # Declare dependencies
        wf.add_dependency("design", "plan")
        wf.add_dependency("implement", "design")
        wf.add_dependency("test", "design")
        wf.add_dependency("test", "implement")

        # Execute
        result = wf.run("Build a REST API for a todo application")

        print("Workflow completed!\n")
        for step_id, step_result in result.get("steps", {}).items():
            print(f"── Step: {step_id} ──")
            text = str(step_result)[:200]
            print(f"   {text}...")
            print()

    # ── Cleanup ──────────────────────────────────────────────────────
    for a in (planner, designer, implementer, tester):
        a.destroy()


if __name__ == "__main__":
    main()
