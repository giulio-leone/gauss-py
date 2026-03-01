"""
03 — Team with strategies (sequential & parallel).

Demonstrates:
  • Creating multiple agents with different roles
  • Building a Team with .add() and .strategy()
  • Sequential strategy: agents run in order, each sees previous output
  • Parallel strategy: agents run concurrently, results merged
"""

import os

from gauss import Agent, ProviderType, Team, OPENAI_DEFAULT


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]

    # ── Create specialized agents ────────────────────────────────────
    researcher = Agent(
        name="researcher",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are a research analyst. Provide factual, detailed research.",
    )

    writer = Agent(
        name="writer",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are a professional writer. Transform research into engaging prose.",
    )

    editor = Agent(
        name="editor",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are an editor. Polish text for clarity and conciseness.",
    )

    # ── Sequential team (pipeline) ───────────────────────────────────
    print("=== Sequential Team ===\n")
    with Team("content-pipeline") as team:
        team.add(researcher).add(writer).add(editor).strategy("sequential")
        result = team.run("Write a 100-word article about quantum computing")
        print("Final text:", result.get("finalText", ""))

    # ── Parallel team (fan-out) ──────────────────────────────────────
    print("\n=== Parallel Team ===\n")
    analyst_a = Agent(
        name="analyst-a",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Analyze from a technical perspective.",
    )

    analyst_b = Agent(
        name="analyst-b",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Analyze from a business perspective.",
    )

    with Team("analysis-team") as team:
        team.add(analyst_a).add(analyst_b).strategy("parallel")
        result = team.run("Evaluate the impact of AI on healthcare")
        print("Combined results:", result.get("finalText", ""))

        # Per-agent results
        for agent_result in result.get("results", []):
            print(f"\n  Agent: {agent_result.get('agent', 'unknown')}")
            print(f"  Output: {agent_result.get('text', '')[:100]}...")

    # ── Cleanup agents ───────────────────────────────────────────────
    for a in (researcher, writer, editor, analyst_a, analyst_b):
        a.destroy()


if __name__ == "__main__":
    main()
