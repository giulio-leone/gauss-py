"""
10 — Network with supervisor-based delegation.

Demonstrates:
  • Creating a multi-agent network
  • Setting a supervisor agent
  • Delegating tasks to specific agents
  • Agent connections within the network
"""

import os

from gauss import Agent, Network, ProviderType, OPENAI_DEFAULT


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]

    # ── Create agents ────────────────────────────────────────────────
    supervisor = Agent(
        name="supervisor",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt=(
            "You are a team supervisor. Route tasks to the right specialist. "
            "Available agents: coder, analyst, writer."
        ),
    )

    coder = Agent(
        name="coder",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are an expert programmer. Write clean, production-ready code.",
    )

    analyst = Agent(
        name="analyst",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are a data analyst. Provide data-driven insights.",
    )

    writer = Agent(
        name="writer",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="You are a technical writer. Write clear documentation.",
    )

    # ── Build network ────────────────────────────────────────────────
    with Network() as net:
        # Add agents with connection declarations
        net.add_agent(supervisor)
        net.add_agent(coder, connections=["supervisor"])
        net.add_agent(analyst, connections=["supervisor"])
        net.add_agent(writer, connections=["supervisor", "coder"])

        # Set the supervisor for task routing
        net.set_supervisor("supervisor")

        # ── Delegate tasks ───────────────────────────────────────────
        print("=== Delegate to Coder ===\n")
        result = net.delegate("coder", "Write a Python function to merge two sorted lists.")
        print(result.get("text", str(result))[:300])

        print("\n=== Delegate to Analyst ===\n")
        result = net.delegate("analyst", "What are the key metrics for an e-commerce platform?")
        print(result.get("text", str(result))[:300])

        print("\n=== Delegate to Writer ===\n")
        result = net.delegate("writer", "Write API documentation for a /users endpoint.")
        print(result.get("text", str(result))[:300])

    # ── Cleanup ──────────────────────────────────────────────────────
    for a in (supervisor, coder, analyst, writer):
        a.destroy()


if __name__ == "__main__":
    main()
