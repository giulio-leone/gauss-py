"""
08 — Graph-based DAG pipeline.

Demonstrates:
  • Building a computation graph with nodes and edges
  • Sequential DAG execution (research → write → review)
  • Fork nodes for parallel fan-out with consensus
"""

import os

from gauss import Agent, Graph, ProviderType, OPENAI_DEFAULT


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]

    # ── Create specialized agents ────────────────────────────────────
    researcher = Agent(
        name="researcher",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Research the topic thoroughly. Provide key facts and data.",
    )

    writer = Agent(
        name="writer",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Write clear, engaging content based on the research provided.",
    )

    reviewer = Agent(
        name="reviewer",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Review the content for accuracy and clarity. Suggest improvements.",
    )

    # ── 1. Linear DAG: research → write → review ────────────────────
    print("=== Linear Graph Pipeline ===\n")

    with Graph() as graph:
        graph.add_node("research", researcher, instructions="Focus on recent developments")
        graph.add_node("write", writer)
        graph.add_node("review", reviewer)
        graph.add_edge("research", "write")
        graph.add_edge("write", "review")

        result = graph.run("The future of renewable energy")

        print("Final text:", result.get("final_text", "")[:300], "...")
        print("\nPer-node outputs:")
        for node_id, output in result.get("outputs", {}).items():
            print(f"  {node_id}: {str(output)[:100]}...")

    # ── 2. Fork node (parallel fan-out) ──────────────────────────────
    print("\n=== Fork Node (Parallel) ===\n")

    analyst_tech = Agent(
        name="analyst-tech",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Analyze from a technical perspective.",
    )

    analyst_biz = Agent(
        name="analyst-biz",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Analyze from a business perspective.",
    )

    synthesizer = Agent(
        name="synthesizer",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=api_key,
        system_prompt="Combine analyses into a coherent summary.",
    )

    with Graph() as graph:
        # Parallel analysis via fork
        graph.add_fork(
            "analyze",
            [analyst_tech, analyst_biz],
            consensus="concat",
        )
        graph.add_node("synthesize", synthesizer)
        graph.add_edge("analyze", "synthesize")

        result = graph.run("Impact of large language models on software development")
        print("Synthesized:", result.get("final_text", "")[:300], "...")

    # ── Cleanup ──────────────────────────────────────────────────────
    for a in (researcher, writer, reviewer, analyst_tech, analyst_biz, synthesizer):
        a.destroy()


if __name__ == "__main__":
    main()
