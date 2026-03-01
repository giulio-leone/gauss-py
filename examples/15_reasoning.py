"""
Example: Reasoning / Extended Thinking

Shows how to use reasoning_effort (OpenAI o-series) and
thinking_budget (Anthropic extended thinking).
"""
from gauss import Agent, OPENAI_REASONING, ANTHROPIC_PREMIUM

# OpenAI with reasoning effort (o4-mini)
reasoner = Agent(
    name="deep-thinker",
    model=OPENAI_REASONING,
    instructions="You are an expert problem solver. Think carefully.",
    reasoning_effort="high",
)

# Anthropic with extended thinking
thinker = Agent(
    name="claude-thinker",
    model=ANTHROPIC_PREMIUM,
    instructions="Analyze complex problems with deep reasoning.",
    thinking_budget=10000,
)

async def main():
    # OpenAI reasoning
    result1 = await reasoner.run("What is 27^3 + 14^3?")
    print(f"OpenAI reasoning: {result1.text}")
    
    # Anthropic extended thinking
    result2 = await thinker.run("Explain the P vs NP problem.")
    print(f"Anthropic response: {result2.text}")
    if result2.thinking:
        print(f"Thinking process: {result2.thinking[:200]}...")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
