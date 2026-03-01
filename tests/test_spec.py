"""Tests for AGENTS.MD and SKILL.MD parsers."""

import pytest

from gauss.spec import AgentSpec, AgentToolSpec, SkillParam, SkillSpec, SkillStep, discover_agents


# ─── AgentSpec ─────────────────────────────────────────────────────

SAMPLE_AGENTS_MD = """# My Agent

## Description
A helpful coding assistant.

## Model
gpt-4o

## Provider
openai

## Instructions
You are a coding assistant that helps with TypeScript.
Always provide examples.

## Tools
### code_review
Reviews code for quality.
```json
{"language": "string"}
```

### test_runner
Runs unit tests.

## Skills
- typescript-expert
- code-review

## Capabilities
- code-generation
- refactoring
- testing

## Environment
- NODE_ENV=production
- LOG_LEVEL=debug
"""


class TestAgentSpec:
    def test_parses_complete_agents_md(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert spec.name == "My Agent"
        assert "coding assistant" in spec.description
        assert spec.model == "gpt-4o"
        assert spec.provider == "openai"
        assert "TypeScript" in spec.instructions

    def test_parses_tools_with_parameters(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert len(spec.tools) == 2
        assert spec.tools[0].name == "code_review"
        assert spec.tools[0].parameters is not None
        assert spec.tools[1].name == "test_runner"

    def test_parses_skills_and_capabilities(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert "typescript-expert" in spec.skills
        assert "code-review" in spec.skills
        assert "code-generation" in spec.capabilities
        assert "refactoring" in spec.capabilities

    def test_parses_environment_variables(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert spec.environment.get("NODE_ENV") == "production"
        assert spec.environment.get("LOG_LEVEL") == "debug"

    def test_has_tool(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert spec.has_tool("code_review") is True
        assert spec.has_tool("nonexistent") is False

    def test_has_capability(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        assert spec.has_capability("testing") is True
        assert spec.has_capability("nonexistent") is False

    def test_minimal_agents_md(self):
        minimal = "# Minimal Agent\n\n## Description\nA simple agent."
        spec = AgentSpec.from_markdown(minimal)
        assert spec.name == "Minimal Agent"
        assert len(spec.tools) == 0
        assert len(spec.skills) == 0

    def test_frozen_dataclass(self):
        spec = AgentSpec.from_markdown(SAMPLE_AGENTS_MD)
        with pytest.raises(AttributeError):
            spec.name = "Modified"  # type: ignore

    def test_yaml_frontmatter(self):
        md = '---\nversion: "1.0"\nauthor: test\n---\n# Frontmatter Agent\n\n## Description\nWith frontmatter.'
        spec = AgentSpec.from_markdown(md)
        assert spec.name == "Frontmatter Agent"
        assert spec.metadata is not None


# ─── SkillSpec ─────────────────────────────────────────────────────

SAMPLE_SKILL_MD = """# Code Review

## Description
Reviews code for quality, correctness, and best practices.

## Steps
1. Read the code carefully
   Action: analyze_code
2. Check for common patterns
   Action: check_patterns
3. Provide feedback

## Inputs
- code (string): The code to review
- language (string, optional): The programming language

## Outputs
- feedback (string): Review feedback
- score (number, optional): Quality score 0-100
"""


class TestSkillSpec:
    def test_parses_complete_skill_md(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        assert skill.name == "Code Review"
        assert "quality" in skill.description

    def test_parses_steps_with_actions(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        assert len(skill.steps) == 3
        assert skill.steps[0].action == "analyze_code"
        assert skill.steps[2].action is None

    def test_parses_inputs_and_outputs(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        assert len(skill.inputs) == 2
        assert len(skill.outputs) == 2
        assert skill.inputs[0].name == "code"
        assert skill.inputs[0].required is True
        assert skill.inputs[1].required is False

    def test_step_count(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        assert skill.step_count == 3

    def test_required_inputs(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        assert len(skill.required_inputs) == 1
        assert skill.required_inputs[0].name == "code"

    def test_minimal_skill_md(self):
        minimal = "# Simple\n\n## Description\nA simple skill."
        skill = SkillSpec.from_markdown(minimal)
        assert skill.name == "Simple"
        assert len(skill.steps) == 0

    def test_frozen_dataclass(self):
        skill = SkillSpec.from_markdown(SAMPLE_SKILL_MD)
        with pytest.raises(AttributeError):
            skill.name = "Modified"  # type: ignore
