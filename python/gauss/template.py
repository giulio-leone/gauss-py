"""
Prompt Templates — composable, reusable prompt construction.

Example::

    from gauss import template

    summarize = template("Summarize the following {{format}}:\\n\\n{{text}}")
    prompt = summarize(format="article", text="Lorem ipsum...")

    # Composition:
    with_tone = template("{{base}}\\n\\nUse a {{tone}} tone.")
    prompt = with_tone(base=summarize(format="article", text="..."), tone="professional")
"""

from __future__ import annotations

import re

__all__ = ["PromptTemplate", "template"]

_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class PromptTemplate:
    """A reusable prompt template with ``{{variable}}`` placeholders.

    Example::

        t = PromptTemplate("Hello {{name}}, you are {{age}} years old.")
        t(name="Alice", age="30")  # "Hello Alice, you are 30 years old."
        t.variables  # ["name", "age"]
    """

    __slots__ = ("_raw", "_variables")

    def __init__(self, template_str: str) -> None:
        self._raw = template_str
        self._variables = list(dict.fromkeys(_VAR_PATTERN.findall(template_str)))

    @property
    def raw(self) -> str:
        """The original template string."""
        return self._raw

    @property
    def variables(self) -> list[str]:
        """List of variable names in order of first appearance."""
        return list(self._variables)

    def __call__(self, **kwargs: str) -> str:
        """Render the template with the given variables.

        Raises:
            KeyError: If a required variable is missing.
        """

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in kwargs:
                raise KeyError(f"Missing template variable: {{{{{key}}}}}")
            return kwargs[key]

        return _VAR_PATTERN.sub(_replace, self._raw)

    def __repr__(self) -> str:
        return f"PromptTemplate(variables={self._variables})"


def template(template_str: str) -> PromptTemplate:
    """Create a reusable prompt template.

    Example::

        from gauss import template

        greet = template("Hello {{name}}!")
        greet(name="World")  # "Hello World!"
    """
    return PromptTemplate(template_str)


# ─── Built-in templates ────────────────────────────────────────────

summarize = template("Summarize the following {{format}} in {{style}}:\n\n{{text}}")

translate = template("Translate the following text to {{language}}:\n\n{{text}}")

code_review = template(
    "Review this {{language}} code for bugs, security issues, and best practices:"
    "\n\n```{{language}}\n{{code}}\n```"
)

classify = template(
    "Classify the following text into one of these categories: {{categories}}"
    "\n\nText: {{text}}\n\nRespond with only the category name."
)

extract = template(
    "Extract the following information from the text: {{fields}}"
    "\n\nText: {{text}}\n\nRespond as JSON."
)
