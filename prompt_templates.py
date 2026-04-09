"""Prompt template loading and rendering.

Templates are stored as markdown files in the prompts/ directory.
Template syntax:
  - {{variable}} - Variable substitution
  - {{#if variable}}...{{/if}} - Conditional block (renders if variable is truthy and non-empty)
"""

from pathlib import Path
import re
from typing import Dict, Any, Optional

PROMPTS_DIR = Path(__file__).parent / "prompts"

_template_cache: Dict[str, str] = {}


def load_template(name: str) -> str:
    """Load a template file by name (without .md extension).

    Args:
        name: Template filename without extension (e.g., 'system_prompt')

    Returns:
        Raw template content as string
    """
    if name in _template_cache:
        return _template_cache[name]

    template_path = PROMPTS_DIR / f"{name}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    _template_cache[name] = content
    return content


def render_template(template: str, context: Dict[str, Any]) -> str:
    """Render a template with the given context.

    Handles:
      - {{variable}} substitution
      - {{#if variable}}...{{/if}} conditional blocks

    Args:
        template: Raw template string
        context: Dictionary of variables for substitution

    Returns:
        Rendered template string
    """
    result = template

    # Handle conditional blocks first: {{#if var}}...{{/if}}
    if_pattern = re.compile(r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)

    def replace_if(match: re.Match) -> str:
        var_name = match.group(1)
        content = match.group(2)
        value = context.get(var_name)
        # Render if truthy and non-empty string
        if value:
            if isinstance(value, str) and value.strip():
                return content
            elif not isinstance(value, str):
                return content
        return ""

    result = if_pattern.sub(replace_if, result)

    # Handle variable substitution: {{var}}
    var_pattern = re.compile(r"\{\{(\w+)\}\}")

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        value = context.get(var_name, "")
        if value is None:
            return ""
        return str(value)

    result = var_pattern.sub(replace_var, result)

    return result


def get_template(name: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Load and render a template by name.

    Args:
        name: Template filename without extension
        context: Optional dictionary of variables for substitution

    Returns:
        Rendered template string
    """
    template = load_template(name)
    if context:
        return render_template(template, context)
    return template


def clear_cache() -> None:
    """Clear the template cache."""
    _template_cache.clear()
