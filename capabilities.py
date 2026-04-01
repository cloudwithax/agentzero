"""Capability detection and adaptive formatting classes."""

import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class Capability(Enum):
    """Core capabilities a model might have."""
    JSON_OUTPUT = "json_output"
    TOOL_USE = "tool_use"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    FEW_SHOT = "few_shot"
    SELF_CORRECTION = "self_correction"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class CapabilityProfile:
    """Profile of a model's capabilities."""
    capabilities: set = field(default_factory=set)
    strategy_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    exploration_rate: float = 0.2
    model_name: str = "unknown"

    def has_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def get_format_strategy(self) -> str:
        if self.has_capability(Capability.JSON_OUTPUT):
            return "json"
        elif self.has_capability(Capability.STRUCTURED_OUTPUT):
            return "structured_markdown"
        return "simple_text"

    def get_max_examples(self) -> int:
        if not self.has_capability(Capability.FEW_SHOT):
            return 1
        if not self.has_capability(Capability.LONG_CONTEXT):
            return 2
        return 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "capabilities": [c.value for c in self.capabilities],
            "format_strategy": self.get_format_strategy(),
            "max_examples": self.get_max_examples(),
        }


class CapabilityDetector:
    """Detects model capabilities through probes."""
    def __init__(self, api_call_func: Optional[Callable] = None):
        self.api_call_func = api_call_func

    def detect_capabilities(self, model_name: str = "unknown") -> CapabilityProfile:
        profile = CapabilityProfile(model_name=model_name)
        profile.capabilities.add(Capability.REASONING)
        profile.capabilities.add(Capability.FEW_SHOT)
        return profile

    def get_default_profile(self, model_name: str) -> CapabilityProfile:
        profile = CapabilityProfile(model_name=model_name)
        known_models = {
            "moonshotai/kimi-k2-instruct-0905": [Capability.JSON_OUTPUT, Capability.TOOL_USE, Capability.CHAIN_OF_THOUGHT, Capability.REASONING, Capability.LONG_CONTEXT, Capability.FEW_SHOT, Capability.SELF_CORRECTION, Capability.STRUCTURED_OUTPUT],
        }
        for known_model, caps in known_models.items():
            if known_model.lower() in model_name.lower():
                profile.capabilities = set(caps)
                return profile
        profile.capabilities = {Capability.REASONING, Capability.FEW_SHOT}
        return profile


class AdaptiveFormatter:
    """Formats prompts based on model capabilities."""
    def __init__(self, profile: CapabilityProfile):
        self.profile = profile

    def format_task_prompt(self, task_description: str, examples: Optional[List[Dict]] = None, output_schema: Optional[Dict] = None, context: Optional[Dict] = None) -> str:
        parts = [f"# Task: {task_description}", ""]
        if examples and self.profile.has_capability(Capability.FEW_SHOT):
            max_ex = self.profile.get_max_examples()
            parts.append("## Examples:\n")
            for ex in examples[:max_ex]:
                parts.extend([f"Input: {ex.get('input', '')}", f"Output: {ex.get('output', '')}", ""])
        if context:
            parts.extend(["## Context:", json.dumps(context, indent=2), ""])
        if output_schema:
            parts.append(self._get_format_instructions(output_schema, self.profile.get_format_strategy()))
        return "\n".join(parts)

    def _get_format_instructions(self, schema: Dict, format_type: str) -> str:
        if format_type == "json":
            example = {k: f"<{k}>" for k in schema.keys() if k != "format"}
            return f"## Output Format:\nRespond with valid JSON:\n```json\n{json.dumps(example, indent=2)}\n```"
        elif format_type == "structured_markdown":
            instr = ["## Output Format:\nUse this structure:\n"]
            for k in schema.keys():
                if k != "format":
                    instr.append(f"{k}: <your {k} here>")
            return "\n".join(instr)
        return f"## Output Format:\nProvide parts: {', '.join(k for k in schema.keys() if k != 'format')}"
