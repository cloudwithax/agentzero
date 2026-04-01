"""Task planning and analysis classes."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from capabilities import Capability, CapabilityProfile


class TaskType(Enum):
    """Types of tasks the planner can handle."""

    MATH = "math"
    RESEARCH = "research"
    CODING = "coding"
    COMPARISON = "comparison"
    GENERIC = "generic"
    ANALYSIS = "analysis"
    CREATIVE = "creative"


@dataclass
class Task:
    """A high-level task to be completed."""

    id: str
    description: str
    type: str
    requirements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single executable step in a task."""

    id: str
    task_id: str
    description: str
    operation: str
    required_capabilities: list[Capability] = field(default_factory=list)
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    max_retries: int = 3
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """A complete execution plan."""

    task_id: str
    steps: list[Step]
    estimated_complexity: float = 1.0


class TaskAnalyzer:
    """Analyzes task descriptions to determine type and requirements."""

    MATH_KEYWORDS = [
        r"\d+\s*[+\-*/]\s*\d+",
        "calculate",
        "compute",
        "sum",
        "product",
        "average",
        "mean",
        "median",
        "solve",
        "equation",
        "formula",
        "math",
        "arithmetic",
        "percentage",
        "percent",
        "ratio",
        "fraction",
        "algebra",
    ]
    RESEARCH_KEYWORDS = [
        "research",
        "find",
        "search",
        "look up",
        "investigate",
        "information about",
        "what is",
        "who is",
        "when did",
        "where is",
        "how does",
        "explain",
        "describe",
        "tell me about",
    ]
    CODING_KEYWORDS = [
        "code",
        "program",
        "function",
        "class",
        "implement",
        "write a",
        "script",
        "algorithm",
        "debug",
        "fix",
        "refactor",
        "optimize",
        "develop",
        "build",
        "create",
    ]
    COMPARISON_KEYWORDS = [
        "compare",
        "difference between",
        "vs",
        "versus",
        "better than",
        "worse than",
        "similarities",
        "differences",
        "pros and cons",
        "advantages",
        "disadvantages",
    ]
    ANALYSIS_KEYWORDS = [
        "analyze",
        "analysis",
        "review",
        "evaluate",
        "assess",
        "examine",
        "study",
        "inspect",
        "check",
        "audit",
    ]
    CREATIVE_KEYWORDS = [
        "write",
        "create",
        "generate",
        "compose",
        "draft",
        "story",
        "poem",
        "essay",
        "article",
        "content",
    ]

    def analyze(self, description: str) -> Task:
        t_type = self._detect_type(description)
        return Task(
            id=f"task_{hash(description) % 100000:05d}",
            description=description,
            type=t_type.value,
            requirements=self._extract_requirements(description),
        )

    def _detect_type(self, description: str) -> TaskType:
        desc = description.lower()
        scores = {
            TaskType.MATH: self._score(desc, self.MATH_KEYWORDS),
            TaskType.RESEARCH: self._score(desc, self.RESEARCH_KEYWORDS),
            TaskType.CODING: self._score(desc, self.CODING_KEYWORDS),
            TaskType.COMPARISON: self._score(desc, self.COMPARISON_KEYWORDS),
            TaskType.ANALYSIS: self._score(desc, self.ANALYSIS_KEYWORDS),
            TaskType.CREATIVE: self._score(desc, self.CREATIVE_KEYWORDS),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else TaskType.GENERIC

    def _score(self, text: str, keywords: list[str]) -> int:
        score = 0
        for k in keywords:
            if k.startswith(r"\d"):
                if re.search(k, text):
                    score += 2
            elif k in text:
                score += 1
        return score

    def _extract_requirements(self, description: str) -> list[str]:
        reqs = []
        if "must" in description.lower():
            reqs.append("strict_compliance")
        if "step by step" in description.lower():
            reqs.append("detailed_steps")
        return reqs


class TaskPlanner:
    """Plans task execution based on model capabilities."""

    def __init__(self, profile: CapabilityProfile):
        self.profile = profile
        self.analyzer = TaskAnalyzer()

    def plan(self, task: Task) -> Plan:
        t_type = (
            TaskType(task.type)
            if task.type in [t.value for t in TaskType]
            else TaskType.GENERIC
        )
        if t_type == TaskType.MATH:
            steps = self._plan_math(task)
        elif t_type == TaskType.RESEARCH:
            steps = self._plan_research(task)
        elif t_type == TaskType.CODING:
            steps = self._plan_coding(task)
        else:
            steps = self._plan_generic(task)
        return Plan(task_id=task.id, steps=self._constrain_steps(steps))

    def _plan_math(self, task: Task) -> list[Step]:
        return [
            Step(
                id=f"{task.id}_extract",
                task_id=task.id,
                description="Extract variables",
                operation="extract_vars",
                required_capabilities=[Capability.REASONING],
            ),
            Step(
                id=f"{task.id}_calc",
                task_id=task.id,
                description="Perform calculation",
                operation="calculate",
                required_capabilities=[Capability.REASONING],
            ),
        ]

    def _plan_research(self, task: Task) -> list[Step]:
        steps = [
            Step(
                id=f"{task.id}_decomp",
                task_id=task.id,
                description="Decompose topic",
                operation="decompose",
                required_capabilities=[Capability.REASONING],
            )
        ]
        if self.profile.has_capability(Capability.TOOL_USE):
            steps.append(
                Step(
                    id=f"{task.id}_search",
                    task_id=task.id,
                    description="Search info",
                    operation="search",
                    required_capabilities=[Capability.TOOL_USE],
                )
            )
        steps.append(
            Step(
                id=f"{task.id}_synth",
                task_id=task.id,
                description="Synthesize findings",
                operation="synthesize",
                required_capabilities=[Capability.REASONING],
            )
        )
        return steps

    def _plan_coding(self, task: Task) -> list[Step]:
        return [
            Step(
                id=f"{task.id}_analyze",
                task_id=task.id,
                description="Analyze requirements",
                operation="analyze",
                required_capabilities=[Capability.REASONING],
            ),
            Step(
                id=f"{task.id}_impl",
                task_id=task.id,
                description="Implement code",
                operation="implement",
                required_capabilities=[Capability.REASONING],
            ),
        ]

    def _plan_generic(self, task: Task) -> list[Step]:
        return [
            Step(
                id=f"{task.id}_exec",
                task_id=task.id,
                description=f"Execute {task.description}",
                operation="execute",
                required_capabilities=[Capability.REASONING],
            )
        ]

    def _constrain_steps(self, steps: list[Step]) -> list[Step]:
        return [
            s
            for s in steps
            if all(self.profile.has_capability(c) for c in s.required_capabilities)
        ] or steps[:1]

    def create_quick_plan(self, description: str) -> Plan:
        return self.plan(self.analyzer.analyze(description))
