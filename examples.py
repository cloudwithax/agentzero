"""Example bank and few-shot learning management."""

import json
import hashlib
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Example:
    """A single few-shot example."""
    id: str
    task_type: str
    input_text: str
    output_text: str
    weight: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def effective_score(self) -> float:
        return 0.6 * self.weight + 0.4 * self.success_rate

    def record_outcome(self, success: bool, outcome_score: Optional[float] = None):
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        if outcome_score is not None:
            alpha = 0.1
            self.weight = (1 - alpha) * self.weight + alpha * outcome_score

    def to_dict(self) -> Dict:
        return {"id": self.id, "task_type": self.task_type, "input": self.input_text, "output": self.output_text, "weight": self.weight, "usage_count": self.usage_count, "success_rate": self.success_rate, "created_at": self.created_at}


class ExampleBank:
    """Bank of few-shot examples with learned selection weights."""
    def __init__(self, exploration_rate: float = 0.1):
        self.examples: Dict[str, List[Example]] = defaultdict(list)
        self.exploration_rate = exploration_rate
        self._example_index: Dict[str, Example] = {}

    def add(self, task_type: str, input_text: str, output_text: str, weight: float = 0.5, metadata: Optional[Dict] = None) -> str:
        content_hash = hashlib.md5(f"{task_type}:{input_text}:{output_text}".encode()).hexdigest()[:12]
        example_id = f"{task_type}_{len(self.examples[task_type]):03d}_{content_hash}"
        example = Example(id=example_id, task_type=task_type, input_text=input_text, output_text=output_text, weight=weight, metadata=metadata or {})
        self.examples[task_type].append(example)
        self._example_index[example_id] = example
        return example_id

    def select(self, task_type: str, k: int = 2, exploration: Optional[float] = None, query: Optional[str] = None) -> List[Example]:
        if task_type not in self.examples or not self.examples[task_type]:
            return []
        examples = self.examples[task_type]
        if random.random() < (exploration or self.exploration_rate):
            return random.sample(examples, min(k, len(examples)))
        
        scored = []
        for ex in examples:
            score = ex.effective_score
            if query:
                q_words, i_words = set(query.lower().split()), set(ex.input_text.lower().split())
                score += 0.2 * (len(q_words & i_words) / max(len(q_words), 1))
            scored.append((ex, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:k]]

    def update(self, example_ids: List[str], outcome: float):
        for ex_id in example_ids:
            if ex_id in self._example_index:
                self._example_index[ex_id].record_outcome(outcome > 0.5, outcome)

    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.add(task_type=item.get("task_type", "generic"), input_text=item.get("input", ""), output_text=item.get("output", ""), weight=item.get("weight", 0.5), metadata=item.get("metadata", {}))
        except FileNotFoundError:
            pass

    def save_to_file(self, filepath: str):
        data = [ex.to_dict() for task_examples in self.examples.values() for ex in task_examples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class AdaptiveFewShotManager:
    """Coordinates example selection with task planning."""
    def __init__(self, example_bank: Optional[ExampleBank] = None):
        self.bank = example_bank or ExampleBank()
        self.recent_selections: List[tuple] = []

    def get_examples_for_task(self, task_type: str, query: str, max_examples: int = 3, exploration: Optional[float] = None) -> List[Dict]:
        examples = self.bank.select(task_type=task_type, k=max_examples, exploration=exploration, query=query)
        self.recent_selections.append((task_type, [ex.id for ex in examples]))
        return [{"input": ex.input_text, "output": ex.output_text, "id": ex.id} for ex in examples]

    def auto_feedback(self, task_type: str, success: bool, efficiency: float = 1.0):
        for t_type, ex_ids in reversed(self.recent_selections):
            if t_type == task_type:
                self.bank.update(ex_ids, (1.0 if success else 0.0) * efficiency)
                break

    def add_example(self, task_type: str, input_text: str, output_text: str, success_score: float = 0.5) -> str:
        return self.bank.add(task_type=task_type, input_text=input_text, output_text=output_text, weight=success_score)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "example_bank": {
                "examples_per_type": {k: len(v) for k, v in self.bank.examples.items()},
                "total_examples": sum(len(v) for v in self.bank.examples.values()),
            },
            "recent_selections": len(self.recent_selections),
        }
