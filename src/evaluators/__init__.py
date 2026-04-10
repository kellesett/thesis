"""
src/evaluators/__init__.py
Evaluator registry: maps evaluator_id → class, provides load_evaluator() factory.

To add a new evaluator:
  1. Create src/evaluators/<name>.py with a class inheriting BaseEvaluator
  2. Add an entry to REGISTRY below
"""
from .base import BaseEvaluator
from .surge import SurGEEvaluator, JudgeFailedError

REGISTRY: dict[str, type[BaseEvaluator]] = {
    "surge": SurGEEvaluator,
}


def load_evaluator(evaluator_id: str, config: dict, dataset) -> BaseEvaluator:
    """Instantiate an evaluator by its id."""
    cls = REGISTRY.get(evaluator_id)
    if cls is None:
        raise ValueError(
            f"Unknown evaluator id: '{evaluator_id}'. "
            f"Available: {list(REGISTRY)}"
        )
    return cls(config, dataset)
